-- The MIT License (MIT)
-- 
-- Copyright (c) 2015 Stanford University.
-- All rights reserved.
-- 
-- Permission is hereby granted, free of charge, to any person obtaining a
-- copy of this software and associated documentation files (the "Software"),
-- to deal in the Software without restriction, including without limitation
-- the rights to use, copy, modify, merge, publish, distribute, sublicense,
-- and/or sell copies of the Software, and to permit persons to whom the
-- Software is furnished to do so, subject to the following conditions:
-- 
-- The above copyright notice and this permission notice shall be included
-- in all copies or substantial portions of the Software.
-- 
-- THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
-- IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
-- FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
-- AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
-- LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
-- FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
-- DEALINGS IN THE SOFTWARE.

local F = {}
package.loaded["ebb.src.functions"] = F

local use_legion = not not rawget(_G, '_legion_env')
local use_single = not use_legion


local LE, legion_env, LW
if use_legion then
  LE            = rawget(_G, '_legion_env')
  legion_env    = LE.legion_env[0]
  LW            = require 'ebb.src.legionwrap'
end

local VERBOSE = rawget(_G, 'EBB_LOG_EBB')


local T                 = require 'ebb.src.types'
local Stats             = require 'ebb.src.stats'

local Pre               = require 'ebb.src.prelude'
local R                 = require 'ebb.src.relations'
local specialization    = require 'ebb.src.specialization'
local semant            = require 'ebb.src.semant'
local phase             = require 'ebb.src.phase'
local stencil           = require 'ebb.src.stencil'


F._INTERNAL_DEV_OUTPUT_PTX = false

-------------------------------------------------------------------------------

local Function    = {}
Function.__index  = Function
F.Function        = Function
local function is_function(obj) return getmetatable(obj) == Function end
F.is_function     = is_function

local UFVersion   = {}
UFVersion.__index = UFVersion
F.UFVersion       = UFVersion
local function is_version(obj) return getmetatable(obj) == UFVersion end
F.is_version      = is_version


require 'ebb.src.ufversions'

-------------------------------------------------------------------------------
--[[ UserFunc:                                                             ]]--
-------------------------------------------------------------------------------


function F.NewFunction(func_ast, luaenv)
  local special = specialization.specialize(luaenv, func_ast)
  
  local ufunc = setmetatable({
    _decl_ast     = special,
    _versions     = {}, -- the versions table is nested
    _name         = special.id,
  }, Function)

  return ufunc
end

function F.NewUFVersion(ufunc, signature)
  local version = setmetatable({
    _ufunc          = ufunc,
    _compile_timer  = Stats.NewTimer(ufunc._name..'_compile_time'),
    _exec_timer     = Stats.NewTimer(ufunc._name..'_execution_time'),
  }, UFVersion)

  for k,v in pairs(signature) do
    version['_'..k] = v
  end

  return version
end

UFVersion._total_function_launch_count =
  Stats.NewCounter('total_function_launch_count')
function F.PrintStats()
  UFVersion._total_function_launch_count:Print()
end

-- Use the following to produce
-- deterministic order of table entries
-- From the Lua Documentation
local function pairs_sorted(tbl, compare)
  local arr = {}
  for k in pairs(tbl) do table.insert(arr, k) end
  table.sort(arr, compare)

  local i = 0
  local iter = function() -- iterator
    i = i + 1
    if arr[i] == nil then return nil
    else return arr[i], tbl[arr[i]] end
  end
  return iter
end

function Function:_get_typechecked(calldepth, relset, strargs)
  -- lookup based on relation, not subset
  local relation = relset
  if R.is_subset(relset) then relation = relset:Relation() end
  -- build lookup key string
  local keystr = tostring(relset)
  for _,arg in ipairs(strargs) do   keystr = keystr..','..arg   end
  -- and perform lookup
  local lookup = self._versions[keystr]
  if lookup then return lookup end

  -- Otherwise, the Lookup failed, so...

  -- make a safe copy that we can explicitly type annotate
  local aname_ast     = self._decl_ast:alpha_rename()

  -- process the first argument's type annotation.  Consistent? Present?
  local annotation    = aname_ast.ptypes[1]
  if annotation then
    local arel = annotation.relation
    if arel ~= relation then
      error('The supplied relation did not match the parameter '..
            'annotation:\n  '..relation:Name()..' vs. '..arel:Name(),
            calldepth)
    end
  else
    -- add an annotation if none was present
    aname_ast.ptypes[1] = T.key(relation)
  end

  -- process the remaining arguments' type annotations.
  for i,str in ipairs(strargs) do
    local annotation = aname_ast.ptypes[i+1]
    if annotation then
      error('Secondary string arguments to functions should be '..
            'untyped arguments', calldepth)
    end
    aname_ast.ptypes[i+1] = T.internal(str)
  end

  -- now actually type and phase check
  local typed_ast       = semant.check( aname_ast )
  local phase_results   = phase.phasePass( typed_ast )
  local field_accesses  = stencil.stencilPass( typed_ast )

  -- cache the type/phase-checking computations
  local cached = {
    typed_ast       = typed_ast,
    phase_results   = phase_results,
    field_accesses  = field_accesses,
    versions        = {},
  }
  self._versions[keystr] = cached

  return cached
end

local function get_ufunc_version(ufunc, typeversion_table, relset, params)
  params = params or {}

  local proc = params.location or Pre.default_processor

  -- To lookup the version we want, we need to construct a signature
  local sig = {
    proc      = proc,
  }
  sig.relation  = relset
  if R.is_subset(relset) then
    sig.relation  = relset:Relation()
    sig.subset    = relset
  end
  if proc == Pre.GPU then   sig.blocksize = params.blocksize or 64  end
  if sig.relation:isElastic() then  sig.is_elastic = true  end

  -- and convert that signature into a string for lookup
  local str_sig = ''
  for k,v in pairs_sorted(sig) do
    str_sig = str_sig .. k .. '=' .. tostring(v) .. ';'
  end

  -- do the actual lookup
  local version = typeversion_table.versions[str_sig]
  if version then return version end

  -- if the lookup failed, then we need to construct a new
  -- version matching this signature
  version = F.NewUFVersion(ufunc, sig)
  version._typed_ast        = typeversion_table.typed_ast
  version._phase_data       = typeversion_table.phase_results
  version._field_accesses   = typeversion_table.field_accesses

  -- and make sure to cache it
  typeversion_table.versions[str_sig] = version

  return version
end

-- this will cause typechecking to fire
function Function:GetVersion(relset, ...)
  return self:_Get_Version(3, relset, ...)
end
function Function:GetAllVersions()
  local vs = {}
  for _,typeversion in pairs(self._versions) do
    for _,version in pairs(typeversion.versions) do
      table.insert(vs, version)
    end
  end
  return vs
end
function Function:_Get_Version(calldepth, relset, ...)
  if not (R.is_subset(relset) or R.is_relation(relset)) then
    error('Functions must be executed over a relation or subset, but '..
          'argument was neither: '..tostring(relset), calldepth)
  end

  -- unpack direct arguments and/or launch parameters
  local args    = {...}
  local params  = {}
  if type(args[#args]) == 'table' then
    params = args[#args]
    args[#args] = nil
  end

  -- check that number of arguments matches, allowing for the
  -- extra first argument in the function signature that is a
  -- key for the relation being mapped over
  local narg_expected = #self._decl_ast.params - 1
  if narg_expected ~= #args then
    error('Function was expecting '..tostring(narg_expected)..
          ' arguments, but got '..tostring(#args), calldepth)
  end
  -- Also, right now we restrict all secondary arguments to be strings
  for i,a in ipairs(args) do
    if type(a) ~= 'string' then
      error('Argument '..tostring(i)..' was expected to be a string; '..
            'Secondary arguments to functions mapped over relations '..
            'must be strings.', calldepth)
    end
  end
  if self._decl_ast.exp then
    error('Functions executed over relations should not return values',
          calldepth)
  end

  -- get the appropriately typed version of the function
  -- and a collection of all the versions associated with it...
  local typeversion = self:_get_typechecked(calldepth+1, relset, args)

  -- now we either retreive or construct the appropriate function version
  local version = get_ufunc_version(self, typeversion, relset, params)

  return version
end

function Function:Compile(relset, ...)
  local version = self:_Get_Version(3, relset, ...)
  version:Compile()
end

function Function:doForEach(relset, ...)
  self:_doForEach(relset, ...)
end
function Function:_doForEach(relset, ...)
  local version = self:_Get_Version(4, relset, ...)

  version:Execute()
end


function Function:getCompileTime()
  local versions  = self:GetAllVersions()
  local sumtime   = Stats.NewTimer('')
  for _,vs in ipairs(versions) do
    sumtime = sumtime + vs._compile_timer
  end
  sumtime:setName(self._name..'_compile_time')
  return sumtime
end
function Function:getExecutionTime()
  local versions  = self:GetAllVersions()
  local sumtime   = Stats.NewTimer('')
  for _,vs in ipairs(versions) do
    sumtime = sumtime + vs._exec_timer
  end
  sumtime:setName(self._name..'_execution_time')
  return sumtime
end


