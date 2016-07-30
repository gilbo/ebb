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

local use_exp = rawget(_G,'EBB_USE_EXPERIMENTAL_SIGNAL')
local use_single = not use_exp

local T                 = require 'ebb.src.types'
local Stats             = require 'ebb.src.stats'
local Util              = require 'ebb.src.util'

local Pre               = require 'ebb.src.prelude'
local R                 = require 'ebb.src.relations'
local specialization    = require 'ebb.src.specialization'
local semant            = require 'ebb.src.semant'
local phase             = require 'ebb.src.phase'
local stencil           = require 'ebb.src.stencil'

F._INTERNAL_DEV_OUTPUT_PTX = false

local function shallowcopy_table(tbl)
  local x = {}
  for k,v in pairs(tbl) do x[k] = v end
  return x
end

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

function Function:setname(name)
  if type(name) ~= 'string' then error('expected string argument', 2) end
  if self._typed_at_least_once then
    error('cannot re-name a function after it has been compiled once', 2)
  end
  self._name = name
  self._decl_ast.id = name
end

local ufunc_version_id = 1
function F.NewUFVersion(ufunc, signature)
  local version = setmetatable({
    _ufunc          = ufunc,
    _compile_timer  = Stats.NewTimer(ufunc._name..'_compile_time'),
    _exec_timer     = Stats.NewTimer(ufunc._name..'_execution_time'),
    _name           = ufunc._name .. '_ufv'..ufunc_version_id
  }, UFVersion)
  ufunc_version_id = ufunc_version_id + 1

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


--[[
Takes in params {
  relset,
  phase_results,
  field_accesses
}
Returns {
  field_use,
  field_accesses,  -- TODO: combine field_use and field_accesses
  insert_data,
  delete_data,
  global_use,
  global_reductions,
}
--]]
function GetAllFieldAndGlobalUses(params)
  local relset    = params.relset
  local relation  = R.is_subset(relset) and relset:Relation() or relset


  -- VERIFY SET OF FIELDS IN PHASE_RESULTS AND FIELD_ACCESSES MATCH
  if use_exp then
    for f, _ in pairs(params.phase_results.field_use) do
      assert(params.field_accesses[f], "ERROR: field " .. f:Name() ..
             " in field uses but not in field accesses.")
    end
    for f, _ in pairs(params.field_accesses) do
      assert(params.phase_results.field_use[f], "ERROR: field " .. f:Name() ..
             " in field accesses but not in field use.")
    end
  end

  local data = {
    field_use         = shallowcopy_table(params.phase_results.field_use),
    field_accesses    = shallowcopy_table(params.field_accesses),
    global_use        = shallowcopy_table(params.phase_results.global_use),
    global_reductions = {},
  }

  -- BOOL MASKS
  if relation:isElastic() then
    if use_exp then error("EXPERIMENTAL UNSUPPORTED ELASTIC") end
    local use_deletes = not not params.phase_results.deletes
    data.field_use[relation._is_live_mask] = phase.PhaseType.New {
      centered  = true,
      read      = true,
      write     = use_deletes,
    }
    if use_exp then
      data.field_accesses[relset._is_live_mask] =
        stencil.NewCenteredAccessPattern {
          field = relset._is_live_mask,
          read  = true,
          write = use_deletes,
        }
    end
  end
  if R.is_subset(relset) and relset._boolmask  then
    data.field_use[relset._boolmask] = phase.PhaseType.New {
      centered  = true,
      read      = true,
    }
    if use_exp then
      data.field_accesses[relset._boolmask] =
        stencil.NewCenteredAccessPattern {
          field = relset._boolmask,
          read  = true,
          write = false,
        }
    end
  end

  -- INSERTS : Here or in UFVersions?
  if params.phase_results.inserts then
    -- max 1 insert allowed right now
    local insert_rel, ast_nodes = next(params.phase_results.inserts)
    data.insert_data = {
      relation    = insert_rel,
      record_type = ast_nodes[1].record_type,
      write_idx   = Pre.Global(T.uint64, 0),
    }
    -- also need to support reductions?
    data.global_use[data.insert_data.write_idx] = phase.PhaseType.New {
      reduceop    = '+',
    }

    for _,f in ipairs(insert_rel._fields) do
      assert(data.field_use[f] == nil, 'trying to add duplicate field')
      data.field_use[f] = phase.PhaseType.New {
        centered = false,
        write    = true,
      }
    end
    data.field_use[insert_rel._is_live_mask] = phase.PhaseType.New {
      centered = false,
      write    = true,
    }
  end
  -- DELETES : Here or in UFVersions?
  if params.phase_results.deletes then
    -- max 1 delete allowed right now
    local del_rel = next(params.phase_results.deletes)
    data.delete_data = {
      relation  = del_rel,
      n_deleted = Pre.Global(T.uint64, 0)
    }
    -- also need to support reductions?
    data.global_use[data.delete_data.n_deleted] = phase.PhaseType.New {
      reduceop    = '+',
    }
  end

  return data
end

local get_ufunc_typetable =
Util.memoize_from(2, function(calldepth, ufunc, relset, ...)
  calldepth = calldepth+1 -- account for the memoization wrapper
  -- ... are string arguments to function call

  local relation      = R.is_subset(relset) and relset:Relation() or relset
  -- make a safe copy that we can explicitly type annotate
  local aname_ast     = ufunc._decl_ast:alpha_rename()

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
  for i,str in ipairs({...}) do
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
  local field_accesses  = use_exp and stencil.stencilPass( typed_ast ) or {}

  local f_g_uses      = GetAllFieldAndGlobalUses {
    relset         = relset,
    phase_results  = phase_results,
    field_accesses = field_accesses,
  }

  return {
    typed_ast       = typed_ast,
    field_use       = f_g_uses.field_use,
    field_accesses  = f_g_uses.field_accesses,
    insert_data     = f_g_uses.insert_data,
    delete_data     = f_g_uses.delete_data,
    global_use      = f_g_uses.global_use,
    versions        = terralib.newlist(),
  }
end)

function Function:_get_typechecked(calldepth, relset, strargs)
  return get_ufunc_typetable(calldepth+1, self, relset, unpack(strargs))
end

local get_cached_ufversion = Util.memoize_named({
  'ufunc', 'typtable', 'relation', 'proc', 'subset',
  'blocksize'
},
function(sig)
  -- Make a copy of tables for each version in case
  local version             = F.NewUFVersion(sig.ufunc, sig)
  local typtable            = sig.typtable
  version._typed_ast        = typtable.typed_ast
  version._field_use        = shallowcopy_table(typtable.field_use)
  version._field_accesses   = shallowcopy_table(typtable.field_accesses)
  version._insert_data      = typtable.insert_data and
                              shallowcopy_table(typtable.insert_data)
  version._delete_data      = typtable.delete_data and
                              shallowcopy_table(typtable.delete_data)
  version._global_use       = shallowcopy_table(typtable.global_use)
  typtable.versions:insert(version)
  return version
end)

local function get_ufunc_version(ufunc, typeversion_table, relset, params)
  params          = params or {}
  local proc      = params.location or Pre.default_processor
  local relation  = R.is_subset(relset) and relset:Relation() or relset

  return get_cached_ufversion {
    ufunc           = ufunc,
    typtable        = typeversion_table,
    relation        = relation,
    subset          = R.is_subset(relset) and relset or nil,
    proc            = proc,
    blocksize       = proc == Pre.GPU and (params.blocksize or 64) or nil,
    is_elastic      = relation:isElastic(),
  }
end

-- NOTE: THESE CALLS ARE DISABLED DUE TO LEGION DESIGN
--        SHOULD THEY BE RE-EXPOSED IN ANOTHER FORM ???
-- this will cause typechecking to fire
--function Function:GetVersion(relset, ...)
--  return self:_Get_Version(3, relset, ...)
--end
--function Function:GetAllVersions()
--  local vs = {}
--  for _,typeversion in pairs(self._versions) do
--    for _,version in pairs(typeversion.versions) do
--      table.insert(vs, version)
--    end
--  end
--  return vs
--end
local function get_func_call_params_from_args(...)
  local N = select('#',...)
  local last_arg = N > 0 and select(N,...) or nil
  if type(last_arg) == 'table' then return last_arg
                               else return {} end
end
function Function:_Get_Type_Version_Table(calldepth, relset, ...)
  self._typed_at_least_once = true
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

  return typeversion
end

-- NOTE: SEE NOTE ABOVE ; DISABLED DUE TO LEGION INTERFACE
--function Function:Compile(relset, ...)
--  local version = self:_Get_Version(3, relset, ...)
--  version:Compile()
--end

function Function:doForEach(relset, ...)
  self:_doForEach(relset, ...)
end
function Function:_doForEach(relset, ...)
  local params      = get_func_call_params_from_args(...)
  local typeversion = self:_Get_Type_Version_Table(4, relset, ...)
  
  -- now we either retrieve or construct the appropriate function version
  local version = get_ufunc_version(self, typeversion, relset, params)

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

function Function:_TESTING_GetFieldAccesses(relset, ...)
  local typeversion = self:_Get_Type_Version_Table(4, relset, ...)
  return typeversion.field_accesses -- these have the stencils in them
end
