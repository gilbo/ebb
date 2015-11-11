local L = {}
package.loaded["ebblib"] = L

local use_legion = not not rawget(_G, '_legion_env')
local use_single = not use_legion

-- Ebb types are created here
local T = require 'ebb.src.types'

local DataArray = use_single and
                  require('ebb.src.rawdata').DataArray

local LE, legion_env, LW
if use_legion then
  LE            = rawget(_G, '_legion_env')
  legion_env    = LE.legion_env[0]
  LW            = require 'ebb.src.legionwrap'
end

local VERBOSE = rawget(_G, 'EBB_LOG_EBB')

-------------------------------------------------------------------------------
--[[ Copy from Type Module:                                                ]]--
-------------------------------------------------------------------------------

L.is_type = T.istype

for _,name in ipairs({

  }) do
  L[name] = T[name]
end
L.int           = T.int
L.uint          = T.uint
L.uint64        = T.uint64
L.bool          = T.bool
L.float         = T.float
L.double        = T.double

L.vector        = T.vector 
L.matrix        = T.matrix 
L.key           = T.key 
L.record        = T.record 
L.internal      = T.internal 
L.query         = T.query 
L.error         = T.error

for _,tchar in ipairs({ 'i', 'f', 'd', 'b' }) do
  for n=2,4 do
    n = tostring(n)
    L['vec'..n..tchar] = T['vec'..n..tchar]
    L['mat'..n..tchar] = T['mat'..n..tchar]
    for m=2,4 do
      local m = tostring(m)
      L['mat'..n..'x'..m..tchar] = T['mat'..n..'x'..m..tchar]
    end
  end
end

-------------------------------------------------------------------------------
--[[ Ebb Constants:                                                        ]]--
-------------------------------------------------------------------------------

local ProcConstant = {}
ProcConstant.__index = ProcConstant
ProcConstant.__tostring = function(proc) return proc.str end
L.CPU = setmetatable({ str = 'CPU' }, ProcConstant)
L.GPU = setmetatable({ str = 'GPU' }, ProcConstant)
L.default_processor = L.CPU
-- global signal from the launch script
if rawget(_G, 'EBB_USE_GPU_SIGNAL') then
  L.default_processor = L.GPU
end

-------------------------------------------------------------------------------
--[[ Ebb object prototypes:                                                ]]--
-------------------------------------------------------------------------------
local function make_prototype(objname,name)
  local tb = {}
  tb.__index = tb
  L["is_"..name] = function(obj) return getmetatable(obj) == tb end
  L[objname] = tb
  return tb
end
local LRelation  = make_prototype("LRelation","relation")
local LField     = make_prototype("LField","field")
local LSubset    = make_prototype("LSubset","subset")
local LIndex     = make_prototype("LIndex","index")
local LGlobal    = make_prototype("LGlobal","global")
local LConstant  = make_prototype("LConstant","constant")
local LMacro     = make_prototype("LMacro","macro")
local LUserFunc  = make_prototype("LUserFunc","function")
local UFVersion  = make_prototype("UFVersion","version")

local C = require "ebb.src.c"
local T = require "ebb.src.types"
local ast = require "ebb.src.ast"
require "ebb.src.builtins"
require "ebb.src.relations"
require "ebb.src.ufversions"

local semant  = require "ebb.src.semant"
local phase   = require "ebb.src.phase"
local Stats   = require "ebb.src.stats"

-------------------------------------------------------------------------------
--[[ LGlobals:                                                             ]]--
-------------------------------------------------------------------------------

function L.Global (typ, init)
  if not T.istype(typ) or not typ:isvalue() then
    error("First argument to L.Global must be an Ebb value type", 2)
  end
  if not T.luaValConformsToType(init, typ) then
    error("Second argument to L.Global must be an "..
          "instance of type " .. tostring(typ), 2)
  end

  local s  = setmetatable({type=typ}, LGlobal)

  if use_single then
    local tt = typ:terratype()
    s.data = DataArray.New({size=1,type=tt})
    s:set(init)

  elseif use_legion then
    s:set(init)
  end

  return s
end

function LGlobal:set(val)
  if not T.luaValConformsToType(val, self.type) then error("value does not conform to type of global: " .. tostring(self.type), 2) end

  if VERBOSE then
    local data_deps = "Ebb LOG: function global-set accesses"
    data_deps = data_deps .. " global " .. tostring(self) .. " in phase EXCLUSIVE ,"
    print(data_deps)
  end

  if use_single then
    local ptr = self.data:open_write_ptr()
    ptr[0] = T.luaToEbbVal(val, self.type)
    self.data:close_write_ptr()

  elseif use_legion then
    local typ    = self.type
    local tt     = typ:terratype()
    local blob   = C.safemalloc( tt )
    blob[0]      = T.luaToEbbVal(val, typ)
    local future = LW.legion_future_from_buffer(legion_env.runtime,
                                                blob,
                                                terralib.sizeof(tt))
    if self.data then
      LW.legion_future_destroy(self.data)
    end
    self.data    = future
  end

end

function LGlobal:get()
  local value

  if VERBOSE then
    local data_deps = "Ebb LOG: function global-get accesses"
    data_deps = data_deps .. " global " .. tostring(self) .. " in phase READ ,"
    print(data_deps)
  end

  if use_single then
    local ptr = self.data:open_read_ptr()
    value = T.ebbToLuaVal(ptr[0], self.type)
    self.data:close_read_ptr()

  elseif use_legion then
    local tt = self.type:terratype()
    local result = LW.legion_future_get_result(self.data)
    local rptr   = terralib.cast(&tt, result.value)
    value = T.ebbToLuaVal(rptr[0], self.type)
    LW.legion_task_result_destroy(result)
  end

  return value
end

--function LGlobal:SetData(data)
--  self.data = data
--end

--function LGlobal:Data()
--  return self.data
--end

--function LGlobal:SetOffset(offset)
--  self.offset = 0
--end

--function LGlobal:Offset()
--  return self.offset
--end

function LGlobal:DataPtr()
    return self.data:_raw_ptr()
end

function LGlobal:Type()
  return self.type
end

-------------------------------------------------------------------------------
--[[ LConstants:                                                           ]]--
-------------------------------------------------------------------------------

local function deep_copy(tbl)
    if type(tbl) ~= 'table' then return tbl
    else
        local cpy = {}
        for i=1,#tbl do cpy[i] = deep_copy(tbl[i]) end
        return cpy
    end
end

function L.Constant (typ, init)
    if not T.istype(typ) or not typ:isvalue() then
        error("First argument to L.Constant must be an "..
              "Ebb value type", 2)
    end
    if not T.luaValConformsToType(init, typ) then
        error("Second argument to L.Constant must be a "..
              "value of type " .. tostring(typ), 2)
    end


    local c = setmetatable({type=typ, value=deep_copy(init)}, LConstant)
    return c
end

function L.LConstant:get()
  return deep_copy(self.value)
end

-------------------------------------------------------------------------------
--[[ LMacros:                                                              ]]--
-------------------------------------------------------------------------------
function L.NewMacro(generator)
    return setmetatable({genfunc=generator}, LMacro)    
end

local specialization = require('ebb.src.specialization')

-------------------------------------------------------------------------------
--[[ LUserFunc:                                                            ]]--
-------------------------------------------------------------------------------

function L.NewUserFunc(func_ast, luaenv)
  local special = specialization.specialize(luaenv, func_ast)
  
  local ufunc = setmetatable({
    _decl_ast     = special,
    _versions     = {}, -- the versions table is nested
    _name         = special.id,
  }, LUserFunc)

  return ufunc
end

function L.NewUFVersion(ufunc, signature)
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
function L.PrintStats()
  UFVersion._total_function_launch_count:print()
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

function L.LUserFunc:_get_typechecked(calldepth, relset, strargs)
  -- lookup based on relation, not subset
  local relation = relset
  if L.is_subset(relset) then relation = relset:Relation() end
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
    aname_ast.ptypes[1] = L.key(relation)
  end

  -- process the remaining arguments' type annotations.
  for i,str in ipairs(strargs) do
    local annotation = aname_ast.ptypes[i+1]
    if annotation then
      error('Secondary string arguments to functions should be '..
            'untyped arguments', calldepth)
    end
    aname_ast.ptypes[i+1] = L.internal(str)
  end

  -- now actually type and phase check
  local typed_ast     = semant.check( aname_ast )
  local phase_results = phase.phasePass( typed_ast )

  -- cache the type/phase-checking computations
  local cached = {
    typed_ast       = typed_ast,
    phase_results   = phase_results,
    versions        = {},
  }
  self._versions[keystr] = cached

  return cached
end

local function get_ufunc_version(ufunc, typeversion_table, relset, params)
  params = params or {}

  local proc = params.location or L.default_processor

  -- To lookup the version we want, we need to construct a signature
  local sig = {
    proc      = proc,
  }
  sig.relation  = relset
  if L.is_subset(relset) then
    sig.relation  = relset:Relation()
    sig.subset    = relset
  end
  if proc == L.GPU then   sig.blocksize = params.blocksize or 64  end
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
  version = L.NewUFVersion(ufunc, sig)
  version._typed_ast  = typeversion_table.typed_ast
  version._phase_data = typeversion_table.phase_results

  -- and make sure to cache it
  typeversion_table.versions[str_sig] = version

  return version
end

-- this will cause typechecking to fire
function L.LUserFunc:GetVersion(relset, ...)
  self:_Get_Version(3, relset, ...)
end
function L.LUserFunc:GetAllVersions()
  local vs = {}
  for _,typeversion in pairs(self._versions) do
    for _,version in pairs(typeversion.versions) do
      table.insert(vs, version)
    end
  end
  return vs
end
function L.LUserFunc:_Get_Version(calldepth, relset, ...)
  if not (L.is_subset(relset) or L.is_relation(relset)) then
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

function L.LUserFunc:Compile(relset, ...)
  local version = self:_Get_Version(3, relset, ...)
  version:Compile()
end

function L.LUserFunc:doForEach(relset, ...)
  self:_doForEach(relset, ...)
end
function L.LUserFunc:_doForEach(relset, ...)
  local version = self:_Get_Version(4, relset, ...)

  version:Execute()
end


function L.LUserFunc:getCompileTime()
  local versions  = self:GetAllVersions()
  local sumtime   = Stats.NewTimer('')
  for _,vs in ipairs(versions) do
    sumtime = sumtime + vs._compile_timer
  end
  sumtime:setName(self._name..'_compile_time')
  return sumtime
end
function L.LUserFunc:getExecutionTime()
  local versions  = self:GetAllVersions()
  local sumtime   = Stats.NewTimer('')
  for _,vs in ipairs(versions) do
    sumtime = sumtime + vs._exec_timer
  end
  sumtime:setName(self._name..'_execution_time')
  return sumtime
end



