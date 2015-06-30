local L = {}
package.loaded["compiler.lisztlib"] = L

local use_legion = not not rawget(_G, '_legion_env')
local use_single = not use_legion

-- Liszt types are created here
local T = require 'compiler.types'

local DataArray = use_single and
                  require('compiler.rawdata').DataArray

local LE, legion_env, LW
if use_legion then
  LE            = rawget(_G, '_legion_env')
  legion_env    = LE.legion_env[0]
  LW            = require 'compiler.legionwrap'
end
-------------------------------------------------------------------------------
--[[ Liszt modules:                                                        ]]--
-------------------------------------------------------------------------------

function L.require( str )
    local loaded_module = require( str )
    return loaded_module
end

-------------------------------------------------------------------------------
--[[ Liszt Constants:                                                      ]]--
-------------------------------------------------------------------------------

local ProcConstant = {}
ProcConstant.__index = ProcConstant
ProcConstant.__tostring = function(proc) return proc.str end
L.CPU = setmetatable({ str = 'CPU' }, ProcConstant)
L.GPU = setmetatable({ str = 'GPU' }, ProcConstant)
L.default_processor = L.CPU
-- global signal from the launch script
if rawget(_G, 'LISZT_USE_GPU_SIGNAL') then
  L.default_processor = L.GPU
end

-------------------------------------------------------------------------------
--[[ Liszt object prototypes:                                              ]]--
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

local C = require "compiler.c"
local T = require "compiler.types"
local ast = require "compiler.ast"
require "compiler.builtins"
require "compiler.relations"
require "compiler.ufunc"

local semant  = require "compiler.semant"
local phase   = require "compiler.phase"

-------------------------------------------------------------------------------
--[[ LGlobals:                                                             ]]--
-------------------------------------------------------------------------------

function L.Global (typ, init)
    if not T.isLisztType(typ) or not typ:isValueType() then error("First argument to L.Global must be a Liszt expression type", 2) end
    if not T.luaValConformsToType(init, typ) then error("Second argument to L.Global must be an instance of type " .. typ:toString(), 2) end

    local s  = setmetatable({type=typ}, LGlobal)

    if use_single then
      local tt = typ:terraType()
      s.data = DataArray.New({size=1,type=tt})
      s:set(init)

    elseif use_legion then
      s:set(init)
    end

    return s
end

function LGlobal:set(val)
  if not T.luaValConformsToType(val, self.type) then error("value does not conform to type of global: " .. self.type:toString(), 2) end

  if use_single then
    self.data:write_ptr(function(ptr)
        ptr[0] = T.luaToLisztVal(val, self.type)
    end)

  elseif use_legion then
    local typ    = self.type
    local tt     = typ:terraType()
    local blob   = C.safemalloc( tt )
    blob[0]      = T.luaToLisztVal(val, typ)
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

  if use_single then
    self.data:read_ptr(function(ptr)
        value = T.lisztToLuaVal(ptr[0], self.type)
    end)

  elseif use_legion then
    local tt = self.type:terraType()
    local result = LW.legion_future_get_result(self.data)
    local rptr   = terralib.cast(&tt, result.value)
    value = T.lisztToLuaVal(rptr[0], self.type)
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
    return self.data:ptr()
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
    if not T.isLisztType(typ) or not typ:isValueType() then
        error("First argument to L.Constant must be a "..
              "Liszt expression type", 2)
    end
    if not T.luaValConformsToType(init, typ) then
        error("Second argument to L.Constant must be a "..
              "value of type " .. typ:toString(), 2)
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

local specialization = require('compiler.specialization')

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
    _ufunc    = ufunc,
  }, UFVersion)

  for k,v in pairs(signature) do
    version['_'..k] = v
  end

  return version
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

function L.LUserFunc:_get_typechecked(relset)
  -- lookup based on relation
  local relation = relset
  if L.is_subset(relset) then relation = relset:Relation() end
  local lookup = self._versions[relation]
  if lookup then return lookup end

  -- Otherwise, the Lookup failed, so...

  -- check that this function makes sense to use as top-level entry
  if #self._decl_ast.params ~= 1 or self._decl_ast.exp then
    error('In order to execute a function over a relation or subset, the '..
          'function must have exactly 1 argument and no return value', 4)
  end

  -- make a safe copy that we can explicitly type annotate
  local aname_ast     = self._decl_ast:alpha_rename()

  -- check the annotation for consistency with the argument
  local annotation    = aname_ast.ptypes[1]
  if annotation then
    local arel = annotation.relation
    if arel ~= relation then
      error('The supplied relation did not match the parameter '..
            'annotation:\n  '..relation:Name()..' vs. '..arel:Name(), 4)
    end
  else
    -- add an annotation if none was present
    aname_ast.ptypes[1] = L.key(relation)
  end

  -- now actually type and phase check
  local typed_ast     = semant.check( aname_ast )
  local phase_results = phase.phasePass( typed_ast )

  -- cache the computation
  local cached = {
    typed_ast       = typed_ast,
    phase_results   = phase_results,
    versions        = {},
  }
  self._versions[relation] = cached

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
  local typed_ast     = typeversion_table.typed_ast
  local phase_results = typeversion_table.phase_results
  version:_Compile(typed_ast, phase_results)

  -- and make sure to cache it
  typeversion_table.versions[str_sig] = version

  return version
end

EXEC_TIMER = 0
function L.LUserFunc:doForEach(relset, params)
  self:_doForEach(relset, params)
end
function L.LUserFunc:_doForEach(relset, params)
  if #self._decl_ast.params ~= 1 or self._decl_ast.exp then
  end
  if not (L.is_subset(relset) or L.is_relation(relset)) then
    error('Functions must be executed over a relation or subset, but '..
          'argument was neither: '..tostring(relset), 3)
  end

  -- get the appropriately typed version of the function
  -- and a collection of all the versions associated with it...
  local typeversion = self:_get_typechecked(relset)

  -- now we either retreive or construct the appropriate function version
  local version = get_ufunc_version(self, typeversion, relset, params)

  --local preexectime = terralib.currenttimeinseconds()
  version:Execute()
  --EXEC_TIMER = EXEC_TIMER + (terralib.currenttimeinseconds() - preexectime)
end






