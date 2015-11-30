local Pre = {}
package.loaded["ebb.src.prelude"] = Pre

local use_legion = not not rawget(_G, '_legion_env')
local use_single = not use_legion

local LE, legion_env, LW
if use_legion then
  LE            = rawget(_G, '_legion_env')
  legion_env    = LE.legion_env[0]
  LW            = require 'ebb.src.legionwrap'
end

local VERBOSE = rawget(_G, 'EBB_LOG_EBB')

-------------------------------------------------------------------------------

local ProcConstant = {}
ProcConstant.__index = ProcConstant
ProcConstant.__tostring = function(proc) return proc.str end
Pre.CPU = setmetatable({ str = 'CPU' }, ProcConstant)
Pre.GPU = setmetatable({ str = 'GPU' }, ProcConstant)
Pre.default_processor = Pre.CPU
-- global signal from the launch script
if rawget(_G, 'EBB_USE_GPU_SIGNAL') then
  Pre.default_processor = Pre.GPU
end
local function SetDefaultProcessor(proc)
  Pre.default_processor = proc
end
local function GetDefaultProcessor()
  return Pre.default_processor
end
Pre.SetDefaultProcessor = SetDefaultProcessor
Pre.GetDefaultProcessor = GetDefaultProcessor

-------------------------------------------------------------------------------

local Global      = {}
Global.__index    = Global
local function is_global(obj) return getmetatable(obj) == Global end
Pre.is_global     = is_global

local Constant    = {}
Constant.__index  = Constant
local function is_constant(obj) return getmetatable(obj) == Constant end
Pre.is_constant   = is_constant

local Macro       = {}
Macro.__index     = Macro
local function is_macro(obj) return getmetatable(obj) == Macro end
Pre.is_macro      = is_macro


-------------------------------------------------------------------------------

local DataArray = use_single and
                  require('ebb.src.rawdata').DataArray

local C   = require 'ebb.src.c'
local T   = require 'ebb.src.types'

-------------------------------------------------------------------------------
--[[ Globals:                                                              ]]--
-------------------------------------------------------------------------------

function Pre.Global (typ, init)
  if not T.istype(typ) or not typ:isvalue() then
    error("First argument to Global() must be an Ebb value type", 2)
  end
  if not T.luaValConformsToType(init, typ) then
    error("Second argument to Global() must be an "..
          "instance of type " .. tostring(typ), 2)
  end

  local s  = setmetatable({_type=typ}, Global)

  if use_single then
    local tt = typ:terratype()
    rawset(s, '_data', DataArray.New({size=1,type=tt}))
    s:set(init)

  elseif use_legion then
    s:set(init)
  end

  return s
end

function Global:__newindex(fieldname,value)
  error("Cannot assign members to Global object", 2)
end

function Global:set(val)
  if not T.luaValConformsToType(val, self._type) then
    error("value does not conform to type of global: "..
          tostring(self._type), 2)
  end

  if VERBOSE then
    local data_deps = "Ebb LOG: function global-set accesses"
    data_deps = data_deps .. " global " .. tostring(self) ..
                " in phase EXCLUSIVE ,"
    print(data_deps)
  end

  if use_single then
    local ptr = self._data:open_write_ptr()
    ptr[0] = T.luaToEbbVal(val, self._type)
    self._data:close_write_ptr()

  elseif use_legion then
    local typ    = self._type
    local tt     = typ:terratype()
    local blob   = C.safemalloc( tt )
    blob[0]      = T.luaToEbbVal(val, typ)
    local future = LW.legion_future_from_buffer(legion_env.runtime,
                                                blob,
                                                terralib.sizeof(tt))
    if self._data then
      LW.legion_future_destroy(self._data)
    end
    rawset(self, '_data', future)
  end

end

function Global:get()
  local value

  if VERBOSE then
    local data_deps = "Ebb LOG: function global-get accesses"
    data_deps = data_deps .. " global " .. tostring(self) .. " in phase READ ,"
    print(data_deps)
  end

  if use_single then
    local ptr = self._data:open_read_ptr()
    value = T.ebbToLuaVal(ptr[0], self._type)
    self._data:close_read_ptr()

  elseif use_legion then
    local tt = self._type:terratype()
    local result = LW.legion_future_get_result(self._data)
    local rptr   = terralib.cast(&tt, result.value)
    value = T.ebbToLuaVal(rptr[0], self._type)
    LW.legion_task_result_destroy(result)
  end

  return value
end

function Global:_Raw_DataPtr()
    return self._data:_raw_ptr()
end

function Global:Type()
  return self._type
end

-------------------------------------------------------------------------------
--[[ Constants:                                                            ]]--
-------------------------------------------------------------------------------

local function deep_copy(tbl)
    if type(tbl) ~= 'table' then return tbl
    else
        local cpy = {}
        for i=1,#tbl do cpy[i] = deep_copy(tbl[i]) end
        return cpy
    end
end

function Pre.Constant (typ, init)
    if not T.istype(typ) or not typ:isvalue() then
        error("First argument to Constant() must be an "..
              "Ebb value type", 2)
    end
    if not T.luaValConformsToType(init, typ) then
        error("Second argument to Constant() must be a "..
              "value of type " .. tostring(typ), 2)
    end


    local c = setmetatable({_type=typ, _value=deep_copy(init)}, Constant)
    return c
end

function Constant:__newindex(fieldname,value)
  error("Cannot assign members to Constant object", 2)
end

function Constant:get()
  return deep_copy(self._value)
end

function Constant:Type()
  return deep_copy(self._value)
end

-------------------------------------------------------------------------------
--[[ LMacros:                                                              ]]--
-------------------------------------------------------------------------------
function Pre.Macro(generator)
    return setmetatable({genfunc=generator}, Macro)    
end

function Macro:__newindex(fieldname,value)
  error("Cannot assign members to Macro object", 2)
end

