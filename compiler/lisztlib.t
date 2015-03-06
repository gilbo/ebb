local L = {}
package.loaded["compiler.lisztlib"] = L

local use_legion = not not rawget(_G, '_legion_env')
local use_single = not use_legion

-- Liszt types are created here
local T = require 'compiler.types'

local DataArray = use_single and
                  require('compiler.rawdata').DataArray
local LW = use_legion and require "compiler.legionwrap"

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
--local LVector    = make_prototype("LVector","vector")
local LMacro     = make_prototype("LMacro","macro")
local LUserFunc  = make_prototype("LUserFunc", "user_func")
local Kernel     = make_prototype("LKernel","kernel")

local C = require "compiler.c"
local T = require "compiler.types"
local ast = require "compiler.ast"
require "compiler.builtins"
require "compiler.relations"
local semant = require "compiler.semant"
local K = (use_single and require "compiler.kernel_single") or
          (use_legion and require "compiler.kernel_legion")

local is_vector = L.is_vector --cache lookup for efficiency

-------------------------------------------------------------------------------
--[[ LGlobals:                                                             ]]--
-------------------------------------------------------------------------------

function L.Global (typ, init)
    if not T.isLisztType(typ) or not typ:isValueType() then error("First argument to L.Global must be a Liszt expression type", 2) end
    if not T.luaValConformsToType(init, typ) then error("Second argument to L.Global must be an instance of type " .. typ:toString(), 2) end

    local s  = setmetatable({type=typ}, LGlobal)
    local tt = typ:terraType()

    if use_single then
      s.data = DataArray.New({size=1,type=tt})
      s:set(init)

    elseif use_legion then
      local cdata = T.luaToLisztVal(init, typ)
      s.data = LW.CreateFuture(typ, cdata)
    end

    return s
end

local function set_cpu_value (_type, data, val)
  if _type:isVector() then
    local v     = is_vector(val) and val or L.NewVector(_type:baseType(), val)
    local sdata = terralib.cast(&_type:terraBaseType(), data:ptr())
    for i = 0, v.N-1 do
      sdata[i] = v.data[i+1]
    end

  -- primitive is easy - just copy it over
  else
    data:ptr()[0] = _type == L.int and val - val % 1 or val
  end
end

function LGlobal:set(val)
  if not T.luaValConformsToType(val, self.type) then error("value does not conform to type of global: " .. self.type:toString(), 2) end

  if use_single then
    self.data:write_ptr(function(ptr)
        ptr[0] = T.luaToLisztVal(val, self.type)
    end)

  elseif use_legion then
    if self.data then LW.DestroyFuture(self.data) end
    local cdata = T.luaToLisztVal(val, self.type)
    self.data = LW.CreateFuture(self.type, cdata)
  end

end


function LGlobal:get()
  local value

  if use_single then
    self.data:read_ptr(function(ptr)
        value = T.lisztToLuaVal(ptr[0], self.type)
    end)

  elseif use_legion then
    value = T.lisztToLuaVal(LW.GetResultFromFuture(self.type, self.data),
                            self.type)
  end

  return value
end

function LGlobal:DataPtr()
    return self.data:ptr()
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

function L.NewUserFunc(func_ast, luaenv)
    local new_user_func = setmetatable({}, L.LUserFunc)

    local special = specialization.specialize(luaenv, func_ast)
    new_user_func.ast = special

    return new_user_func
end






