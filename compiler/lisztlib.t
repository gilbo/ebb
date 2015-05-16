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
  legion_env    = LE.legion_env:get()
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
--local LVector    = make_prototype("LVector","vector")
local LMacro     = make_prototype("LMacro","macro")
local LUserFunc  = make_prototype("LUserFunc", "function")
local Kernel     = make_prototype("LKernel","kernel")

local C = require "compiler.c"
local T = require "compiler.types"
local ast = require "compiler.ast"
require "compiler.builtins"
require "compiler.relations"
local semant = require "compiler.semant"
require "compiler.kernel"

local is_vector = L.is_vector --cache lookup for efficiency

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
    local blob   = global(tt)
    -- ensure a byte for byte copy
    (blob:getpointer())[0] = T.luaToLisztVal(val, typ)
    local future = LW.legion_future_from_buffer(legion_env.runtime,
                                                blob:getpointer(),
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
    local new_user_func = setmetatable({}, L.LUserFunc)

    local special = specialization.specialize(luaenv, func_ast)
    new_user_func.ast = special

    return new_user_func
end

function L.LUserFunc:ForEachOver(relset, params)
  if #self.ast.params ~= 1 or self.ast.exp then
    error('In order to execute a function over a relation or subset, '..
          'the function must have exactly 1 argument and no return value', 3)
  end

  local relation = relset
  if L.is_subset(relset) then relation = relset:Relation() end

  -- otherwise, try caching a kernel based on the relation
  if not self.kernel_cache then self.kernel_cache = {} end
  local cached_kernel = self.kernel_cache[relation]
  if not cached_kernel then
    cached_kernel = L.NewKernelFromFunction(self, relation)
    self.kernel_cache[relation] = cached_kernel
  end

  cached_kernel(relset, params)
end






