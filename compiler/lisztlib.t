local L = {}
package.loaded["compiler.lisztlib"] = L

local use_legion = rawget(_G, '_legion')
local use_direct = not use_legion

-- Liszt types are created here
local T = terralib.require 'compiler.types'

local DataArray = use_direct and
                  terralib.require('compiler.rawdata').DataArray
local Lg = use_legion and terralib.require "compiler.legion_data"

-- Use the following to produce
-- deterministic order of table entries
-- From the Lua Documentation
function pairs_sorted(tbl, compare)
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

-------------------------------------------------------------------------------
--[[ Liszt modules:                                                        ]]--
-------------------------------------------------------------------------------

function L.require( str )
    local loaded_module = terralib.require( str )
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
local LVector    = make_prototype("LVector","vector")
local LMacro     = make_prototype("LMacro","macro")
local LUserFunc  = make_prototype("LUserFunc", "user_func")
local Kernel     = make_prototype("LKernel","kernel")

local C = terralib.require "compiler.c"
local T = terralib.require "compiler.types"
local ast = terralib.require "compiler.ast"
terralib.require "compiler.builtins"
terralib.require "compiler.relations"
terralib.require "compiler.serialization"
local semant = terralib.require "compiler.semant"
local Ksingle = terralib.require "compiler.kernel_single"

local is_vector = L.is_vector --cache lookup for efficiency

-------------------------------------------------------------------------------
--[[ LGlobals:                                                             ]]--
-------------------------------------------------------------------------------
function L.NewGlobal (typ, init)
    if not T.isLisztType(typ) or not typ:isValueType() then error("First argument to L.NewGlobal must be a Liszt expression type", 2) end
    if not T.luaValConformsToType(init, typ) then error("Second argument to L.NewGlobal must be an instance of type " .. typ:toString(), 2) end

    local s  = setmetatable({type=typ}, LGlobal)
    local tt = typ:terraType()

    s.data = DataArray.New({size=1,type=tt})
    s:set(init)
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

    self.data:write_ptr(function(ptr)
        if self.type:isVector() then
            if not L.is_vector(val) then
                val = L.NewVector(self.type:baseType(), val)
            end
            for i=0, val.N-1 do
                ptr[0].d[i] = val.data[i+1]
            end
        else
            ptr[0] = val
        end
    end)
end


function LGlobal:get()
    local value

    self.data:read_ptr(function(ptr)
        if self.type:isPrimitive() then
            value = ptr[0]
        else
            value = {}
            for i=0, self.type.N-1 do
                value[i+1] = ptr[0].d[i]
            end
            value = L.NewVector(self.type:baseType(), value)
        end
    end)

    return value
end

function LGlobal:DataPtr()
    return self.data:ptr()
end

-------------------------------------------------------------------------------
--[[ LVectors:                                                             ]]--
-------------------------------------------------------------------------------
function L.NewVector(dt, init)
    if not (T.isLisztType(dt) and dt:isPrimitive()) then
        error("First argument to L.NewVector() should "..
              "be a primitive Liszt type", 2)
    end
    if not is_vector(init) and #init == 0 then
        error("Second argument to L.NewVector should either be "..
              "an LVector or an array", 2)
    end
    local N = is_vector(init) and init.N or #init
    if not T.luaValConformsToType(init, L.vector(dt, N)) then
        error("Second argument to L.NewVector() does not "..
              "conform to specified type", 2)
    end

    local data = {}
    if is_vector(init) then init = init.data end
    for i = 1, N do
        -- convert to integer if necessary
        data[i] = dt == L.int and init[i] - init[i] % 1 or init[i] 
    end

    return setmetatable({N=N, type=L.vector(dt,N), data=data}, LVector)
end

function LVector.__add (v1, v2)
    if not is_vector(v1) or not is_vector(v2) then
        error("Cannot add non-vector type to vector", 2)
    elseif v1.N ~= v2.N then
        error("Cannot add vectors of differing lengths", 2)
    elseif v1.type == L.bool or v2.type == L.bool then
        error("Cannot add boolean vectors", 2)
    end

    local data = { }
    local tp = T.type_meet(v1.type:baseType(), v2.type:baseType())

    for i = 1, #v1.data do
        data[i] = v1.data[i] + v2.data[i]
    end
    return L.NewVector(tp, data)
end

function LVector.__sub (v1, v2)
    if not is_vector(v1) then
        error("Cannot subtract vector from non-vector type", 2)
    elseif not is_vector(v2) then
        error("Cannot subtract non-vector type from vector", 2)
    elseif v1.N ~= v2.N then
        error("Cannot subtract vectors of differing lengths", 2)
    elseif v1.type == bool or v2.type == bool then
        error("Cannot subtract boolean vectors", 2)
    end

    local data = { }
    local tp = T.type_meet(v1.type:baseType(), v2.type:baseType())

    for i = 1, #v1.data do
        data[i] = v1.data[i] - v2.data[i]
    end

    return L.NewVector(tp, data)
end

function LVector.__mul (a1, a2)
    if is_vector(a1) and is_vector(a2) then
        error("Cannot multiply two vectors", 2)
    end
    local v, a
    if is_vector(a1) then   v, a = a1, a2
    else                    v, a = a2, a1 end

    if     v.type:isLogical()  then
        error("Cannot multiply a non-numeric vector", 2)
    elseif type(a) ~= 'number' then
        error("Cannot multiply a vector by a non-numeric type", 2)
    end

    local tm = L.float
    if v.type == int and a % 1 == 0 then tm = L.int end

    local data = {}
    for i = 1, #v.data do
        data[i] = v.data[i] * a
    end
    return L.NewVector(tm, data)
end

function LVector.__div (v, a)
    if     is_vector(a)    then error("Cannot divide by a vector", 2)
    elseif v.type:isLogical()  then error("Cannot divide a non-numeric vector", 2)
    elseif type(a) ~= 'number' then error("Cannot divide a vector by a non-numeric type", 2)
    end

    local data = {}
    for i = 1, #v.data do
        data[i] = v.data[i] / a
    end
    return L.NewVector(L.float, data)
end

function LVector.__mod (v1, a2)
    if is_vector(a2) then error("Cannot modulus by a vector", 2) end
    local data = {}
    for i = 1, v1.N do
        data[i] = v1.data[i] % a2
    end
    local tp = T.type_meet(v1.type:baseType(), L.float)
    return L.NewVector(tp, data)
end

function LVector.__unm (v1)
    if v1.type:isLogical() then error("Cannot negate a non-numeric vector", 2) end
    local data = {}
    for i = 1, #v1.data do
        data[i] = -v1.data[i]
    end
    return L.NewVector(v1.type, data)
end

function LVector.__eq (v1, v2)
    if v1.N ~= v2.N then return false end
    for i = 1, v1.N do
        if v1.data[i] ~= v2.data[i] then return false end
    end
    return true
end


-------------------------------------------------------------------------------
--[[ LMacros:                                                              ]]--
-------------------------------------------------------------------------------
function L.NewMacro(generator)
    return setmetatable({genfunc=generator}, LMacro)    
end

L.Where = L.NewMacro(function(field,key)
    if field == nil or key == nil then
        error("Where expects 2 arguments")
    end
    local w = ast.Where:DeriveFrom(field)
    w.field = field
    w.key   = key
    local q = ast.Quote:DeriveFrom(field)
    q.code  = semant.check({}, w)
    q.node_type = q.code.node_type
    return q
end)

local specialization = terralib.require('compiler.specialization')

function L.NewUserFunc(func_ast, luaenv)
    local new_user_func = setmetatable({}, L.LUserFunc)

    local special = specialization.specialize(luaenv, func_ast)
    new_user_func.ast = special

    return new_user_func
end






