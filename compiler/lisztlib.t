local L = {}
package.loaded["compiler.lisztlib"] = L

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
local Kernel     = make_prototype("LKernel","kernel")

local C = terralib.require "compiler.c"
local T = terralib.require "compiler.types"
local ast = terralib.require "compiler.ast"
terralib.require "compiler.builtins"
local LDB = terralib.require "compiler.ldb"
local semant = terralib.require "compiler.semant"
local K = terralib.require "compiler.kernel"
--L.LDB = LDB

local is_vector = L.is_vector --cache lookup for efficiency

-------------------------------------------------------------------------------
--[[ LGlobals:                                                             ]]--
-------------------------------------------------------------------------------
function L.NewGlobal (typ, init)
    if not T.isLisztType(typ) or not typ:isValueType() then error("First argument to L.NewGlobal must be a Liszt expression type", 2) end
    if not T.luaValConformsToType(init, typ) then error("Second argument to L.NewGlobal must be an instance of type " .. typ:toString(), 2) end

    local s  = setmetatable({type=typ}, LGlobal)
    local tt = typ:terraType()
    s.data   = terralib.cast(&tt, C.malloc(terralib.sizeof(tt)))
    s:setTo(init)
    return s
end


function LGlobal:setTo(val)
   if not T.luaValConformsToType(val, self.type) then error("value does not conform to type of global: " .. self.type:toString(), 2) end
      if self.type:isVector() then
          local v     = is_vector(val) and val or L.NewVector(self.type:baseType(), val)
          local sdata = terralib.cast(&self.type:terraBaseType(), self.data)
          for i = 0, v.N-1 do
              sdata[i] = v.data[i+1]
          end
    -- primitive is easy - just copy it over
    else
        self.data[0] = self.type == L.int and val - val % 1 or val
    end
end

function LGlobal:value()
    if self.type:isPrimitive() then return self.data[0] end

    local ndata = {}
    local sdata = terralib.cast(&self.type:terraBaseType(), self.data)
    for i = 1, self.type.N do ndata[i] = sdata[i-1] end

    return L.NewVector(self.type:baseType(), ndata)
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
    local q = ast.QuoteExpr:DeriveFrom(field)
    q.exp   = semant.check({}, w)
    q.node_type = q.exp.node_type
    return q
end)







