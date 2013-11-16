local T = terralib.require('compiler/types')
local t, Type = T.t, T.Type

local L = {}
local LDB = terralib.require('include/ldb')
L.LDB = LDB

local DECL = terralib.require('include/decl')


local C = DECL.C

-- terra type of a field that refers to another relation
local REF_TYPE    = t.uint
-- terra type of an orientation field
local ORIENT_TYPE = t.uint8


-------------------------------------------------------------------------------
--[[ Export Liszt types:                                                   ]]--
-------------------------------------------------------------------------------
L.int    = t.int
L.uint   = t.uint
L.float  = t.float
L.double = t.double
L.bool   = t.bool
L.vector = t.vector


-------------------------------------------------------------------------------
--[[ Liszt object prototypes:                                              ]]--
-------------------------------------------------------------------------------

-- export object testing routines
L.is_relation = DECL.is_relation
L.is_field    = DECL.is_field
L.is_scalar   = DECL.is_scalar
L.is_vector   = DECL.is_vector
L.is_macro    = DECL.is_macro

local is_vector = L.is_vector

-- local aliases for objects
local LScalar = DECL.LScalar
local LVector = DECL.LVector
local LMacro  = DECL.LMacro

--[[
- An LRelation contains its size and fields as members.  The _index member
- refers to an array of the compressed row values for the index field.

- An LField stores its fieldname, type, an array of data, and a pointer
- to another LRelation if the field itself represents relational data.
--]]


-------------------------------------------------------------------------------
--[[ LScalars:                                                             ]]--
-------------------------------------------------------------------------------
function L.NewScalar (typ, init)
    if not T.Type.isLisztType(typ) or not typ:isValueType() then error("First argument to L.NewScalar must be a Liszt expression type", 2) end
    if not T.luaValConformsToType(init, typ) then error("Second argument to L.NewScalar must be an instance of type " .. typ:toString(), 2) end

    local s  = setmetatable({type=typ}, LScalar)
    local tt = typ:terraType()
    s.data   = terralib.cast(&tt, C.malloc(terralib.sizeof(tt)))
    s:setTo(init)
    return s
end


function LScalar:setTo(val)
   if not T.luaValConformsToType(val, self.type) then error("value does not conform to scalar type " .. self.type:toString(), 2) end
      if self.type:isVector() then
          local v     = is_vector(val) and val or L.NewVector(self.type:baseType(), val)
          local sdata = terralib.cast(&self.type:terraBaseType(), self.data)
          for i = 0, v.N-1 do
              sdata[i] = v.data[i+1]
          end
    -- primitive is easy - just copy it over
    else
        self.data[0] = self.type == t.int and val - val % 1 or val
    end
end

function LScalar:value()
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
    if not (T.Type.isLisztType(dt) and dt:isPrimitive()) then
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

    return setmetatable({N=N, type=t.vector(dt,N), data=data}, LVector)
end

function LVector:__codegen ()
    local v1, v2, v3 = self.data[1], self.data[2], self.data[3]
    local v4, v5, v6 = self.data[4], self.data[5], self.data[6]
    local btype = self.type:terraBaseType()

    if     self.N == 2 then
        return `vectorof(btype, v1, v2)
    elseif self.N == 3 then
        return `vectorof(btype, v1, v2, v3)
    elseif self.N == 4 then
        return `vectorof(btype, v1, v2, v3, v4)
    elseif self.N == 5 then
        return `vectorof(btype, v1, v2, v3, v4, v5)
    elseif self.N == 6 then
        return `vectorof(btype, v1, v2, v3, v4, v5, v6)
    end

    local s = symbol(self.type:terraType())
    local t = symbol()
    local q = quote
        var [s]
        var [t] = [&btype](&s)
    end

    for i = 1, self.N do
        local val = self.data[i]
        q = quote 
            [q] 
            @[t] = [val]
            t = t + 1
        end
    end
    return quote [q] in [s] end
end

function LVector.__add (v1, v2)
    if not is_vector(v1) or not is_vector(v2) then
        error("Cannot add non-vector type to vector", 2)
    elseif v1.N ~= v2.N then
        error("Cannot add vectors of differing lengths", 2)
    elseif v1.type == t.bool or v2.type == t.bool then
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

    local tm = t.float
    if v.type == int and a % 1 == 0 then tm = t.int end

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
    return L.NewVector(t.float, data)
end

function LVector.__mod (v1, a2)
    if is_vector(a2) then error("Cannot modulus by a vector", 2) end
    local data = {}
    for i = 1, v1.N do
        data[i] = v1.data[i] % a2
    end
    local tp = T.type_meet(v1.type:baseType(), t.float)
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



return L
