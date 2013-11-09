local R = terralib.require('runtime/mesh')
local O = terralib.require('runtime/liszt')
local T = terralib.require('compiler/types')
local t, Type = T.t, T.Type

local L = {}

local C = terralib.includecstring [[
    #include <stdlib.h>
    #include <string.h>
    #include <stdio.h>
    #include <math.h>
]]

local RTYPE = t.uint  -- terra type of a field that refers to another relation
local OTYPE = t.uint8 -- terra type of an orientation field


--------------------------------------------------------------------------------
--[[ Export Liszt types:                                                    ]]--
--------------------------------------------------------------------------------
L.int    = t.int
L.uint   = t.uint
L.float  = t.float
L.bool   = t.bool
L.vector = t.vector


--------------------------------------------------------------------------------
--[[ Liszt object prototypes:                                               ]]--
--------------------------------------------------------------------------------
local function make_prototype(tb)
    tb.__index = tb
    return tb
end

--[[
- An LRelation contains its size and fields as members.  The _index member
- refers to an array of the compressed row values for the index field.

- An LField stores its fieldname, type, an array of data, and a pointer
- to another LRelation if the field itself represents relational data.
--]]

--------------------------------------------------------------------------------
--[[ Field prototype:                                                       ]]--
--------------------------------------------------------------------------------
local LRelation = make_prototype {kind=T.Type.kinds.relation}
local LField    = make_prototype {kind=T.Type.kinds.field}
local LScalar   = make_prototype {kind=T.Type.kinds.scalar}
local LVector   = make_prototype {kind=T.Type.kinds.vector}
local LMacro    = make_prototype {kind=T.Type.kinds.macro}


--------------------------------------------------------------------------------
--[[ LRelation methods                                                      ]]--
--------------------------------------------------------------------------------
function L.NewRelation(size, debugname)
    return setmetatable( {
        _size      = size,
        _fields    = terralib.newlist(),
        _debugname = debugname or "anon"
    },
    LRelation)
end

-- prevent user from modifying the lua table
function LRelation:__newindex(fieldname,value)
    error("Cannot assign members to LRelation object (did you mean to call self:NewField?)")
end 

local function is_relation (obj) return getmetatable(obj) == LRelation end
local function is_macro (obj) return getmetatable(obj) == LMacro end

local function isValidFieldType (typ)
    return is_relation(typ) or is_macro(typ) or T.Type.isLisztType(typ) and typ:isExpressionType()
end

function LRelation:NewField (name, typ)    
    if not isValidFieldType(typ) then
        error("NewField expects a Liszt type, Relation, or Macro as the 2nd argument", 2)
    end

    local f    = setmetatable({}, LField)
    if is_relation(typ) then
        f.relation = typ
        f.type = RTYPE
        f.macro = nil
    elseif is_macro(typ) then
        f.relation = nil
        f.type = LMacro
        f.macro = typ
    else
        f.relation = nil
        f.type = typ
        f.macro = nil
    end
    f.name     = name
    f.table    = self

    rawset(self, name, f)
    self._fields:insert(f)
    return f
end

function LRelation:LoadIndexFromMemory(name,row_idx)
    assert(self._index == nil)
    local f = self[name]
    assert(f ~= nil)
    assert(f.data == nil)
    assert(f.relation ~= nil) -- index field must be a relational type

    local ttype  = f.type:terraType()
    local tsize  = terralib.sizeof(ttype)
    local nbytes = (f.relation._size + 1) * tsize

    local mem = terralib.cast(&ttype,C.malloc(nbytes))
    rawset(self, "_index", mem)
    local memT = terralib.typeof(row_idx)
    assert(memT == &ttype)

    C.memcpy(self._index,row_idx,nbytes)
    f.data = terralib.cast(&ttype,C.malloc(self._size*tsize))
    for i = 0, f.relation._size - 1 do
        local b = self._index[i]
        local e = self._index[i+1]
        for j = b, e - 1 do
            f.data[j] = i
        end
    end
end

function LRelation:dump()
    print(self._debugname, "size: ".. tostring(self._size))
    for i,f in ipairs(self._fields) do
        f:dump()
    end
end


--------------------------------------------------------------------------------
--[[ Field methods:                                                         ]]--
--------------------------------------------------------------------------------
local terra copy_bytes (dest : &uint8, src : &uint8, length : uint, size : uint, stride : uint, offset : uint)
    src = src + offset

    -- dont potentially copy past the length of the source array:
    var copy_len : int
    if stride < size then copy_len = stride else copy_len = size end

    for i = 0, length do
        C.memcpy(dest,src,copy_len)
        src  = src  + stride
        dest = dest + size
    end
end

-- specify stride, offset in bytes, default stride reads memory contiguously and default offset reads from mem ptr directly
function LField:LoadFromMemory (mem, stride, offset)
    local ttype = self.type:terraType()
    local tsize = terralib.sizeof(ttype)

    -- Terra vectors are sized at a power of two bytes, whereas arrays are just N * sizeof(basetype)
    -- so, if we are storing arrays as terra vectors, tsize is >= the size of the data for one field
    -- entry in a contiguous array.
    local dsize = ttype:isvector() and terralib.sizeof(ttype.type[ttype.N]) or tsize

    if not stride then stride = dsize end
    if not offset then offset = 0     end

    assert(stride >= dsize)
    assert(self.data == nil)

    local nbytes = self.table._size * tsize
    local bytes  = C.malloc(nbytes)
    self.data    = terralib.cast(&ttype,bytes)

    -- cast mem to void* to avoid automatic conversion issues for pointers to non-primitive types
    mem = terralib.cast(&opaque, mem)

    -- If the array is laid out contiguously in memory, just do a memcpy
    -- otherwise, read with a stride
    if (stride == tsize) then
        C.memcpy(self.data,mem,nbytes)
    else
        copy_bytes(bytes,mem,self.table._size, tsize, stride, offset)
    end
end

function LField:LoadFromCallback (callback)
    local bytes  = C.malloc(self.table._size * terralib.sizeof(self.type:terraType()))
    self.data    = terralib.cast(&self.type:terraType(),bytes)

    for i = 0, self.table._size-1 do
        callback(self.data + i, i)
    end
end    

function LField:dump()
    print(self.name..":")
    if not self.data then
        print("...not initialized")
        return
    end

    local N = self.table._size
    if (self.type:isVector()) then
        for i = 0, N-1 do
            local s = ''
            for j = 0, self.type.N-1 do
                s = s .. tostring(self.data[i][j]) .. ' '
            end
            print("", i, s)
        end
    else
        for i = 0, N-1 do
            print("",i, self.data[i])
        end
    end
end


--------------------------------------------------------------------------------
--[[ LScalars:                                                              ]]--
--------------------------------------------------------------------------------
function L.NewScalar (typ, init)
    if not T.Type.isLisztType(typ) or not typ:isExpressionType() then error("First argument to L.NewScalar must be a Liszt expression type", 2) end
    if not T.conformsToType(init, typ) then error("Second argument to L.NewScalar must be an instance of type " .. typ:toString(), 2) end

    local s  = setmetatable({type=typ}, LScalar)
    local tt = typ:terraType()
    s.data   = terralib.cast(&tt, C.malloc(terralib.sizeof(tt)))
    s:setTo(init)
    return s
end


function LScalar:setTo(val)
   if not T.conformsToType(val, self.type) then error("value does not conform to scalar type " .. self.type:toString(), 2) end
      if self.type:isVector() then
          local v     = T.Type.isVector(val) and val or L.NewVector(self.type:baseType(), val)
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


--------------------------------------------------------------------------------
--[[ LVectors:                                                              ]]--
--------------------------------------------------------------------------------
function L.NewVector(dt, init)
    if not T.Type.isLisztType(dt) or not dt:isPrimitive() then error("First argument to L.NewVector() should be a primitive Liszt type", 2) end
    if not T.Type.isVector(init) and #init == 0 then error("Second argument to L.NewVector should be either an LVector or an array", 2) end
    local N = T.Type.isVector(init) and init.N or #init
    if not T.conformsToType(init, L.vector(dt, N)) then
        error("Second argument to L.NewVector() does not conform to specified type", 2)
    end

    local data = {}
    if T.Type.isVector(init) then init = init.data end
    for i = 1, N do
        data[i] = dt == L.int and init[i] - init[i] % 1 or init[i] -- convert to integer if necessary
    end

    return setmetatable({N=N, type=t.vector(dt,N), data=data}, LVector)
end

function LVector:__codegen ()
    local v1, v2, v3, v4, v5, v6 = self.data[1], self.data[2], self.data[3], self.data[4], self.data[5], self.data[6]
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
    if not Type.isVector(v1) or not Type.isVector(v2) then
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
    if not Type.isVector(v1) then
        error("Cannot subtract vector from non-vector type", 2)
    elseif not Type.isVector(v2) then
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
    if Type.isVector(a1) and Type.isVector(a2) then error("Cannot multiply two vectors", 2) end
    local v, a
    if Type.isVector(a1) then v, a = a1, a2 else v, a = a2, a1 end

    if     v.type:isLogical()  then error("Cannot multiply a non-numeric vector", 2)
    elseif type(a) ~= 'number' then error("Cannot multiply a vector by a non-numeric type", 2)
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
    if     Type.isVector(a)    then error("Cannot divide by a vector", 2)
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
    if Type.isVector(a2) then error("Cannot modulus by a vector", 2) end
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


--------------------------------------------------------------------------------
--[[ LMacros:                                                               ]]--
--------------------------------------------------------------------------------
function L.NewMacro(generator)
    return setmetatable({genfunc=generator}, LMacro)    
end


-------------------------------------------------------------------------------
--[[ Code that loads all relations from mesh file formats:                 ]]--
-------------------------------------------------------------------------------
local topo_elems = { 'vertices', 'edges', 'faces', 'cells' }

-- other mesh relations - we have a separate table for each of these relations
local mesh_rels_new = {
    {global = 'verticesofvertex', name = "vtov", orientation = false, t1 = "vertices", t2 = "vertices", n1 = "v1",      n2 = "v2"},
    {global = 'edgesofvertex',    name = "vtoe", orientation = true,  t1 = "vertices", t2 = "edges",    n1 = "vertex",  n2 = "edge"},
    {global = 'facesofvertex',    name = "vtof", orientation = false, t1 = "vertices", t2 = "faces",    n1 = "vertex",  n2 = "face"},
    {global = 'cellsofvertex',    name = "vtoc", orientation = false, t1 = "vertices", t2 = "cells",    n1 = "vertex",  n2 = "cell"},
    {global = 'facesofedge',      name = "etof", orientation = true,  t1 = "edges",    t2 = "faces",    n1 = "edge",    n2 = "face"},
    {global = 'cellsofedge',      name = "etoc", orientation = false, t1 = "edges",    t2 = "cells",    n1 = "edge",    n2 = "cell"},
    {global = 'verticesofface',   name = "ftov", orientation = false, t1 = "faces",    t2 = "vertices", n1 = "face",    n2 = "vertex"},
    {global = 'edgesofface',      name = "ftoe", orientation = true,  t1 = "faces",    t2 = "edges",    n1 = "face",    n2 = "edge"},
    {global = 'verticesofcell',   name = "ctov", orientation = false, t1 = "cells",    t2 = "vertices", n1 = "cell",    n2 = "vertex"},
    {global = 'edgesofcell',      name = "ctoe", orientation = false, t1 = "cells",    t2 = "edges",    n1 = "cell",    n2 = "edge"},
    {global = 'facesofcell',      name = "ctof", orientation = true,  t1 = "cells",    t2 = "faces",    n1 = "cell",    n2 = "face"},
    {global = 'cellsofcell',      name = "ctoc", orientation = false, t1 = "cells",    t2 = "cells",    n1 = "c1",      n2 = "c2"}
}
-- these relations go directly into the corresponding element table
-- etov goes into table for edges, ftoc goes into table for faces
local mesh_rels_topo = {
    {name = "etov", table = "edges", ft = "vertices", n1 = "head",    n2 = "tail"},
    {name = "ftoc", table = "faces", ft = "cells",    n1 = "outside", n2 ="inside"}
}

local function initMeshRelations(mesh)
    -- initialize list of relations
    local auto = {}
    -- basic element relations

    for _i, topo_elem in ipairs(topo_elems) do
        local tsize = tonumber(mesh["n"..topo_elem])
        local t     = L.NewRelation(tsize, topo_elem)
        auto[topo_elem] = t
        t:NewField('value', RTYPE)
        local terra init (mem: &RTYPE:terraType(), i : int)
            mem[0] = i
        end
        t.value:LoadFromCallback(init)
    end

    -- other mesh relations
    for k, rel_tuple in pairs(mesh_rels_new) do
        local globalname = rel_tuple.global
        local rel_name   = rel_tuple.name
        local tsize      = mesh[rel_name].row_idx[mesh["n"..rel_tuple.t1]]
        local rel_table  = L.NewRelation(tsize, globalname)

        -- store table with name intended for global scope
        auto[globalname] = rel_table

        rel_table:NewField(rel_tuple.n1, auto[rel_tuple.t1])
        rel_table:NewField(rel_tuple.n2, auto[rel_tuple.t2])

        -- if our lmesh field has orientation encoded into the relation, extract the
        -- orientation and load it as a separate field
        if rel_tuple.orientation then
            local rtype = RTYPE:terraType()
            local otype = OTYPE:terraType()

            local ordata   = terralib.cast(&otype, C.malloc(rel_table._size * terralib.sizeof(otype)))
            local vdata    = terralib.cast(&rtype, C.malloc(rel_table._size * terralib.sizeof(rtype)))
            local srcdata  = terralib.cast(&rtype, mesh[rel_name].values)

            local terra extract_orientation()
                var exp  : rtype = 8 * [terralib.sizeof(rtype)] - 1
                var mask : rtype = C.pow(2,exp) - 1

                for i = 0, rel_table._size do
                    ordata[i] = srcdata[i] >> exp
                    vdata[i]  = srcdata[i] and mask
                end
            end
            extract_orientation()

            rel_table[rel_tuple.n2]:LoadFromMemory(vdata)
            rel_table:NewField('orientation', L.bool)
            rel_table.orientation:LoadFromMemory(terralib.cast(&bool, ordata))
            C.free(ordata)
            C.free(vdata)
        else
            rel_table[rel_tuple.n2]:LoadFromMemory(mesh[rel_name].values)
        end

        rel_table:LoadIndexFromMemory(rel_tuple.n1, mesh[rel_name].row_idx)
    end

    for k, rel_tuple in pairs(mesh_rels_topo) do
        local rel_name = rel_tuple.name
        local rel_table = auto[rel_tuple.table]

        local f1 = rel_table:NewField(rel_tuple.n1, auto[rel_tuple.ft])
        local f2 = rel_table:NewField(rel_tuple.n2, auto[rel_tuple.ft])

        local tsize = terralib.sizeof(rel_table[rel_tuple.n1].type:terraType())

        f1:LoadFromMemory(mesh[rel_name].values,2*tsize)
        f2:LoadFromMemory(mesh[rel_name].values,2*tsize, tsize)
    end
    return auto
end

-- returns all relations from the given file
function L.initMeshRelationsFromFile(filename)
    local ctx, mesh = R.loadMesh(filename)
    local M         = initMeshRelations(mesh)

    M.__mesh = mesh
    M.__ctx  = ctx

    -- load position data
    local S = terralib.includec('runtime/single/liszt_runtime.h')
    local pos_data = terralib.cast(&float[3], S.lLoadPosition(ctx))
    M.vertices:NewField("position", L.vector(L.float,3))
    M.vertices.position:LoadFromMemory(pos_data)
    C.free(pos_data)
    return M
end

local el_types = {
    vertices = O.L_VERTEX,
    edges    = O.L_EDGE,
    faces    = O.L_FACE,
    cells    = O.L_CELL
}

function L.loadSetFromMesh(M, relation, name)
    -- create a new relation singleton representing the boundary set
    local data, size = O.loadBoundarySet(M.__ctx, el_types[relation._debugname], name)

    local s = L.NewRelation(tonumber(size), name)
    s:NewField('value', relation)
    s.value:LoadFromMemory(data)
    return s    
end

return L
