local R = terralib.require('runtime/mesh')
local T = terralib.require('compiler/types')

local L = {}

local C = terralib.includecstring [[
    #include <stdlib.h>
    #include <string.h>
    #include <stdio.h>
    #include <math.h>
]]

local RTYPE = T.t.uint  -- terra type of a field that refers to another relation
local OTYPE = T.t.uint8 -- terra type of an orientation field


--------------------------------------------------------------------------------
--[[ Export Liszt types:                                                    ]]--
--------------------------------------------------------------------------------
L.int    = T.t.int
L.uint   = T.t.uint
L.float  = T.t.float
L.bool   = T.t.bool
L.vector = T.t.vector


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

- An LField stores it's fieldname, type, an array of data, and a pointer
- to another LRelation if the field itself represents relational data.
--]]

--------------------------------------------------------------------------------
--[[ Field prototype:                                                       ]]--
--------------------------------------------------------------------------------
local LRelation = make_prototype {kind=T.Type.kinds.relation}
local LField    = make_prototype {kind=T.Type.kinds.field}
local LScalar   = make_prototype {kind=T.Type.kinds.scalar}
local LVector   = make_prototype {kind=T.Type.kinds.vector}


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

local function isValidFieldType (typ)
    return is_relation(typ) or T.Type.isLisztType(typ) and typ:isExpressionType()
end

function LRelation:NewField (name, typ)    
    assert(isValidFieldType(typ))

    local f    = setmetatable({}, LField)
    f.relation = is_relation(typ) and typ   or nil
    f.type     = is_relation(typ) and RTYPE or typ

    f.name  = name
    f.table = self

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

    rawset(self, "_index", terralib.cast(&ttype,C.malloc(nbytes)))
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
    print(self._debugname, "size: "..self._size)
    for i,f in ipairs(self._fields) do
        f:dump()
    end
end


--------------------------------------------------------------------------------
--[[ Field methods:                                                         ]]--
--------------------------------------------------------------------------------
local terra copy_bytes (dest : &uint8, src : &uint8, length : uint, size : uint, stride : uint, offset : uint)
    src = src + offset

    -- don't potentially copy past the length of the source array:
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
        for i = 0, N - 1 do
            local s = ''
            for j = 0, self.type.N - 1 do
                s = s .. tostring(self.data[i][j]) .. ' '
            end
            print("", i, s)
        end
    else
        for i = 0,N - 1 do
            print("",i, self.data[i])
        end
    end
end


--------------------------------------------------------------------------------
--[[ LScalars:                                                              ]]--
--------------------------------------------------------------------------------
function L.NewScalar ()
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
    local rels      = initMeshRelations(mesh)

    -- load position data
    local S = terralib.includec('runtime/single/liszt_runtime.h')
    local pos_data = terralib.cast(&float[3], S.lLoadPosition(ctx))
    rels.vertices:NewField("position", L.vector(L.float,3))
    rels.vertices.position:LoadFromMemory(pos_data)
    C.free(pos_data)
    return rels
end

return L
