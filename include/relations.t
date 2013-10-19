local R = terralib.require('runtime/mesh')

local L = {}

local C = terralib.includecstring [[
    #include <stdlib.h>
    #include <string.h>
    #include <stdio.h>
    #include <math.h>
]]

--[[
- A table contains size, fields and _indexrelations, which point to tables
  that are indexed by this table. Example, _indexrelations for vertices table
  will point to vtov, vtoe etc. Use table:getrelationtable(topo_elem) to get a
  relation table. Example, vertices:getrelationtable(edges) will give vtoe.
- A table also contains _index that has the compressed row values for index
  field, and the corresponding expanded index field and other field values.
- A field contains fieldname, type of field, pointer to its table and expanded
  data.
--]]

local table = {}
table.__index = table
function L.istable(t)
    return getmetatable(t) == table
end

local key = {}

function L.newtable(size, debugname)
    return setmetatable( {
        _size      = size,
        _fields    = terralib.newlist(),
        _debugname = debugname or "anon"
    },
    table)
end

local field = {}

field.__index = field
function L.isfield(f)
    return getmetatable(f) == field
end

function L.newfield(t)
    return { type = t } 
end

function table:__newindex(fieldname,value)
    local typ = value.type --TODO better error checking
    local f = setmetatable({},field)
    rawset(self,fieldname,f)
    f.name     = fieldname
    f.table    = self
    f.type     = typ
    f.realtype = L.istable(f.type) and uint32 or f.type
    self._fields:insert(f)
end 

-- If default value is false,    field is initialized as 0 to tablesize - 1
-- If default value is true,     field is initialized to 0
-- If default value is a number, field is initialized to the given value
function table:initializenumfield(fieldname, defaultval)
    self[fieldname] = L.newfield(uint32)
    local f = self[fieldname]
    f.data = {}
    if type(defaultval) == "boolean" then
        if defaultval == false then
            for i = 0, self._size - 1 do
                f.data[i] = i
            end
        else
            for i = 0, self._size - 1 do
                f.data[i] = 0
            end
        end
    else
        assert(type(defaultval) == "number")
        for i = 0, self._size -1 do
            f.data[i] = defaultval
        end
    end
end

terra copy_bytes (dest : &uint8, src : &uint8, length : uint, size : uint, stride : uint, offset : uint)
    src = src + offset
    for i = 0, length do
        C.memcpy(dest,src,size)
        src  = src  + stride
        dest = dest + size
    end
end

-- specify stride, offset in bytes, default stride reads memory contiguously and default offset reads from mem ptr directly

function field:loadfrommemory (mem, stride, offset)
    if not stride then stride = terralib.sizeof(self.realtype) end
    if not offset then offset = 0 end

    assert(stride >= terralib.sizeof(self.realtype))
    assert(self.data == nil)

    local nbytes = self.table._size * terralib.sizeof(self.realtype)
    local bytes  = C.malloc(nbytes)
    self.data    = terralib.cast(&self.realtype,bytes)

    -- cast mem to void* to avoid automatic conversion issues for pointers to non-primitive types
    mem = terralib.cast(&opaque, mem)

    -- If the array is laid out contiguously in memory, just do a memcpy
    -- otherwise, read with a stride
    if (stride == terralib.sizeof(self.realtype)) then
        C.memcpy(self.data,mem,nbytes)
    else
        copy_bytes(bytes,mem,self.table._size,terralib.sizeof(self.realtype), stride, offset)
    end
end

function table:loadindexfrommemory(fieldname,row_idx)
    assert(self._index == nil)
    local f = self[fieldname]
    assert (f)
    assert(f.data == nil)

    assert(L.istable(f.type))
    local realtypesize = terralib.sizeof(f.realtype)
    local nbytes = (f.type._size + 1)*realtypesize
    rawset(self, "_index", terralib.cast(&f.realtype,C.malloc(nbytes)))
    local memT = terralib.typeof(row_idx)
    assert(memT == &f.realtype)
    C.memcpy(self._index,row_idx,nbytes)
    f.data = terralib.cast(&f.realtype,C.malloc(self._size*realtypesize))
    for i = 0, f.type._size - 1 do
        local b = self._index[i]
        local e = self._index[i+1]
        for j = b, e - 1 do
            f.data[j] = i
        end
    end
end

function field:dump()
    print(self.name..":")
    if not self.data then
        print("...not initialized")
        return
    end

    local N = self.table._size
    for i = 0,N - 1 do
        print("",i, self.data[i])
    end
end

function table:dump()
    print(self._debugname, "size: "..self._size)
    for i,f in ipairs(self._fields) do
        f:dump()
    end
end


-------------------------------------------------------------------------------
--[[ Temporary code that loads all relations from mesh file formats:       ]]--
-------------------------------------------------------------------------------
local topo_elems = {
    vertices = 'vertex',
    edges    = 'edge',
    faces    = 'face',
    cells    = 'cell'
}

-- other mesh relations - we have a separate table for each of these relations
local mesh_rels_new = {
    {global = 'verticesofvertex', name = "vtov", orientation = false, t1 = "vertices", t2 = "vertices", n1 = "v1", n2 = "v2"},
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

    for topo_elem, topo_str  in pairs(topo_elems) do
        local tsize = tonumber(mesh["n"..topo_elem])
        auto[topo_elem] = L.newtable(tsize, topo_elem)
        auto[topo_elem]:initializenumfield(topo_str, false)
    end

    -- other mesh relations
    for k, rel_tuple in pairs(mesh_rels_new) do
        local globalname = rel_tuple.global
        local rel_name   = rel_tuple.name
        local tsize      = mesh[rel_name].row_idx[mesh["n"..rel_tuple.t1]]
        local rel_table  = L.newtable(tsize, globalname)

        -- store table with name intended for global scope
        auto[globalname] = rel_table

        local ftype = uint32
        local otype = uint8

        rel_table[rel_tuple.n1] = L.newfield(auto[rel_tuple.t1])
        rel_table[rel_tuple.n2] = L.newfield(auto[rel_tuple.t2])

        -- if our lmesh field has orientation encoded into the relation, extract the
        -- orientation and load it as a separate field
        if rel_tuple.orientation then
            local ordata   = C.malloc(rel_table._size * terralib.sizeof(otype))
            local srcdata  = terralib.cast(&ftype, mesh[rel_name].values)
            local destdata = terralib.cast(&otype, ordata)

            local terra extract_orientation()
                var exp  : uint32 = 8 * [terralib.sizeof(ftype)] - 1
                var mask : uint32 = C.pow(2,exp) - 1

                var src  : &ftype = srcdata
                var dest : &otype = destdata

                for i = 0, rel_table._size - 1 do
                    dest[i] = src[i] >> exp
                    src[i]  = src[i] and mask
                end
            end
            extract_orientation()

            rel_table[rel_tuple.n2]:loadfrommemory(mesh[rel_name].values)
            rel_table.orientation = L.newfield(bool)
            rel_table.orientation:loadfrommemory(terralib.cast(&bool, destdata))
            C.free(ordata)
        else
            rel_table[rel_tuple.n2]:loadfrommemory(mesh[rel_name].values)
        end

        rel_table:loadindexfrommemory(rel_tuple.n1, mesh[rel_name].row_idx)
    end

    for k, rel_tuple in pairs(mesh_rels_topo) do
        local rel_name = rel_tuple.name
        local rel_table = auto[rel_tuple.table]
        rel_table[rel_tuple.n1] = L.newfield(auto[rel_tuple.ft])
        rel_table[rel_tuple.n2] = L.newfield(auto[rel_tuple.ft])
        local bytesize = terralib.sizeof(rel_table[rel_tuple.n1].realtype)
        rel_table[rel_tuple.n1]:loadfrommemory(mesh[rel_name].values,2*bytesize)
        rel_table[rel_tuple.n2]:loadfrommemory(mesh[rel_name].values,2*bytesize, bytesize)
    end
    return auto
end

-- returns all relations from the given file
function L.initMeshRelationsFromFile(filename)
    local mesh = R.readMesh(filename)
    local rels = initMeshRelations(mesh)
    return rels
end

return L
