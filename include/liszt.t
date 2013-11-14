local R = terralib.require('runtime/mesh')
local O = terralib.require('runtime/liszt')
local T = terralib.require('compiler/types')
local t, Type = T.t, T.Type

local L = {}
local LDB = terralib.require('include/ldb')
L.LDB = LDB

local DECL = terralib.require('include/decl')


local C = DECL.C --[[terralib.includecstring [[
    #include <stdlib.h>
    #include <string.h>
    #include <stdio.h>
    #include <math.h>
]]

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


-------------------------------------------------------------------------------
--[[ Code that loads all relations from mesh file formats:                 ]]--
-------------------------------------------------------------------------------
local topo_elems    = { 'vertices', 'edges', 'faces', 'cells' }

-- other mesh relations - we have a separate table for each of these relations
local mesh_rels_new = {
    {name = 'verticesofvertex', old_name = "vtov", orientation = false, t1 = "vertices", t2 = "vertices", n1 = "v1",     n2 = "v2"},
    {name = 'edgesofvertex',    old_name = "vtoe", orientation = true,  t1 = "vertices", t2 = "edges",    n1 = "vertex", n2 = "edge"},
    {name = 'facesofvertex',    old_name = "vtof", orientation = false, t1 = "vertices", t2 = "faces",    n1 = "vertex", n2 = "face"},
    {name = 'cellsofvertex',    old_name = "vtoc", orientation = false, t1 = "vertices", t2 = "cells",    n1 = "vertex", n2 = "cell"},
    {name = 'facesofedge',      old_name = "etof", orientation = true,  t1 = "edges",    t2 = "faces",    n1 = "edge",   n2 = "face"},
    {name = 'cellsofedge',      old_name = "etoc", orientation = false, t1 = "edges",    t2 = "cells",    n1 = "edge",   n2 = "cell"},
    {name = 'verticesofface',   old_name = "ftov", orientation = false, t1 = "faces",    t2 = "vertices", n1 = "face",   n2 = "vertex"},
    {name = 'edgesofface',      old_name = "ftoe", orientation = true,  t1 = "faces",    t2 = "edges",    n1 = "face",   n2 = "edge"},
    {name = 'verticesofcell',   old_name = "ctov", orientation = false, t1 = "cells",    t2 = "vertices", n1 = "cell",   n2 = "vertex"},
    {name = 'edgesofcell',      old_name = "ctoe", orientation = false, t1 = "cells",    t2 = "edges",    n1 = "cell",   n2 = "edge"},
    {name = 'facesofcell',      old_name = "ctof", orientation = true,  t1 = "cells",    t2 = "faces",    n1 = "cell",   n2 = "face"},
    {name = 'cellsofcell',      old_name = "ctoc", orientation = false, t1 = "cells",    t2 = "cells",    n1 = "c1",     n2 = "c2"}
}
-- these relations go directly into the corresponding element table
-- etov goes into table for edges, ftoc goes into table for faces
local mesh_rels_topo = {
    {old_name = "etov", dest = "edges", ft = "vertices", n1 = "head",    n2 = "tail"},
    {old_name = "ftoc", dest = "faces", ft = "cells",    n1 = "outside", n2 = "inside"}
}

-- TEMP NOTE:
-- This code was removed from the fields/relations themselves
-- in order to keep their interface cleaner.
-- Most of this grunginess makes various assumptions about
-- how the LMesh is laid out.
--  I assume that this code will be factored out into a
-- separate LMesh -> Relational Mesh converter at some point

-- generic initialization from memory
-- will do Terra type coercions
local function initFieldViaCopy(field, src)
    assert(field.data == nil)

    local ftype = field.type:terraType()

    field:LoadFromCallback(terra( dst : &ftype, i : int )
        @dst = src[i]
    end)
end

-- we assume srcmem has terra type &uint32
-- TODO: it would be good to assert this; don't know how
-- have a type error on offset function now at least...
local function row32_copy_callback(srcmem, stride, offset)
    stride = stride or 1
    offset = offset or 0
    srcmem = (terra() : &uint32 return srcmem + offset end)()

    return terra(dst : &uint64, i : int)
        @dst = srcmem[stride * i]
    end
end
local function initRowFromMemory32(field, srcmem, stride, offset)
    if not field.type:isRow() then
        error('can only call initRowFromMemory32 on row fields', 2)
    end

    stride      = stride or 1
    offset      = offset or 0
    assert(field.data == nil)

    field:LoadFromCallback(row32_copy_callback(srcmem, stride, offset))
end

-- used to load the more irregular first index from a CRS array
local function initRowFromCRSindex32(field, row_idx)
    if not field.type:isRow() then
        error('can only call initRowFromCRSindex32 on row fields', 2)
    end

    assert(field.data == nil)

    -- we create a temporary array with exactly the data we'd like to copy in
    local N         = field.owner._size
    local tmp_mem   =
        terralib.cast(&uint64,C.malloc(N * terralib.sizeof(uint64)))
    local idxN      = field.type.relation._size

    -- note: rows_idx is size idxN + 1 to hold the final stop
    for i = 0, idxN-1 do
        local start = row_idx[i]
        local stop  = row_idx[i+1]
        for j = start, stop-1 do
            tmp_mem[j] = i
        end
    end

    -- having constructed the temp, we copy it over and delete the temp
    initFieldViaCopy(field, tmp_mem)
    C.free(tmp_mem)
end

local function initMeshRelations(mesh)
    -- initialize list of relations
    local relations = {}
    -- basic element relations

    for _i, name in ipairs(topo_elems) do
        local n_rows       = tonumber(mesh["n"..name])
        relations[name]    = LDB.NewRelation(n_rows, name)
        --relations[name]:NewField('value', REF_TYPE)
        --local terra init (mem: &REF_TYPE:terraType(), i : int)
        --    mem[0] = i
        --end
        --t.value:LoadFromCallback(init)
    end

    -- other mesh relations
    for k, xtoy in pairs(mesh_rels_new) do
        local name       = xtoy.name
        local old_name   = xtoy.old_name
        local n_t1       = relations[xtoy.t1]:size()
        local n_rows     = mesh[old_name].row_idx[n_t1]
        local rel        = LDB.NewRelation(n_rows, name)

        -- store table with name intended for global scope
        relations[name] = rel

        rel:NewField(xtoy.n1, relations[xtoy.t1])
        rel:NewField(xtoy.n2, relations[xtoy.t2])

        initRowFromCRSindex32(rel[xtoy.n1], mesh[old_name].row_idx)

        -- if our lmesh field has orientation encoded into the relation,
        -- extract the orientation and load it as a separate field
        if xtoy.orientation then
            local function alloc(n, typ)
                return terralib.cast(&typ, C.malloc(n * terralib.sizeof(typ)))
            end
            local ordata   = alloc(n_rows, bool)
            local vdata    = alloc(n_rows, uint32)
            local srcdata  = terralib.cast(&uint32, mesh[old_name].values)

            (terra ()
                var exp  : uint32 = 8 * [terralib.sizeof(uint32)] - 1
                var mask : uint32 = C.pow(2,exp) - 1

                for i = 0, rel._size do
                    ordata[i] = bool(srcdata[i] >> exp)
                    vdata[i]  = srcdata[i] and mask
                end
            end)()

            rel:NewField('orientation', L.bool)
            initFieldViaCopy(rel.orientation, ordata)
            initRowFromMemory32(rel[xtoy.n2], vdata)
            C.free(ordata)
            C.free(vdata)
        else
            initRowFromMemory32(rel[xtoy.n2], mesh[old_name].values)
        end
    end

    for k, xtoy in pairs(mesh_rels_topo) do
        local old_name  = xtoy.old_name
        local rel       = relations[xtoy.dest]

        local f1 = rel:NewField(xtoy.n1, relations[xtoy.ft])
        local f2 = rel:NewField(xtoy.n2, relations[xtoy.ft])

        local values = terralib.cast(&uint32, mesh[old_name].values)
        initRowFromMemory32(f1, values, 2)
        initRowFromMemory32(f2, values, 2, 1)
    end
    return relations
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
    M.vertices.position:LoadFromCallback(
        terra( dst : &vector(float,3), i : int)
            @dst = vector(pos_data[i][0], pos_data[i][1], pos_data[i][2])
        end)
    C.free(pos_data)
    return M
end


-- TODO: can we get rid of the following stuff somehow or design for it?
-- The code below is almost certainly broken
local el_types = {
    vertices = O.L_VERTEX,
    edges    = O.L_EDGE,
    faces    = O.L_FACE,
    cells    = O.L_CELL
}

function L.loadSetFromMesh(M, relation, name, fieldname)
    if not fieldname then fieldname = 'value' end
    -- create a new relation singleton representing the boundary set
    local data, size =
        O.loadBoundarySet(M.__ctx, el_types[relation._name], name)
    data = terralib.cast(&uint32, data)

    local s = LDB.NewRelation(tonumber(size), name)
    s:NewField(fieldname, relation)
    initRowFromMemory32(s[fieldname], data)
    return s
end

return L
