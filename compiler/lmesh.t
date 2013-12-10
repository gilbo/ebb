local LMesh = {}
package.loaded["compiler.lmesh"] = LMesh
local L = terralib.require "compiler.lisztlib"
local PN = terralib.require "compiler.pathname"
local LDB = terralib.require "compiler.ldb"
local Particle = terralib.require "compiler.particle"
local lisztlibrary = os.getenv("LISZT_RUNTIME")
terralib.linklibrary(lisztlibrary)

local C = terralib.require "compiler.c"
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

local function alloc(n, typ)
    return terralib.cast(&typ, C.malloc(n * terralib.sizeof(typ)))
end
            
function initFieldFromIndex(rel,name, key, row_idx)
    rel:NewField(name, key)
    local f = rel[name]
    local numindices = key:Size()
    local fsize = f:Size()
    local scratch = alloc(fsize,uint64)
    for i = 0, numindices-1 do
        local start = row_idx[i]
        local finish = row_idx[i+1]
        assert(start >= 0 and start <= fsize)
        assert(finish >= start and finish <= fsize)
        for j = start, finish - 1 do
            scratch[j] = i
        end
    end
    f:LoadFromMemory(scratch)
    C.free(scratch)
    rel:CreateIndex(name)
end

local function initMeshRelations(mesh)
    import "compiler.liszt"
    -- initialize list of relations
    local relations = {}
    -- basic element relations

    for _i, name in ipairs(topo_elems) do
        local n_rows       = tonumber(mesh["n"..name])
        relations[name]    = L.NewRelation(n_rows, name)
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
        local n_t1       = relations[xtoy.t1]:Size()
        local n_rows     = mesh[old_name].row_idx[n_t1]
        local rel        = L.NewRelation(n_rows, name)

        -- store table with name intended for global scope
        relations[name] = rel

        initFieldFromIndex(rel,xtoy.n1, relations[xtoy.t1], mesh[old_name].row_idx)
        rel:NewField(xtoy.n2, relations[xtoy.t2])
        -- if our lmesh field has orientation encoded into the relation,
        -- extract the orientation and load it as a separate field
        if xtoy.orientation then
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
        --setup the correct macros
        relations[xtoy.t1]:NewFieldMacro(xtoy.t2,L.NewMacro(function(f)
            return liszt `L.Where(rel.[xtoy.n1],f).[xtoy.n2]
        end))
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

local ffi = require "ffi"
local function sanitizeName(name)
    return name:gsub("[^%w]","_"):gsub("^[^%a]","_")
end
-- code in common between file loading and procedural loading
local function LoadCallback(callback)
    local meshdata = callback()
    
    local M = initMeshRelations(meshdata.mesh)
    
    for i = 0,meshdata.nBoundaries-1 do
        local b = meshdata.boundaries[i]
        local name = ffi.string(b.name)
        local relationname = ffi.string(b.type)
        assert(M[relationname])
        name = sanitizeName(name)
        local s = L.NewRelation(tonumber(b.size), name)
        s:NewField("value",M[relationname])
        s.value:LoadFromMemory(b.data)
        M[name] = s
    end
    for i = 0,meshdata.nFields-1 do
        local f = meshdata.fields[i]
        local name = ffi.string(f.name)
        local rname = ffi.string(f.elemtype)
        local typestring = ffi.string(f.datatype)
        local relation = M[rname]
        
        local elemtyp = L[typestring]
        
        local typ = elemtyp
        local nelems = tonumber(f.nelems)
        if nelems > 1 then
            typ = L.vector(typ,nelems)
        end
        local elemtypt = elemtyp:terraType()
        local typt = typ:terraType()
        local tdata = terralib.cast(&elemtypt,f.data)
        name = sanitizeName(name)
        relation:NewField(name,typ)
        local field = relation[name]
        local templatesize = terralib.sizeof(typt)
        local nelemstemplate = templatesize / terralib.sizeof(elemtypt)
        local template = terralib.cast(&elemtypt,C.malloc(templatesize*field:Size()))
        for i = 0, field:Size()-1 do
            for j = 0,nelems-1 do
                template[nelemstemplate*i + j] = tdata[nelems*i + j]
            end 
        end
        relation[name]:LoadFromMemory(template)
        C.free(template)
    end
    return M
end

-- returns all relations from the given file
function LMesh.Load(filename)
    if PN.is_pathname(filename) then filename = tostring(filename) end
    return LoadCallback(function()
        local meshdata = terralib.new(C.LMeshData)
        C.LMeshLoadFromFile(filename,meshdata)
        return meshdata
    end)
end


-------------------------------------------------------------------------------
--[[ Generic grid relations                                                ]]--
-------------------------------------------------------------------------------
local function VertexId(coord, dimensions)
    local stridey = (dimensions[3] + 1)
    local stridex = (dimensions[2] + 1) * stridey
    return stridex * coord[1] + stridey * coord[2] + coord[3]
end

local function EdgeId(dir, start, dimensions)
    if dir < 0 or dir > 2 then error("dir should be a number between 0 and 2, inclusive", 2) end

    local dirstart = 0
    local dx, dy, dz = unpack(dimensions)
    if dir >= 1 then
        dirstart = dx * (dy + 1) * (dz + 1)
    end
    if dir == 2 then
        dirstart = dirstart + (dx + 1) * dy * (dz + 1)
    end

    local stridey = dz + (dir == 2 and 0 or 1)
    local stridex = (dy + (dir == 1 and 0 or 1)) * stridey
    return dirstart + stridex * start[1] + stridey * start[2] + start[3]
end

local function Grid_ctoe(dimensions)
    local num_cells = 1 + dimensions[1] * dimensions[2] * dimensions[3]
    local row_idx = {}
    row_idx[0] = 0
    for i = 1, num_cells do
        row_idx[i] = (i - 1) * 12
    end
    local values = alloc(row_idx[num_cells], uint32)

    local i = 0
    for x = 0,dimensions[1] - 1 do
    for y = 0,dimensions[2] - 1 do
    for z = 0,dimensions[3] - 1 do
        for dir = 0,2 do
            for ed1 = 0,1 do
            for ed2 = 0,1 do
                local start = {x, y, z}
                local idx1 = (dir + 1) % 3 + 1
                start[idx1] = start[idx1] + ed1
                local idx2 = (dir + 2) % 3 + 1
                start[idx2] = start[idx2] + ed2
                values[i] = EdgeId(dir, start, dimensions)
                i = i + 1
            end end
        end
    end end end

    return {row_idx = row_idx, values = values}
end

local function Grid_ctov(dimensions)
    local num_cells = 1 + dimensions[1] * dimensions[2] * dimensions[3]
    local row_idx = {}
    row_idx[0] = 0
    for i = 1, num_cells do
        row_idx[i] = (i - 1) * 8
    end
    local values = alloc(row_idx[num_cells], uint32)

    local i = 0
    for x = 0,dimensions[1] - 1 do
    for y = 0,dimensions[2] - 1 do
    for z = 0,dimensions[3] - 1 do
        for dx = 0,1 do
        for dy = 0,1 do
        for dz = 0,1 do
            local coord = {x + dx, y + dy, z + dz}
            values[i] = VertexId(coord, dimensions)
            i = i + 1
        end end end
    end end end

    return {row_idx = row_idx, values = values}
end

local function Grid_etov(dimensions, num_edges)
    local row_idx = {}
    for i = 0, num_edges do
        row_idx[i] = i * 2
    end
    local values = alloc(num_edges * 2, uint32)

    local i = 0
    for dir = 0,2 do
        for sx = 0,dimensions[1] - (dir == 0 and 1 or 0) do
        for sy = 0,dimensions[2] - (dir == 1 and 1 or 0) do
        for sz = 0,dimensions[3] - (dir == 2 and 1 or 0) do
            local tail = {sx, sy, sz}
            local head = {tail[1], tail[2], tail[3]}
            head[dir + 1] = head[dir + 1] + 1
            assert(i % 2 == 0)
            assert(EdgeId(dir, tail, dimensions) == i / 2)
            values[i] = VertexId(head, dimensions)
            i = i + 1
            values[i] = VertexId(tail, dimensions)
            i = i + 1
        end end end
    end

    return {row_idx = row_idx, values = values}
end

local function GridMesh(dimensions)
    local m = {}
    local dx, dy, dz = dimensions[1], dimensions[2], dimensions[3]
    m.nvertices = (dx + 1) * (dy + 1) * (dz + 1)
    m.nedges = dx * (dy + 1) * (dz + 1) + (dx + 1) * dy * (dz + 1) +
               (dx + 1) * (dy + 1) * dz
    m.nfaces = 0 --(dx + 1) * dy * dz + dx * (dy + 1) * dz + dx * dy * (dz + 1)
    m.ncells = 1 + dx * dy * dz

    local empty_idx_v = {}
    for i=0,m.nvertices do
        empty_idx_v[i] = 0
    end
    local empty_idx_e = {}
    for i=0,m.nedges do
        empty_idx_e[i] = 0
    end
    local empty_idx_f = {}
        empty_idx_f[0] = 0
    local empty_idx_c = {}
    for i=0,m.ncells do
        empty_idx_c[i] = 0
    end
    local empty_array = alloc(0, uint32)
    m.vtov = {row_idx = empty_idx_v, values = empty_array}
    m.vtoe = {row_idx = empty_idx_v, values = empty_array}
    m.vtof = {row_idx = empty_idx_v, values = empty_array}
    m.vtoc = {row_idx = empty_idx_v, values = empty_array}
    m.etov = Grid_etov(dimensions, m.nedges)
    m.etof = {row_idx = empty_idx_e, values = empty_array}
    m.etoc = {row_idx = empty_idx_e, values = empty_array}
    m.ftov = {row_idx = empty_idx_f, values = empty_array}
    m.ftoe = {row_idx = empty_idx_f, values = empty_array}
    m.ftoc = {row_idx = empty_idx_f, values = empty_array}
    m.ctov = Grid_ctov(dimensions)
    m.ctoe = Grid_ctoe(dimensions)
    m.ctof = {row_idx = empty_idx_c, values = empty_array}
    m.ctoc = {row_idx = empty_idx_c, values = empty_array}

    return m
end

local function GridMeshData(dimensions, position_data)
    local fields = {}
    fields[0] = {
        name = "position",
        elemtype = "vertices",
        datatype = "double",
        nelems = 3, 
        data = position_data
    }
    return {
        nFields = 1,
        fields = fields,
        nBoundaries = 0,
        mesh = GridMesh(dimensions)
    }
end


-------------------------------------------------------------------------------
--[[ Uniform grid                                                          ]]--
-------------------------------------------------------------------------------
local terra VectorSet(data : &double, i : uint,
                      x : double, y : double, z : double)
    data[i * 3] = x
    data[i * 3 + 1] = y
    data[i * 3 + 2] = z
end
    
local function UniformPositionData(dimensions, minExtent, maxExtent)
    local nvertices = (dimensions[1] + 1) * (dimensions[2] + 1) * (dimensions[3] + 1)
    local data = alloc(nvertices * 3, double)
    local spacing = {(maxExtent[1] - minExtent[1]) / dimensions[1],
                     (maxExtent[2] - minExtent[2]) / dimensions[2],
                     (maxExtent[3] - minExtent[3]) / dimensions[3]}
    local i = 0
    for x=0,dimensions[1] do
    for y=0,dimensions[2] do
    for z=0,dimensions[3] do
        VectorSet(data, i, minExtent[1] + spacing[1] * x,
                           minExtent[2] + spacing[2] * y,
                           minExtent[3] + spacing[3] * z)
        i = i + 1
    end end end
    return data
end

-- returns relations necessary for a uniform grid with some number of particles
function LMesh.LoadUniformGrid(numParticles, dimensions, minExtent, maxExtent)
    if minExtent == nil then minExtent = {0, 0, 0} end
    if maxExtent == nil then maxExtent = dimensions end

    local meshdata = GridMeshData(dimensions,
            UniformPositionData(dimensions, minExtent, maxExtent))
    local M = LoadCallback(function()
        return meshdata
    end)

    Particle.initUniformGrid(M, numParticles, dimensions, minExtent, maxExtent)
    return M
end
