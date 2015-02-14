local LMesh = {}
package.loaded["domains.lmesh"] = LMesh
local L = require "compiler.lisztlib"
local PN = require "lib.pathname"
local lisztlibrary = tostring(PN.liszt_root()..'runtime/libsingle_runtime.so')
terralib.linklibrary(lisztlibrary)

local C = require "compiler.c"
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

    field:LoadFunction(function(i) return src[i] end)
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

    field:LoadFunction(function(i)
        return srcmem[stride*i + offset]
    end)

    --field:LoadFromCallback(row32_copy_callback(srcmem, stride, offset))
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
        assert(start >= 0 and start < fsize)
        assert(finish >= start and finish <= fsize)
        for j = start, finish - 1 do
            scratch[j] = i
        end
    end
    f:LoadFromMemory(scratch)
    C.free(scratch)
    rel:GroupBy(name)
end

local function initMeshRelations(mesh)
    import "compiler.liszt"
    -- initialize list of relations
    local relations = {}
    -- basic element relations

    for _i, name in ipairs(topo_elems) do
        local n_rows       = tonumber(mesh["n"..name])
        relations[name]    = L.NewRelation { size = n_rows, name = name }
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
        local rel        = L.NewRelation { size = n_rows, name = name }

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

                for i = 0, [rel:Size()] do
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
-- returns all relations from the given file
function LMesh.Load(filename)
    if PN.is_pathname(filename) then filename = tostring(filename) end
    local meshdata = terralib.new(C.LMeshData)
    C.LMeshLoadFromFile(filename,meshdata)
    
    local M = initMeshRelations(meshdata.mesh)
    
    for i = 0,meshdata.nBoundaries-1 do
        local b = meshdata.boundaries[i]
        local name = ffi.string(b.name)
        local relationname = ffi.string(b.type)
        assert(M[relationname])
        name = sanitizeName(name)
        local s = L.NewRelation { size = tonumber(b.size), name = name }
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
