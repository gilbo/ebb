local L = {}

local utils = terralib.require("relations/relations_util")
local mesh_h = terralib.includec("runtime/common/mesh_crs.h")
local c = terralib.includec("stdio.h")

--[[
- For mesh relations corresponding to CRS relations in original mesh, first
  field represents index field and second field represents corresponding
  values.
- Oriented relations have a field orientation.
- For relation tables for vertices, edges, faces and cells, elements are
  initialized to length-1, with corresponding fields if any.
--]]

-- basic relations
local topo_elems = {
    "vertices",
    "edges",
    "faces",
    "cells"
    }

-- other mesh relations
local mesh_rels_new = {
    {name = "vtov", orientation = 0, t1 = "vertices", t2 = "vertices", n1 = "v1", n2 = "v2"},
    {name = "vtoe", orientation = 1, t1 = "vertices", t2 = "edges", n1 = "v", n2 = "e"},
    {name = "vtof", orientation = 0, t1 = "vertices", t2 = "faces", n1 = "v", n2 = "f"},
    {name = "vtoc", orientation = 0, t1 = "vertices", t2 = "cells", n1 = "v", n2 = "c"},
    {name = "etof", orientation = 1, t1 = "edges", t2 = "faces", n1 = "e", n2 = "f"},
    {name = "etoc", orientation = 0, t1 = "edges", t2 = "cells", n1 = "e", n2 = "c"},
    {name = "ftov", orientation = 0, t1 = "faces", t2 = "vertices", n1 = "f", n2 = "v"},
    {name = "ftoe", orientation = 1, t1 = "faces", t2 = "edges", n1 = "f", n2 = "e"},
    {name = "ctov", orientation = 0, t1 = "cells", t2 = "vertices", n1 = "c", n2 = "v"},
    {name = "ctoe", orientation = 0, t1 = "cells", t2 = "edges", n1 = "c", n2 = "e"},
    {name = "ctof", orientation = 1, t1 = "cells", t2 = "faces", n1 = "c", n2 = "f"},
    {name = "ctoc", orientation = 0, t1 = "cells", t2 = "cells", n1 = "c1", n2 = "c2"}
    }
local mesh_rels_topo = {
    {name = "etov", table = "edges", ft = "vertices", n1 = "head", n2 = "tail"},
    {name = "ftoc", table = "faces", ft = "cells", n1 = "outside", n2 ="inside"}
    }

local function link_runtime ()
	local osf = assert(io.popen('uname', 'r'))
	local osname = assert(osf:read('*l'))
	osf:close()

	if osname == 'Linux' then
		terralib.linklibrary("runtime/single/libruntime_single.so")
	elseif osname == 'Darwin' then
		terralib.linklibrary("runtime/single/runtime_single.dylib")
	else
		error("Unknown Operating System")
	end
end
link_runtime()

terra L.readMesh(filename : rawstring) : &mesh_h.Mesh
    c.printf("Loading mesh file ...\n")
    var mesh : &mesh_h.Mesh = mesh_h.lMeshInitFromFile(filename)
    c.printf("Loaded mesh file\n")
    return mesh
end

struct FieldParams {
    nvertices : int;
    nedges : int;
    nfaces : int;
    ncells : int;
}

terra L.getMeshParams(mesh : &mesh_h.Mesh) : FieldParams
    var params : FieldParams
    params.nvertices = mesh.nvertices
    params.nedges = mesh.nedges
    params.nfaces = mesh.nfaces
    params.ncells = mesh.ncells
    return params
end

function L.initMeshRelations(mesh, params)
    -- initialize list of relations
    local elems = {}
    local rels = {}
    -- basic element relations
    for k, topo_elem in pairs(topo_elems) do
        local tsize = params["n"..topo_elem]
        elems[topo_elem] = utils.newtable(tsize, topo_elem)
        elems[topo_elem]:initializenumfield("values", false)
    end
    -- other mesh relations
    for k, rel_tuple in pairs(mesh_rels_new) do
        local rel_name = rel_tuple.name
        local tsize = mesh[rel_name].row_idx[params["n"..rel_tuple.t1]]
        local rel_table = utils.newtable(tsize, rel_name)
        rels[rel_name] = rel_table
        rel_table[rel_tuple.n1] = utils.newfield(elems[rel_tuple.t1])
        rel_table[rel_tuple.n2] = utils.newfield(elems[rel_tuple.t2])
        if rel_tuple.orientation then
            local datasize = rel_table[rel_tuple.n2]:loadfrommemoryskipmsb(mesh[rel_name].values)
            rel_table.orientation = utils.newfield(bool)
            rel_table.orientation:loadfrommemoryorientation(mesh[rel_name].values, datasize)
        else
            rel_table[rel_tuple.n2]:loadfrommemory(mesh[rel_name].values)
        end
        rel_table:loadindexfrommemory(rel_tuple.n1, mesh[rel_name].row_idx)
        elems[rel_tuple.t1]:addrelation(rel_tuple.t2, rel_table)
    end
    for k, rel_tuple in pairs(mesh_rels_topo) do
        local rel_name = rel_tuple.name
        local rel_table = elems[rel_tuple.table]
        rel_table[rel_tuple.n1] = utils.newfield(elems[rel_tuple.ft])
        rel_table[rel_tuple.n2] = utils.newfield(elems[rel_tuple.ft])
        rel_table[rel_tuple.n1]:loadalternatefrommemory(mesh[rel_name].values[0])
        rel_table[rel_tuple.n2]:loadalternatefrommemory(mesh[rel_name].values[1])
    end
    return elems, rels
end

-- Test code

print("Testing code ...")

local mesh = L.readMesh("relations/mesh.lmesh")

-- TODO: Speak to Zach and remove this - problem accessing mesh elements
-- Remove this eventually, this should not be required.
local params = L.getMeshParams(mesh)

local elems, rels = L.initMeshRelations(mesh, params)

for i,t in pairs(elems) do
    print("** Elem table **")
    t:dump()
end

for i,t in pairs(rels) do
    print("## Other rels table ##")
    t:dump()
end

return L
