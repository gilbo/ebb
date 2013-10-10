local L = {}

local utils = terralib.require("relations/relations_util")
local mesh_h = terralib.includec("runtime/common/mesh_crs.h")
local c = terralib.includec("stdio.h")

--[[
For mesh relations corresponding to CRS relations in original mesh, x (first
field) represents index field and y (second field) the corresponding values.
--]]

-- basic relations
local topo_elems = {
    "vertices",
    "edges",
    "faces",
    "cells"
    }

-- other mesh relations
local mesh_rels = {
    {"vtov", 1, "vertices", "vertices"},
    {"vtoe", 1, "vertices", "edges"},
    {"vtof", 1, "vertices", "faces"},
    {"vtoc", 1, "vertices", "cells"},
    {"etof", 1, "edges", "faces"},
    {"etoc", 1, "edges", "cells"},
    {"ftov", 1, "faces", "vertices"},
    {"ftoe", 1, "faces", "edges"},
    {"ctov", 1, "cells", "vertices"},
    {"ctoe", 1, "cells", "edges"},
    {"ctof", 1, "cells", "faces"},
    {"ctoc", 1, "cells", "cells"},
    {"etov", 0, "head", "tail", "edges", "vertices"},
    {"ftoc", 0, "outside", "inside", "faces", "cells"}
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
    c.printf("Number of vertices = %i\n", mesh.nvertices)
    c.printf("Number of edges = %i\n", mesh.nedges)
    c.printf("Number of faces = %i\n", mesh.nfaces)
    c.printf("Number of cells = %i\n", mesh.ncells)
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
    end
    -- other mesh relations
    for k, rel_tuple in pairs(mesh_rels) do
        local rel_name = rel_tuple[1]
        if rel_tuple[2] == 1 then
            local x = rel_tuple[3]
            local y = rel_tuple[4]
            local tsize = mesh[rel_name].row_idx[params["n"..x]]
            rels[rel_name] = utils.newtable(tsize, rel_name)
            local rel_table = rels[rel_name]
            rel_table.x = utils.newfield(elems[x])
            rel_table.y = utils.newfield(elems[y])
            rel_table.y:loadfrommemory(mesh[rel_name].values)
            rel_table:loadindexfrommemory("x", mesh[rel_name].row_idx)
            elems[x]:addrelation(y, rel_table)
        else
            local first = rel_tuple[3]
            local second = rel_tuple[4]
            local rel_table = elems[rel_tuple[5]]
            local rel_field = elems[rel_tuple[6]]
            rel_table[first] = utils.newfield(rel_field)
            rel_table[second] = utils.newfield(rel_field)
            rel_table[first]:loadalternatefrommemory(mesh[rel_name].values[0])
            rel_table[second]:loadalternatefrommemory(mesh[rel_name].values[1])
        end
    end
    return elems, rels
end

-- Test code

print("Testing code ...")

local mesh = L.readMesh("relations/mesh.lmesh")

-- TODO: Speak to Zach and remove this - problem accessing mesh elements
local params = L.getMeshParams(mesh)

local elems, rels = L.initMeshRelations(mesh, params)

for i,t in pairs(elems) do
    print("** Elem table **")
    t:dump()
end
--
--for i,t in pairs(rels) do
--    print("## Other rels table ##")
--    t:dump()
--end
