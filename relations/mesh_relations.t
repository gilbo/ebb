local L = {}

local utils = terralib.require("relations/relations_util")
local mesh_h = terralib.includec("runtime/common/mesh_crs.h")
local c = terralib.includec("stdio.h")

local vertices = "vertices"
local edges = "edges"
local faces = "faces"
local cells = "cells"

local topo_elems = {
    vertices,
    edges,
    faces,
    cells
    }

local mesh_rels = {
    {"vtov", 1, vertices, vertices},
    {"vtoe", 1, vertices, edges},
    {"vtof", 1, vertices, faces},
    {"vtoc", 1, vertices, cells},
    {"etof", 1, edges, faces},
    {"etoc", 1, edges, cells},
    {"ftov", 1, faces, vertices},
    {"ftoe", 1, faces, edges},
    {"ctov", 1, cells, vertices},
    {"ctoe", 1, cells, edges},
    {"ctof", 1, cells, faces},
    {"ctoc", 1, cells, cells}
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
    var numvertices = mesh.nvertices
    c.printf("Number of vertices = %i\n", numvertices)
    c.printf("vtov = %i\n", mesh.vtov.row_idx[9])
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
    for k, topo_elem in pairs(topo_elems) do
        local tsize = params["n"..topo_elem]
        elems[topo_elem] = utils.newtable(tsize, topo_elem)
    end
    for k, rel_tuple in pairs(mesh_rels) do
        local rel_name = rel_tuple[1]
        local x = rel_tuple[3]
        local y = rel_tuple[4]
        local tsize = mesh[rel_name].row_idx[params["n"..x]]
        rels[rel_name] = utils.newtable(tsize, rel_name)
        local rel_table = rels[rel_name]
        if rel_tuple[2] then
            rel_table.x = utils.newfield(elems[x])
            rel_table.y = utils.newfield(elems[y])
            rel_table.y:loadfrommemory(mesh[rel_name].values)
            rel_table:loadindexfrommemory("x", mesh[rel_name].row_idx)
            rel_table:dump()
        else
            -- TODO: Ask Zach about CRSConst
            rel_table.x = utils.newfield(elems[x])
            rel_table.y = utils.newfield(elems[y])
            rel_table.y:loadfrommemory(mesh[rel_name].values)
            rel_table:loadindexfrommemory("x", mesh[rel_name].row_idx)
            rel_table:dump()
        end
    end
end

-- Test code

print("Testing code ...")

local mesh = L.readMesh("relations/mesh.lmesh")

-- TODO: Speak to Zach and remove this - problem accessing mesh elements
local params = L.getMeshParams(mesh)

local elems, rels = L.initMeshRelations(mesh, params)

--[[
local function newmem(T,data)
    --this is probably not safe... but is just used here for debugging
    return terralib.cast(&T,terralib.new(T[#data],data))
end

local faces = utils.newtable(2,"faces")
local edges = utils.newtable(9,"edges")

local ftoe = utils.newtable(5,"ftoe")
ftoe.face = utils.newfield(faces)
ftoe.edge = utils.newfield(edges)

ftoe.edge:loadfrommemory(newmem(uint32,{0,1,5,7,8}))
ftoe:loadindexfrommemory("face",newmem(uint32,{0,2,5}))

ftoe:dump()
--]]
