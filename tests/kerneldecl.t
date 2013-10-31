import "compiler/liszt"
local relations = terralib.require "include/relations"
local mesh_relations  = relations.initMeshRelationsFromFile("examples/mesh.lmesh")
local faces = mesh_relations.faces

local pass_kernel = liszt_kernel(f in faces)
end
pass_kernel:generate()
