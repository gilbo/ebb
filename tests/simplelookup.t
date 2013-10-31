import "compiler/liszt"
local relations = terralib.require "include/relations"
local mesh_relations  = relations.initMeshRelationsFromFile("examples/mesh.lmesh")

local pass_kernel = liszt_kernel(v in mesh_relations.vertices)
    v.position = {0,0,0}
end
pass_kernel:generate()
