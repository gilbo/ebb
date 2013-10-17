local mesh_rels = terralib.require("runtime/mesh_relations")

-- Test code
print("Testing code ...")
local rels = mesh_rels.initMeshRelationsFromFile("examples/mesh.lmesh")
for i,t in pairs(rels) do
    print("** Relation table **")
    t:dump()
end
