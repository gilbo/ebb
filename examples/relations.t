local L = terralib.require("include/relations")

-- Test code
local relations = L.initMeshRelationsFromFile("examples/rmesh.lmesh")

for i,t in pairs(relations) do
    print("** Relation table **")
    t:dump()
end

