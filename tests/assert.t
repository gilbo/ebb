import "compiler/liszt"

mesh  = L.initMeshRelationsFromFile("examples/mesh.lmesh")

local pass_assert = liszt_kernel(f in mesh.faces)
    L.assert(true)
end
pass_assert()
