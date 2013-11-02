import "compiler/liszt"
mesh = L.initMeshRelationsFromFile("examples/mesh.lmesh")

local fail_assert = liszt_kernel(f in mesh.faces)
    L.assert(2 + 2 == 5)
end
fail_assert()

