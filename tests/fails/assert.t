import "compiler/liszt"
mesh = L.initMeshRelationsFromFile("examples/mesh.lmesh")

local fail_assert = liszt_kernel(f in mesh.faces)
    L.assert(false)
end
fail_assert()
