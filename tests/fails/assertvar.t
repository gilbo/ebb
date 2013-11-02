import "compiler/liszt"
mesh = L.initMeshRelationsFromFile("examples/mesh.lmesh")

local fail_assert = liszt_kernel(f in mesh.faces)
    var x = 5
    L.assert(x == 4)
end

fail_assert()
