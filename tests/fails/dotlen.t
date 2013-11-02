import "compiler/liszt"

local assert, dot = L.assert, L.dot
mesh = L.initMeshRelationsFromFile("examples/mesh.lmesh")

local v1 = L.NewVector(L.float, {1, 2, 3})
local v2 = L.NewVector(L.float, {5, 7})

local test_dot = liszt_kernel(f in mesh.faces)
    assert(dot(v1, v2) == 19)
end
test_dot()
