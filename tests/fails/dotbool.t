import "compiler/liszt"
mesh = L.initMeshRelationsFromFile("examples/mesh.lmesh")

local assert, dot = L.assert, L.dot

local v1 = L.NewVector(L.float, {1, 2, 3})
local v2 = L.NewVector(L.bool, {true, true, false})

local test_dot = liszt_kernel(f in mesh.faces)
    assert(dot(v1, v2) == 52)
end

test_dot()
