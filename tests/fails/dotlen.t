import "compiler/liszt"

local assert = L.assert
local dot = L.dot
mesh = LoadMesh("examples/mesh.lmesh")

local v1 = Vector.new(L.float, {1, 2, 3})
local v2 = Vector.new(L.float, {5, 7})

local test_dot = liszt_kernel(f)
    assert(dot(v1, v2) == 19)
end

mesh.faces:map(test_dot)
