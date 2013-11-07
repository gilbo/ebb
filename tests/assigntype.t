import "compiler/liszt"
require "tests/test"

mesh = L.initMeshRelationsFromFile("examples/mesh.lmesh")
sf   = L.NewScalar(L.float, 0.0)

local k = liszt_kernel (c in mesh.cells)
	sf.a = 1
end

test.fail_function(k, "select operator not supported")
