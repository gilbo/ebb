import "compiler.liszt"
require "tests/test"

local LMesh = terralib.require "compiler.lmesh"
local mesh = LMesh.Load("examples/mesh.lmesh")

sf   = L.NewScalar(L.float, 0.0)

local k = liszt_kernel (c : mesh.cells)
	sf.a = 1
end

test.fail_kernel(k, mesh.cells, "select operator not supported")
