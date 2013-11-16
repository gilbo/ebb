import "compiler/liszt"
require "tests/test"

local LMesh = terralib.require("compiler/liblmesh")
local mesh = LMesh.Load("examples/mesh.lmesh")

sf   = L.NewScalar(L.float, 0.0)

local k = liszt_kernel (c in mesh.cells)
	sf.a = 1
end

test.fail_function(k, "select operator not supported")
