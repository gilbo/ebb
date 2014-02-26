import "compiler.liszt"
require "tests/test"

local LMesh = terralib.require "compiler.lmesh"
local mesh = LMesh.Load("examples/mesh.lmesh")

sf   = L.NewScalar(L.float, 0.0)

test.fail_function(function()
  liszt_kernel (c : mesh.cells)
    sf.a = 1
  end
end, "select operator not supported")
