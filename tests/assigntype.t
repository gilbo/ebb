import "compiler.liszt"
require "tests/test"

local LMesh = L.require "domains.lmesh"
local mesh = LMesh.Load("examples/mesh.lmesh")

sf   = L.Global(L.float, 0.0)

test.fail_function(function()
  liszt kernel test (c : mesh.cells)
    sf.a = 1
  end
end, "select operator not supported")
