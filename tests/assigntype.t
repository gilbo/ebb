import "compiler.liszt"
require "tests/test"

local LMesh = L.require "domains.lmesh"
local mesh = LMesh.Load("examples/mesh.lmesh")

sf   = L.Global(L.float, 0.0)

test.fail_function(function()
  local liszt test (c : mesh.cells)
    sf.a = 1
  end
  mesh.cells:map(test)
end, "select operator not supported")
