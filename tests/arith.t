import "compiler.liszt"
require "tests.test"


local LMesh = L.require "domains.lmesh"
mesh = LMesh.Load("examples/mesh.lmesh")

test.fail_function(function()
  liszt kernel(v : mesh.vertices)
    var x = {1, 2, 3} ^ 4
  end
end, "invalid types")

test.fail_function(function()
  liszt kernel(v : mesh.vertices)
    var x = {1, 2, 3} ^ {4, 5, 6}
  end
end, "invalid types")

test.fail_function(function()
  liszt kernel(v : mesh.vertices)
    var x = 5 < true
  end
end, "invalid types")

local t = {}
local r = {}
test.fail_function(function()
  liszt kernel(v : mesh.vertices)
    var x = r < t
  end
end, "invalid types")
