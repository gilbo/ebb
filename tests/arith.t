import "compiler.liszt"
require "tests.test"


local LMesh = L.require "domains.lmesh"
mesh = LMesh.Load("examples/mesh.lmesh")

test.fail_function(function()
  local liszt t(v : mesh.vertices)
    var x = {1, 2, 3} + 4
  end
  mesh.vertices:map(t)
end, "incompatible types")

test.fail_function(function()
  local liszt t(v : mesh.vertices)
    var x = {1, 2, 3} / {4, 5, 6}
  end
  mesh.vertices:map(t)
end, "invalid types")

test.fail_function(function()
  local liszt t(v : mesh.vertices)
    var x = 5 < true
  end
  mesh.vertices:map(t)
end, "invalid types")

local t = {}
local r = {}
test.fail_function(function()
  local liszt t(v : mesh.vertices)
    var x = r < t
  end
  mesh.vertices:map(t)
end, "invalid types")
