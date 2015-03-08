import "compiler.liszt"
require "tests/test"
local LMesh = L.require "domains.lmesh"
local mesh = LMesh.Load("examples/mesh.lmesh")

test.fail_function(function()
  local liszt t(v : mesh.vertices)
  	var x = {1, 2, true}
  end
  mesh.vertices:map(t)
end, "must be of the same type")

test.fail_function(function()
  local liszt t(v : mesh.vertices)
  	var x = {1, 2, {2, 3}}
  end
  mesh.vertices:map(t)
end, "can only contain scalar values")
