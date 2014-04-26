import "compiler.liszt"
require "tests/test"
local LMesh = L.require "domains.lmesh"
local mesh = LMesh.Load("examples/mesh.lmesh")

test.fail_function(function()
  liszt kernel(v : mesh.vertices)
  	var x = {1, 2, true}
  end
end, "must be of the same type")

test.fail_function(function()
  liszt kernel(v : mesh.vertices)
  	var x = {1, 2, {2, 3}}
  end
end, "can only contain values of boolean or numeric type")
