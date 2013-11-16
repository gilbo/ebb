import "compiler/liszt"
require "tests/test"
local LMesh = terralib.require("compiler/liblmesh")
local mesh = LMesh.Load("examples/mesh.lmesh")

local vk = liszt_kernel(v in mesh.vertices)
	var x = {1, 2, true}
end
test.fail_function(vk, "must be of the same type")

local vk2 = liszt_kernel(v in mesh.vertices)
	var x = {1, 2, {2, 3}}
end
test.fail_function(vk2, "can only contain values of boolean or numeric type")
