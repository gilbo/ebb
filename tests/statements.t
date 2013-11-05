import "compiler/liszt"
require "tests/test"

mesh = L.initMeshRelationsFromFile("examples/mesh.lmesh")
mesh.vertices:NewField('val', L.float)
mesh.vertices.val:LoadFromCallback(terra (mem : &float, i : uint) mem[0] = 1 end)
red = L.NewScalar(L.float, 0.0)

-- checking decl statement, if statement, proper scoping
local l = liszt_kernel (v in mesh.vertices)
	var y : float
	if v.val == 1.0 then
		y = 1.0
	else
		y = 0.0
	end

	red += y
end
l()

test.eq(red:value(), mesh.vertices._size)
