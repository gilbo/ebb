import "compiler.liszt"
local LMesh = terralib.require "compiler.lmesh"
local M = LMesh.Load("examples/mesh.lmesh")


function shift(x,y,z)
	local pass_kernel = liszt_kernel(v in M.vertices)
	    v.position = {x,y,z}
	end
	pass_kernel()
end

function check(x,y,z)
	for i = 0, M.vertices._size - 1 do
		local v = M.vertices.position.data[i]
		assert(v[0] == x)
		assert(v[1] == y)
		assert(v[2] == z)
	end
end

shift(0,0,0)
check(0,0,0)

shift(5,5,5)
check(5,5,5)

shift(-1,6,3)
check(-1,6,3)
