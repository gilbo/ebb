import "compiler.liszt"
local LMesh = L.require "domains.lmesh"
local M = LMesh.Load("examples/mesh.lmesh")


function shift(x,y,z)
	local liszt pass_func (v : M.vertices)
	    v.position = {x,y,z}
	end
	M.vertices:map(pass_func)
end

function check(x,y,z)
	M.vertices.position:MoveTo(L.CPU)
	for i = 0, M.vertices:Size() - 1 do
		local v = M.vertices.position:DataPtr()[i]
		assert(v.d[0] == x)
		assert(v.d[1] == y)
		assert(v.d[2] == z)
	end
	M.vertices.position:MoveTo(L.default_processor)
end

shift(0,0,0)
check(0,0,0)

shift(5,5,5)
check(5,5,5)

shift(-1,6,3)
check(-1,6,3)
