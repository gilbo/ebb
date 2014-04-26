import "compiler.liszt"
require "tests/test"

local LMesh = L.require "domains.lmesh"
local mesh = LMesh.Load("examples/mesh.lmesh")
local com = L.NewGlobal(L.vector(L.double, 3), {0, 0, 0})

function center_of_mass ()
	com:setTo({0,0,0})
	local sum_pos = liszt kernel (v : mesh.vertices)
		com += v.position
	end
	sum_pos(mesh.vertices)
	return com:value() / mesh.vertices._size
end

local function displace_mesh (delta_x, delta_y, delta_z)
	local d = L.NewVector(L.double, {delta_x, delta_y, delta_z})
	local dk = liszt kernel (v : mesh.vertices)
		v.position += d
	end
	dk(mesh.vertices)
end

test.aeq(center_of_mass().data, {0, 0, 0})

displace_mesh(1, 0, 0)
test.aeq(center_of_mass().data, {1, 0, 0})

displace_mesh(.4, .5, .8)
test.fuzzy_aeq(center_of_mass().data, {1.4, .5, .8})

displace_mesh(-3, -4, -5)
test.fuzzy_aeq(center_of_mass().data, {-1.6, -3.5, -4.2})

