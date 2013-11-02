import "compiler/liszt"
require "tests/test"

local mesh = L.initMeshRelationsFromFile("examples/mesh.lmesh")
local com = L.NewScalar(L.vector(L.float, 3), {0, 0, 0})

function center_of_mass ()
	com:setTo({0,0,0})
	local sum_pos = liszt_kernel (v in mesh.vertices)
		com += v.position
	end
	sum_pos()
	return com:value() / mesh.vertices._size
end

local function displace_mesh (delta_x, delta_y, delta_z)
	local d = L.NewVector(L.float, {delta_x, delta_y, delta_z})
	local dk = liszt_kernel (v in mesh.vertices)
		v.position += d
	end
	dk()
end

test.aeq(center_of_mass().data, {0, 0, 0})

displace_mesh(1, 0, 0)
test.aeq(center_of_mass().data, {1, 0, 0})

displace_mesh(.4, .5, .8)
test.fuzzy_aeq(center_of_mass().data, {1.4, .5, .8})

displace_mesh(-3, -4, -5)
test.fuzzy_aeq(center_of_mass().data, {-1.6, -3.5, -4.2})

