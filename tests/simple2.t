import 'ebb'
local L = require 'ebblib'
require "tests/test"

local ioOff = require 'ebb.domains.ioOff'
local mesh  = ioOff.LoadTrimesh('tests/octa.off')
local com   = L.Global(L.vector(L.float, 3), {0, 0, 0})

function center_of_mass ()
	com:set({0,0,0})
	local sum_pos = ebb (v : mesh.vertices)
		com += L.vec3f(v.pos)
	end
	mesh.vertices:foreach(sum_pos)
	local sz = mesh.vertices:Size()
	local c  = com:get()
	return { c[1]/sz, c[2]/sz, c[3]/sz }
end

local function displace_mesh (delta_x, delta_y, delta_z)
	local d = L.Constant(L.vec3d, {delta_x, delta_y, delta_z})
	local dk = ebb (v : mesh.vertices)
		v.pos += d
	end
	mesh.vertices:foreach(dk)
end

test.aeq(center_of_mass(), {0, 0, 0})

displace_mesh(1, 0, 0)
test.aeq(center_of_mass(), {1, 0, 0})

displace_mesh(.4, .5, .8)
test.fuzzy_aeq(center_of_mass(), {1.4, .5, .8})

displace_mesh(-3, -4, -5)
test.fuzzy_aeq(center_of_mass(), {-1.6, -3.5, -4.2})

