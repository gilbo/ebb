import "compiler.liszt"
local LMesh = terralib.require "compiler.lmesh"
local M = LMesh.Load("examples/mesh.lmesh")

local V = M.vertices
local P = V.position

loc_data = {}
for i = 0, V._size - 1 do
	loc_data[i] = {P.data[i].d[0], P.data[i].d[1], P.data[i].d[2]}
end

function shift(x,y,z)
	local shift_kernel = liszt_kernel(v : M.vertices)
	    v.position += {x,y,z}
	end
	shift_kernel(M.vertices)

	for i = 0, V._size - 1 do
		local v = P.data[i]
		local d = loc_data[i]

		d[1] = d[1] + x
		d[2] = d[2] + y
		d[3] = d[3] + z

		--print("Pos " .. tostring(i) .. ': (' .. tostring(v[0]) .. ', ' .. tostring(v[1]) .. ', ' .. tostring(v[2]) .. ')')
		--print("Loc " .. tostring(i) .. ': (' .. tostring(d[1]) .. ', ' .. tostring(d[2]) .. ', ' .. tostring(d[3]) .. ')')
		assert(v.d[0] == d[1])
		assert(v.d[1] == d[2])
		assert(v.d[2] == d[3])
	end
end

shift(0,0,0)
shift(5,5,5)
shift(-1,6,3)

