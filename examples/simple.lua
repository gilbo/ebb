import "compiler.liszt"

local LMesh = L.require "domains.lmesh"
local PN = terralib.require 'compiler.pathname'
local mesh = LMesh.Load(PN.scriptdir():concat("mesh.lmesh"):tostring())
--local pos  = mesh:fieldWithLabel(Vertex, Vector(float, 3), "position")

function main ()
	-- declare a global to store the computed centroid of the mesh
	local com = L.NewGlobal(L.vector(L.float, 3), {0, 0, 0})

	-- compute centroid
	local sum_pos = liszt_kernel(v : mesh.vertices)
		com += v.position
	end
	--mesh.vertices:map(sum_pos)
	sum_pos(mesh.vertices)

	local center = com:value() / mesh.vertices:Size()

	-- output
	print("center of mass is: (" .. center.data[1] .. ", " .. center.data[2] .. ', ' .. center.data[3] .. ")")
end

main()
