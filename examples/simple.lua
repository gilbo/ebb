import "compiler.liszt"

mesh = LoadMesh("examples/mesh.lmesh")
pos  = mesh:fieldWithLabel(Vertex, Vector(float, 3), "position")

function main ()
	-- declare a scalar to store the computed centroid of the mesh
	local com = mesh:scalar(Vector(float, 3), {0, 0, 0})

	-- compute centroid
	local sum_pos = liszt_kernel (v)
		com = com + pos(v)
	end
	mesh.vertices:map(sum_pos)

	local center = com:value() / mesh.vertices:size()

	-- output
	print("center of mass is: (" .. center.data[1] .. ", " .. center.data[2] .. ', ' .. center.data[3] .. ")")
end

main()
