require "include/liszt"

mesh  = LoadMesh("simple.obj")
pos   = mesh:fieldWithLabel(Vertex, Vector.type(3, Float), "position")
field = mesh:field(Face, Float, 0.0)

function main ()
	local com = Vector.new(float, 0.0, 0.0, 0.0)

	local sum_pos = liszt_kernel (v)
		com = com + pos(v)
	end

	mesh.cells.map(sum_pos)
	com = com / mesh.cells.size

	print("center of mass of mesh: " .. tostring(com))
end

main()
