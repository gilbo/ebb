require "include/liszt"
import "compiler/liszt"

mesh  = LoadMesh("examples/mesh.lmesh")
pos   = mesh:fieldWithLabel(Vertex, Vector.type(float, 3), "position")
field = mesh:field(Face, float, 0.0)

function main ()
	local com = Vector.new(float, {0.0, 0.0, 0.0})

	local sum_pos = liszt_kernel (v)
	for v in mesh.vertices do
		com = com + pos(v)
	end
	end

	-- kernel application
	mesh.vertices.map(sum_pos)

	com = com / mesh.vertices.size()

	print("center of mass of mesh: " .. tostring(com))
end

main()
