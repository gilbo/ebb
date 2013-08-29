require "include/liszt"
import "compiler/liszt"

mesh  = LoadMesh("examples/mesh.lmesh")
pos   = mesh:fieldWithLabel(Vertex, Vector(float, 3), "position")
field = mesh:field(Face, float, 0.0)

function main ()
	local com = mesh:scalar(Vector.type(float, 3))--Vector.new(float, {0.0, 0.0, 0.0})
	com = {0, 0, 0}

	local upval = 5

	local sum_pos = liszt_kernel (v)
		-- com = com + pos(v)
		var x = 3
		var y = x + 2
		var z = upval
		--upval = 2
	end

	print("sum_pos:")
	sum_pos:printpretty()

	-- kernel application
	mesh.vertices.map(sum_pos)

	--com = com / mesh.vertices.size()
	-- print("center of mass of mesh: " .. tostring(com))
end

main()
