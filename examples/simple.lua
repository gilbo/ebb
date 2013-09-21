import "compiler/liszt"

mesh   = LoadMesh("examples/mesh.lmesh")
pos    = mesh:fieldWithLabel(Vertex, Vector.type(float, 3), "position")
field  = mesh:field(Face, float, 0.0)
field2 = mesh:field(Face, float, 0.0)
field3 = mesh:field(Face, float, 0.0)

local a = global(int, 6)

function main ()
	local com = mesh:scalar(Vector.type(float, 3))--Vector.new(float, {0.0, 0.0, 0.0})
	-- com.setTo({0,0,0})

	local upval = 5

	local vv = Vector.new(float, {1,2,3})

	local sum_pos = liszt_kernel (f)
		field(f) = field(f) - 3 + 243.3
		field2(f) = field3(f)
		-- com = com + 1
	end

	mesh.vertices:map(sum_pos)
	mesh.vertices:map(sum_pos)
	-- com = com / mesh.vertices.size()
	-- print("center of mass of mesh: " .. tostring(com))
end

main()
