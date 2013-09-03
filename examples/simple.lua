require "include/liszt"
import "compiler/liszt"

mesh  = LoadMesh("examples/mesh.lmesh")
pos   = mesh:fieldWithLabel(Vertex, Vector.type(float, 3), "position")
field = mesh:field(Face, float, 0.0)

local a = global(int, 6)

function main ()
	local com = mesh:scalar(Vector.type(float, 3))--Vector.new(float, {0.0, 0.0, 0.0})
	com = {0, 0, 0}

	local upval = 5

	local vv = Vector.new(float, {1,2,3})

	local sum_pos = liszt_kernel (v)
		var x = 9
		var xx = x - 4
		var y = x + -(6 * 3)
		var z = upval
		var b = a
		var q = true
		var x = q
		var z = not x
		var y = not z or x

		var x = 3 * vv
		var y = vv / 4.2
		var z = x + y
		var a = y - x

		var x = 3
		var y = 4
--[[
		if q then
			var x = 5
			y = x
		else
			y = 9
		end
		y = x
]]
		if y == x * 2 then
			x = 4
		elseif y == x then
			x = 5
		end
--[[
		if y == x * 2 then
			x = 4
		end

		if y == x * 2 then
			x = 4
		elseif y == x then
			x = 5
		else
			var a = true
		end
]]--


	end

	print("sum_pos:")
	sum_pos:printpretty()
	-- call to make sure it works!
	sum_pos()

	-- kernel application
	-- mesh.vertices.map(sum_pos)

	--com = com / mesh.vertices.size()
	-- print("center of mass of mesh: " .. tostring(com))
end

main()
