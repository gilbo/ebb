require "include/liszt"
mesh = LoadMesh("some file...")

position = mesh.fieldWithLabel(Vertex, Vector.type(3,Float), "position")
field    = mesh.field(Cell, Vector.type(2,Int), Vector.new(0,0))
field2   = mesh.field(Cell,   Int, 0)
myid     = mesh.field(Vertex, Int, 0) --,Vec[_2,Int]](Vec(0,0))

function main ()
	local b = liszt_kernel (cell) 
		var a = Vector.new(Int, 1, 2)
		var b = 3 + 4 + 5
		field(cell) = b * a
	end

	local c = liszt_kernel (cell)
		var x = a.b + c(d)
	end

	local d = liszt_kernel (cell)
		if true then
			var a = field(cell)
		else
			var a = field2(cell)
		end
	end

	local e = liszt_kernel (cell)
		table.field(cell) = x*(5-3)
	end

	mesh.cells().map(
		liszt_kernel (cell)
			field(cell) = Vec(1, 2)

			for v in vertices(cell) do
				myid(v) = myid(v) + 12
			end
		end
	)

-- add parsing for do end blocks, for blocks, tupled assignments?
-- add tests for while loop, repeat loop, pretty much all statements!

	print("b:")
	b:pretty_print()
	print()
	print("c:")
	c:pretty_print()
	print()
	print("d:")
	d:pretty_print()
	print()
	print("e:")
	e:pretty_print()
	print()



end

main()
	
--[[
	)
	mesh.cells().map(
		liszt_kernel (cell)
			print(cell, field(cell))
			print(a)
		end
	)
	print(a)

	mesh.cells().map(
		liszt_kernel (cell)
			a += Vector(1, 2)
		end
	)

	mesh.vertices().map(
		liszt_kernel (v)
			myid(v) += 1
		end
	)

	mesh.vertices().map(
		liszt_kernel (v)
			for c in cells(v)
				field2(c) += 1
			end
		end
	)
--]]
	
