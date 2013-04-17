require "include/liszt"
mesh = LoadMesh("some file...")

position = mesh.fieldWithLabel(Vertex, Vector.type(3,Float), "position")
field    = mesh.field(Cell, Vector.type(2,Int), Vector.new(0,0))
field2   = mesh.field(Cell, Int, 0)
myid     = mesh.field(Vertex, Int, 0) --,Vec[_2,Int]](Vec(0,0))

function main ()
	local a = Vector.new(Int, 1, 2)
	local b = liszt_kernel (cell) 
		local a = 3 + 4 + 5
		field(cell) = a
	end

	terralib.tree.printraw(b)

--[[
	print("b: " .. tostring(b) .. " " .. b.name)
	for i=1,#b.statements do
		print(i .. " " .. b.statements[i].name)
	end
--]]
	
--[[
	mesh.cells().map(
		liszt_kernel (cell)
			field(cell) = Vec(1, 2)

			for v in vertices(cell) do
				myid(v) += 12
			end
		end
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
	
end

main()