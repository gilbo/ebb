mesh = LoadMesh("some file...")

-- not sure what to do with field construction syntax...
position = mesh.fieldWithLabel(Vertex, Vec(3,Float), "position")
field    = mesh.field(Cell, Vec(2,Int), Vec(0,0))
field2   = mesh.field(Cell, Int, 0)
myid     = mesh.field(Vertex, Int, 0) --,Vec[_2,Int]](Vec(0,0))

function main () {
	print("hello")

	local a = LisztVec(1,2)

	mesh.cells().map(
		liszt_kernel (cell)
			print(cell)
			field(cell) = LisztVec(ID(cell), ID(cell)+ 1)

			for v in vertices(cell) do
				print(position(v))
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
			a += LisztVec(1, 2)
		end
	)

	mesh.vertices().map(
		liszt_kernel (v)
			myid(v) += 1
		end
	)

	--	...

	mesh.vertices().map(
		liszt_kernel (v)
			for c in cells(v)
				field2(c) += 1
			end
		end
	)

	-- ...
}