require "include/liszt"
mesh = LoadMesh("examples/mesh.lmesh")

position = mesh:fieldWithLabel(Vertex, Vector.type(float,3), "position")
-- field    = mesh:field(Cell, Vector.type(int,2), Vector.new(0,0))
field2   = mesh:field(Cell,   int, 0)
myid     = mesh:field(Vertex, int, 0) --,Vec[_2,Int]](Vec(0,0))

v1 = { value = 1, kind = 'int'}

function addvar1()
	local v3 = 3
end

function main ()
	-- init statements, binary expressions, 
	-- field lookups (lvalues), function calls (multiple arguments)
	
	local v2 = 2

	addvar1()

	local b = liszt_kernel (cell) 

		var a = Vector.new(Int, 1, 2)
		var b = 3 + 4 * 5
		var c = 6 / 7 + -8 ^ 9
		field(cell) = b * a
	end

	local c = liszt_kernel (cell)
		var x = a.b + c(d)
	end

	-- if statement, field index, init statements
	local d = liszt_kernel (cell)
		if true then
			var a = field(cell)
		else
			var a = field2(cell)
		end
	end

	-- table lookup and field index:
	local e = liszt_kernel (cell)
		table.field(cell) = x*(5-3)
	end

	-- break statement
	local f = liszt_kernel (cell)
		table.field(cell) = x*(5-3)
		var x = y
		break
	end

	-- repeat, do statements
	liszt_kernel (cell)
		repeat
			var x
			var y = x
		until true

		do
			field(cell) = b * a
		end
	end

	-- while statement
	liszt_kernel (v)
		while false do
			var x
			var y
		end
	end

end

main()
