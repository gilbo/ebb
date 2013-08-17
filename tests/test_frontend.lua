require "include/liszt"
import "compiler/liszt"
mesh = LoadMesh("examples/mesh.lmesh")

position = mesh:fieldWithLabel(Vertex, Vector.type(float,3), "position")
-- field    = mesh:field(Cell, Vector.type(int,2), Vector.new(int, 2, {0,0}))
field2   = mesh:field(Cell,   int, 0)
myid     = mesh:field(Vertex, int, 0) --,Vec[_2,Int]](Vec(0,0))

function main ()
	-- init statements, binary expressions, 
	-- field lookups (lvalues), function calls (multiple arguments)
	local b = liszt_kernel (cell) 
		var a = Vector.new(int, 1, 2)
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


	-- TODO:
    -- add parsing for do end blocks, for blocks, tupled assignments?
    -- add tests/implementation for while loop, repeat loop, pretty much all statements!

    --generic for
        local f = liszt_kernel (cell)
                for cell in mesh.cells do
                        field2(cell) = field2(cell) + 12
                end
                for cell = 1,3 do
                    t = 1
                end
                for cell = 1,3,1 do
                    t = 1
                end
        end
--        terralib.tree.printraw(f)
    --[[ 
    --generic for
	local fortest = liszt_kernel (cell)
		field(cell) = Vec(1, 2)

		for v in vertices(cell) do
			myid(v) = myid(v) + 12
		end
	end

	mesh.cells().map(
		liszt_kernel (cell)
			print(cell, field(cell))
			print(a)
		end
	)
	print(a)

	-- if we ever add parsing for reduction operator?
	mesh.cells().map(
		liszt_kernel (cell)
			a += Vector(1, 2)
		end
	)
	--]]
	


	--[[
	-- print parsed ASTs for debugging purposes:
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
	]]--


