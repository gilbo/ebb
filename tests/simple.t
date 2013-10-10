import "compiler/liszt"

mesh  = LoadMesh("examples/mesh.lmesh")
pos   = mesh:fieldWithLabel(Vertex, Vector.type(float, 3), "position")
field = mesh:field(Face, float, 0.0)

local a = global(int, 43)

function main ()
	local com = mesh:scalar(Vector.type(float, 3), {0, 0, 0})--Vector.new(float, {0.0, 0.0, 0.0})
	-- com = {0, 0, 0}

	local upval = 5

	local vv = Vector.new(float, {1,2,3})

	local test_kernel = liszt_kernel (v)
		-- BinaryOp, UnaryOp, InitStatement, Number, Bool, and RValue codegen tests
		var x = 9
		assert(x == 9)
		var xx = x - 4
		assert(xx == 5)
		var y = x + -(6 * 3)
		assert(y == -9)
		var z = upval
		assert(z == 5)
		var b = a
		assert(b == 43)
		var q = true
		assert(q == true)
		var x = q  -- Also, test re-declaring variables (symbols for 'x' should now be different)
		assert(x == true)
		var z = not x
		assert(z == false)
		var y = not z or x
		assert(y == true)
		var z = not true and false or true
		assert(z == true)

		-- Codegen for vectors (do types propogate correctly?)
		var x = 3 * vv
		var y = vv / 4.2
		var z = x + y
		var a = y - x

		-- IfStatement tests
		var x = 3
		var y = 4

		if q then
			var x = 5
			assert(x == 5)
			y = x
			assert(y == 5)
		else
			y = 9
			assert(y == 9)
		end
		assert(x == 3)
		assert(y == 5)
		y = x
		assert(y == 3)

		if y == x * 2 then
			x = 4
		elseif y == x then
			x = 5
		end
		assert(x == 5)

		if y == x * 2 then
			x = 4
		end
		assert(x == 5)

		var a = 3
		if y == x * 2 then
			x = 4
			assert(false)
		elseif y == x then
			x = 5
			assert(false)
		else
			var a = true
			assert(a == true)
		end
		assert(a == 3)

		-- While Statement tests --
		-- if either of these while statements doesn't terminate, then our codegen scoping is wrong!
		var a = true
		while a do
			a = false
		end

		var b = true
		while b ~= a do
			a = true
			var b = false
		end
		b = false

		var x = true
		var y = 3
		do
			var x = false
			assert(x == false)
			if x then
				y = 5
				assert(false)
			else
				y = 4
				assert(y == 4)
			end
		end
		assert(x == true)
		assert(y == 4)

		-- RepeatStatement tests -- 
		var x = 0
		var y = 0
		-- again, if this doesn't terminate, then our scoping is wrong
		repeat
			y = y + 1
			var x = 5
		until x == 5
		assert(x == 0)

		y = 0
		repeat
			y = y + 1
		until y == 5
		assert(y == 5)

		-- DeclStatement tests --
		var c
		c = 12

		var x
		x = true

		-- this should be fine
		var y
		var y

		-- should be able to assign w/an expression after declaring,
		-- checking with var e to make sure expressions are the same.
		var zip = 43.3
		var doo
		doo = zip * c
		var dah = zip * c
		assert(doo == dah)
		
		var a = 43.3
		var d
		d = a * vv
		var e = a * vv
		-- assert(d == e) -- vector equality not supported?

		-- Numeric for tests: --
		for i = 1, 5 do
			var x = i
			if x == 3 then break end
		end
		assert(x == true)
	end

	mesh.vertices:map(test_kernel)
end

main()
