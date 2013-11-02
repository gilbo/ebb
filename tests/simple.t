import "compiler/liszt"

mesh  = LoadMesh("examples/mesh.lmesh")
pos   = mesh:fieldWithLabel(L.vertex, L.vector(L.float, 3), "position")
field = mesh:field(L.face, L.float, 0.0)

local assert = L.assert

--local a = global(int, 43)
local a = 43

function main ()
	local com   = mesh:scalar(L.vector(L.float, 3), {0, 0, 0})--Vector.new(float, {0.0, 0.0, 0.0})
	local upval = 5
	local vv    = Vector.new(L.float, {1,2,3})

	local test_bool = liszt_kernel (v)
		var q = true
		var x = q  -- Also, test re-declaring variables (symbols for 'x' should now be different)
		var z = not q
		var t = not not q
		var y = z == false
		assert(x == true)
		assert(q == true)
		assert(z == false)
		assert(y == true)
		assert(t == q)
	end
	mesh.vertices:map(test_bool)

	local test_decls = liszt_kernel(v)
		-- DeclStatement tests --
		var c : int
		c = 12

		var x : bool
		x = true

		var z : bool
		do z = true end
		assert(z == true)

		var z : int
		do z = 4 end
		assert(z == 4)

		-- this should be fine
		var y : int
		var y : int

		-- should be able to assign w/an expression after declaring,
		-- checking with var e to make sure expressions are the same.
		var zip = 43.3
		var doo : float
		doo = zip * c
		var dah = zip * c
		var x = doo == dah
		assert(doo == dah)
	end
	mesh.vertices:map(test_decls)

	local test_conditionals = liszt_kernel (v)
		-- IfStatement tests
		var q = true
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
	end
	mesh.vertices:map(test_conditionals)

	local test_arith = liszt_kernel (v)
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

		var a = 43.3
		var d : Vector(float, 3)
		d = a * vv
		var e = a * vv
		-- assert(d == e) -- vector equality not supported?
	end
	mesh.vertices:map(test_arith)

	local test_while = liszt_kernel(v)
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
	end
	mesh.vertices:map(test_while)

	local test_do = liszt_kernel (v)
		var b = false
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
	end
	mesh.vertices:map(test_do)

	local test_repeat = liszt_kernel (v)
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
	end

	local test_for = liszt_kernel (v)
		-- Numeric for tests: --
		var x = true
		for i = 1, 5 do
			var x = i
			if x == 3 then break end
		end
		assert(x == true)
	end
	mesh.vertices:map(test_for)
end

main()
