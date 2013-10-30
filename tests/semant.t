import "compiler/liszt"
require "tests/test"

-- Field and scalar objs
mesh = LoadMesh("examples/mesh.lmesh")
f1   = mesh:field(Cell, float, 0)
f2   = mesh:field(Vertex, Vector(float, 3), {0, 0, 0})
s1   = mesh:scalar(int, 0)

-- Global lua var
checkthis1 = 1
local checkthis2 = 2

local a = {}
a.b     = {}
a.b.c   = {}
a.b.c.d = 4

-- ...let the testing begin!

-- Should fail b/c checkthis1 is not a scalar
function fail1() 
	local t = liszt_kernel (cell)
		checkthis1 = f1(cell)
	end
	mesh.cells:map(t)
end

-- Should fail when we re-assign a new value to x, since it originally
-- refers to a topological element
function fail2()
	local k = liszt_kernel (cell)
	    var x = cell
    	x = cell
	end
	mesh.cells:map(k)
end

-- Should fail because we do not allow assignments to fields
-- (only to indexed fields, scalars, and local vars)
function fail3()
	local k = liszt_kernel (cell)
		f1 = 5
	end
	mesh.cells:map(k)
end

-- Should fail because we do not allow the user to alias fields,
-- or any other entity that would confuse stencil generation, in the kernel
function fail4()
	local k = liszt_kernel (cell)
		var z = f1
	end
	mesh.cells:map(k)
end

function fail5()
	local k = liszt_kernel(cell)
		undefined = 3
	end
	mesh.cells:map(k)
end

function fail6()
	local k = liszt_kernel (cell)
		var floatvar = 2 + 3.3
		floatvar = true
	end
	mesh.cells:map(k)
end

function fail7()
	local k = liszt_kernel (cell)
		var local7 = 2.0
		do
			var local8 = 2
		end

		var cond = true
		while cond ~= cond do
			local8 = 3
			local7 = 4.5
		end
	end
	mesh.cells:map(k)
end

function fail8()
	local k = liszt_kernel (cell)
		if 4 < 2 then
			var local8 = true
		-- Should fail here, since local8 is not defined in this scope
		elseif local8 then
			var local9 = true
		elseif 4 < 3 then
			var local9 = 2
		else
			var local10 = local7
		end
	end
	mesh.cells:map(k)
end

function fail9()
	local k = liszt_kernel (cell)
		var local1 = 3.4
		do
			var local1 = true
			local1 = 2.0 -- should fail, local1 is of type bool
		end
	end
	mesh.cells:map(k)
end

function fail10()
	local k = liszt_kernel (cell)
		assert(4 == true)
	end
	mesh.cells:map(k)
end

function fail11()
	local v = Vector.new(float, {1, 1, 1})
	local k = liszt_kernel (cell)
		assert(v == 1)
	end
	mesh.cells:map(k)
end

function fail12()
	local k = liszt_kernel (cell)
		a.b = 12
	end
	mesh.cells:map(k)
end

function fail13()
	local k = liszt_kernel (cell)
		var v
		if false then
			v = true
		end
		v = 5
	end
	mesh.cells:map(k)
end

-- need typechecker fail test
test.fail_function(fail1, "The left side of an assignment must be an lvalue")
test.fail_function(fail2, "The left side of an assignment must be an lvalue")
test.fail_function(fail3, "The left side of an assignment must be an lvalue")
test.fail_function(fail4, "Can only assign")
test.fail_function(fail5, "Variable 'undefined' is not defined")
test.fail_function(fail6, "invalid conversion from bool to float")
test.fail_function(fail7, "Variable 'local8' is not defined")
test.fail_function(fail8, "Variable 'local8' is not defined")
test.fail_function(fail9, "invalid conversion from int to bool")
test.fail_function(fail10, "invalid types for operator")
test.fail_function(fail11, "Expected a boolean")
test.fail_function(fail12, "Cannot use lua tables as Liszt values")
test.fail_function(fail13, "invalid conversion from int to bool")


-- Nothing should fail in this kernel:
local k = liszt_kernel (cell)
    f1(cell) = 3.0
    var lc = 4.0

	var local1 = a.b.c.d
	var local2 = 2.0
	var local3 = local1 + local2
	var local5 = 2 + 3.3
	var local4 = checkthis1 + checkthis2
	var local7 = 8 <= 9

	3 + 4

	do
		var local1 = true
	end
	local1 = 3.0
	var local1 = false

	var local9 = 0
	for i = 1, 4, 1 do
		local9 += i * i
	end
end
mesh.cells:map(k)
