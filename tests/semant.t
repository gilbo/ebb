import "compiler/liszt"
require "tests/test"

local assert = L.assert

-- Field and scalar objs
mesh = LoadMesh("examples/mesh.lmesh")
f1   = mesh:field(L.cell,   L.float, 0)
f2   = mesh:field(L.vertex, L.vector(L.float, 3), {0, 0, 0})
s1   = mesh:scalar(L.int, 0)

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
	local v = Vector.new(L.float, {1, 1, 1})
	local k = liszt_kernel (cell)
		assert(v)
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
		var v : bool
		if false then
			v = true
		end
		v = 5
	end
	mesh.cells:map(k)
end

function fail14()
	local tbl = {}
	local k = liszt_kernel (cell)
		var x = 3 + tbl
	end
	mesh.cells:map(k)
end

function fail15()
	local tbl = {}
	local k = liszt_kernel (cell)
		var x = tbl
	end
	mesh.cells:map(k)
end

function fail16()
	local tbl = {}
	local k = liszt_kernel (cell)
		tbl.x = 4
	end
	mesh.cells:map(k)
end

function fail17()
	local tbl = {x={}}
	local k = liszt_kernel (cell)
		tbl.x.y = 4
	end
	mesh.cells:map(k)
end

function fail18()
	local tbl = {x={y=4}}
	local k = liszt_kernel (cell)
		var x = tbl.x
	end
	mesh.cells:map(k)
end


-- need typechecker fail test
test.fail_function(fail1,  "assignments in a Liszt kernel are only valid")
test.fail_function(fail2,  "assignments in a Liszt kernel are only valid")
test.fail_function(fail3,  "assignments in a Liszt kernel are only valid")
test.fail_function(fail4,  "can only assign")
test.fail_function(fail5,  "variable 'undefined' is not defined")
test.fail_function(fail6,  "invalid conversion from bool to float")
test.fail_function(fail7,  "variable 'local8' is not defined")
test.fail_function(fail8,  "variable 'local8' is not defined")
test.fail_function(fail9,  "invalid conversion from int to bool")
test.fail_function(fail10, "invalid types for operator")
test.fail_function(fail11, "expected a boolean")
test.fail_function(fail12, "assignments in a Liszt kernel are only valid")
test.fail_function(fail13, "invalid conversion from int to bool")
test.fail_function(fail14, "invalid types")
test.fail_function(fail15, "can only assign")
test.fail_function(fail16, "lua table tbl does not have member 'x'")
test.fail_function(fail17, "lua table tbl.x does not have member 'y'")
test.fail_function(fail18, "can only assign")

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
