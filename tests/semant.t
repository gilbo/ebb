import "compiler/liszt"
require "tests/test"

local lassert, lprint = L.assert, L.print


---------------------------
-- Field and scalar objs --
---------------------------
mesh = L.initMeshRelationsFromFile("examples/mesh.lmesh")
mesh.cells:NewField('f1', L.float)
mesh.cells:NewField('f2', L.vector(L.float, 3))
s1 = L.NewScalar(L.int, 0)


------------------------
-- Initialize fields: --
------------------------
mesh.cells.f1:LoadFromCallback(terra (mem : &float, i : int) mem[0] = 0 end)
mesh.cells.f2:LoadFromCallback(
	terra (mem : &vector(float, 3), i : int)
		mem[0] = vectorof(float, 0, 0, 0)
	end
)

---------------------
-- Global lua vars --
---------------------
checkthis1 = 1
local checkthis2 = 2

local a = {}
a.b     = {}
a.b.c   = {}
a.b.c.d = 4

-------------------------------
-- ...let the testing begin! --
-------------------------------
-- Should fail b/c checkthis1 is not a scalar
local fail1 = liszt_kernel (cell in mesh.cells)
	checkthis1 = cell.f1
end

-- Should fail when we re-assign a new value to x, since it originally
-- refers to a topological element
local fail2 = liszt_kernel (cell in mesh.cells)
	var x = cell
    x = cell
end

-- Should fail because we do not allow assignments to fields
-- (only to indexed fields, scalars, and local vars)
local fail3 = liszt_kernel (cell in mesh.cells)
	mesh.cells.f1 = 5
end

-- Should fail because we do not allow the user to alias fields,
-- or any other entity that would confuse stencil generation, in the kernel
local fail4 = liszt_kernel (cell in mesh.cells)
	var z = mesh.cells.f1
end

local fail5 = liszt_kernel(cell in mesh.cells)
	undefined = 3
end

-- Can't assign a value of a different type to a variable that has already
-- been initialized
local fail6 = liszt_kernel (cell in mesh.cells)
	var floatvar = 2 + 3.3
	floatvar = true
end

-- local8 is not in scope in the while loop
local fail7 = liszt_kernel (cell in mesh.cells)
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

local fail8 = liszt_kernel (cell in mesh.cells)
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

local fail9 = liszt_kernel (cell in mesh.cells)
	var local1 = 3.4
	do
		var local1 = true
		local1 = 2.0 -- should fail, local1 is of type bool
	end
end

local fail10 = liszt_kernel (cell in mesh.cells)
	lassert(4 == true) -- binary op will fail here, type mismatch
end

local v = L.NewVector(L.float, {1, 1, 1})
local fail11 = liszt_kernel (cell in mesh.cells)
	lassert(v == 1) -- assert fail, comparison returns a vector of bools
end

local fail12 = liszt_kernel (cell in mesh.cells)
	a.b = 12
end

local fail13 = liszt_kernel (cell in mesh.cells)
	var v
	if false then
		v = true
	end
	v = 5
end

local tbl = {}
local fail14 = liszt_kernel (cell in mesh.cells)
	var x = 3 + tbl
end

local fail15 = liszt_kernel (cell in mesh.cells)
	var x = tbl
end

local fail16 = liszt_kernel (cell in mesh.cells)
	tbl.x = 4
end

local tbl = {x={}}
local fail17 = liszt_kernel (cell in mesh.cells)
	tbl.x.y = 4
end

tbl.x.z = 4
local fail18 = liszt_kernel (cell in mesh.cells)
	var x = tbl.x
end

-- need typechecker fail test
test.fail_function(fail1,  "assignments in a Liszt kernel are only valid")
test.fail_function(fail2,  "can only assign")
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
local good = liszt_kernel (cell in mesh.cells)
    cell.f1 = 3.0
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
	lassert(local9 == 14)
end

good()
