import "compiler.liszt"
require "tests/test"

local lassert, lprint = L.assert, L.print


---------------------------
-- Field and scalar objs --
---------------------------
local LMesh = terralib.require "compiler.lmesh"
local mesh = LMesh.Load("examples/mesh.lmesh")
mesh.cells:NewField('f1', L.float)
mesh.cells:NewField('f2', L.vector(L.float, 3))
s1 = L.NewScalar(L.int, 0)


------------------------
-- Initialize fields: --
------------------------
mesh.cells.f1:LoadConstant(0)
mesh.cells.f2:LoadConstant(L.NewVector(L.float, {0,0,0}))


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
test.fail_function(function()
 	liszt_kernel (cell : mesh.cells)
		checkthis1 = cell.f1
	end
end, "assignments in a Liszt kernel are only")

-- Should fail when we re-assign a new value to x, since it originally
-- refers to a topological element
test.fail_function(function()
  liszt_kernel (cell : mesh.cells)
		var x = cell
  	  x = cell
	end
end, "cannot re%-assign")

-- Should fail because we do not allow assignments to fields
-- (only to indexed fields, scalars, and local vars)
test.fail_function(function()
	fail3 = liszt_kernel (cell : mesh.cells)
		mesh.cells.f1 = 5
	end
end, "assignments in a Liszt kernel are only")

-- Should fail because we do not allow the user to alias fields,
-- or any other entity that would confuse stencil generation, in the kernel
test.fail_function(function()
  liszt_kernel (cell : mesh.cells)
		var z = mesh.cells.f1
	end
end, "can only assign")

test.fail_function(function()
 	liszt_kernel(cell : mesh.cells)
		undefined = 3
	end
end, "variable 'undefined' is not defined")

-- Can't assign a value of a different type to a variable that has already
-- been initialized
test.fail_function(function()
	liszt_kernel (cell : mesh.cells)
		var floatvar = 2 + 3.3
		floatvar = true
	end
end, "Could not coerce expression of type 'bool' into type 'double'")

-- local8 is not in scope in the while loop
test.fail_function(function()
	liszt_kernel (cell : mesh.cells)
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
end, "variable 'local8' is not defined")

test.fail_function(function()
	liszt_kernel (cell : mesh.cells)
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
end, "variable 'local8' is not defined")

test.fail_function(function()
	liszt_kernel (cell : mesh.cells)
		var local1 = 3.4
		do
			var local1 = true
			local1 = 2.0 -- should fail, local1 is of type bool
		end
	end
end, "Could not coerce expression of type 'int' into type 'bool'")

test.fail_function(function()
	liszt_kernel (cell : mesh.cells)
		lassert(4 == true) -- binary op will fail here, type mismatch
	end
end, "invalid types for operator")

local v = L.NewVector(L.float, {1, 1, 1})
test.fail_function(function()
	liszt_kernel (cell : mesh.cells)
		lassert(v) -- assert fail, comparison returns a vector of bools
	end
end, "expected a boolean")

test.fail_function(function()
	liszt_kernel (cell : mesh.cells)
		a.b = 12
	end
end, "assignments in a Liszt kernel are only")

test.fail_function(function()
	liszt_kernel (cell : mesh.cells)
		var v : L.bool
		if false then
			v = true
		end
		v = 5
	end
end, "Could not coerce expression of type 'int' into type 'bool'")

local tbl = {}
test.fail_function(function()
	liszt_kernel (cell : mesh.cells)
		var x = 3 + tbl
	end
end, "invalid types")

test.fail_function(function()
	liszt_kernel (cell : mesh.cells)
		var x = tbl
	end
end, "can only assign")

test.fail_function(function()
	liszt_kernel (cell : mesh.cells)
		tbl.x = 4
	end
end, "lua table does not have member 'x'")

local tbl = {x={}}
test.fail_function(function()
	liszt_kernel (cell : mesh.cells)
		tbl.x.y = 4
	end
end, "lua table does not have member 'y'")

tbl.x.z = 4
test.fail_function(function()
	liszt_kernel (cell : mesh.cells)
		var x = tbl.x
	end
end, "can only assign")

test.fail_function(function()
	liszt_kernel (cell : mesh.cells)
		for i = 1, 4, 1 do
			var x = 3
		end
		var g = i
	end
end, "variable 'i' is not defined")



-- Nothing should fail in this kernel:
local good = liszt_kernel (cell : mesh.cells)
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

good(mesh.cells)
