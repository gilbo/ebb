import "compiler.liszt"
require "tests/test"

local lassert, lprint = L.assert, L.print


---------------------------
-- Field and Global objs --
---------------------------
local R = L.NewRelation { name="R", size=6 }
R:NewField('f1', L.float)
R:NewField('f2', L.vector(L.float, 3))
s1 = L.Global(L.int, 0)


------------------------
-- Initialize fields: --
------------------------
R.f1:Load(0)
R.f2:Load({0,0,0})


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
-- Should fail b/c checkthis1 is not a global
test.fail_function(function()
 	local liszt t(cell : R)
		checkthis1 = cell.f1
	end
	R:map(t)
end, "Illegal assignment: left hand side cannot be assigned")

-- Should fail when we re-assign a new value to x, since it originally
-- refers to a topological element
test.fail_function(function()
	local liszt t(cell : R)
		var x = cell
  	  	x = cell
	end
	R:map(t)
end, "Illegal assignment: variables of key type cannot be re%-assigned")

-- Should fail because we do not allow assignments to fields
-- (only to indexed fields, globals, and local vars)
test.fail_function(function()
	local fail3 = liszt(cell : R)
		R.f1 = 5
	end
	R:map(fail3)
end, "Illegal assignment: left hand side cannot be assigned")

-- Should fail because we do not allow the user to alias fields,
-- or any other entity that would confuse stencil generation, in the function
test.fail_function(function()
	local liszt t(cell : R)
		var z = R.f1
	end
	R:map(t)
end, "can only assign")

test.fail_function(function()
 	local liszt t(cell : R)
		undefined = 3
	end
	R:map(t)
end, "variable 'undefined' is not defined")

-- Can't assign a value of a different type to a variable that has already
-- been initialized
test.fail_function(function()
	local liszt t(cell : R)
		var floatvar = 2 + 3.3
		floatvar = true
	end
	R:map(t)
end, "Could not coerce expression of type 'bool' into type 'double'")

-- local8 is not in scope in the while loop
test.fail_function(function()
	local liszt t(cell : R)
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
	R:map(t)
end, "variable 'local8' is not defined")

test.fail_function(function()
	local liszt t(cell : R)
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
	R:map(t)
end, "variable 'local8' is not defined")

test.fail_function(function()
	local liszt t(cell : R)
		var local1 = 3.4
		do
			var local1 = true
			local1 = 2.0 -- should fail, local1 is of type bool
		end
	end
	R:map(t)
end, "Could not coerce expression of type 'double' into type 'bool'")

test.fail_function(function()
	local liszt t(cell : R)
		lassert(4 == true) -- binary op will fail here, type mismatch
	end
	R:map(t)
end, "incompatible types: int and bool")

local v = L.Constant(L.vec3f, {1, 1, 1})
test.fail_function(function()
	local liszt t(cell : R)
		lassert(v) -- assert fail, comparison returns a vector of bools
	end
	R:map(t)
end, "expected a boolean")

test.fail_function(function()
	local liszt t(cell : R)
		a.b = 12
	end
	R:map(t)
end, "Illegal assignment: left hand side cannot be assigned")

test.fail_function(function()
	local liszt t(cell : R)
		var v : L.bool
		if false then
			v = true
		end
		v = 5
	end
	R:map(t)
end, "Could not coerce expression of type 'int' into type 'bool'")

local tbl = {}
test.fail_function(function()
	local liszt t(cell : R)
		var x = 3 + tbl
	end
	R:map(t)
end, "invalid types")

test.fail_function(function()
	local liszt t(cell : R)
		var x = tbl
	end
	R:map(t)
end, "can only assign")

test.fail_function(function()
	local liszt t(cell : R)
		tbl.x = 4
	end
	R:map(t)
end, "lua table does not have member 'x'")

local tbl = {x={}}
test.fail_function(function()
	local liszt t(cell : R)
		tbl.x.y = 4
	end
	R:map(t)
end, "lua table does not have member 'y'")

tbl.x.z = 4
test.fail_function(function()
	local liszt t(cell : R)
		var x = tbl.x
	end
	R:map(t)
end, "can only assign")

test.fail_function(function()
	local liszt t(cell : R)
		for i = 1, 4, 1 do
			var x = 3
		end
		var g = i
	end
	R:map(t)
end, "variable 'i' is not defined")



-- Nothing should fail in this function:
local good = liszt (cell : R)
    cell.f1 = 3
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
	local1 = 3
	var local1 = false

	var local9 = 0
	for i = 1, 4, 1 do
		local9 += i * i
	end
	lassert(local9 == 14)
end
R:map(good)
