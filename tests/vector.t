package.path = package.path .. ";./tests/?.lua;?.lua"
local test = require "test"
import "compiler.liszt"
local types = terralib.require "compiler.types"

------------------
-- Should pass: --
------------------
local a  = L.NewVector(L.float, {1,     2, 3.29})
local z  = L.NewVector(L.float, {4, 5.392,    6})

local a4 = L.NewVector(L.float, {3.4, 4.3, 5, 6.153})
local ai = L.NewVector(L.int,   {2, 3, 4})
local ab = L.NewVector(L.bool,  {true, false, true})

local b  = 3 * a
local b2 = a * 3
local c  = a + b
local d  = a + ai
local e  = ai / 4.5
local f  = c - L.NewVector(L.float, {8, 8, 8})
local g  = a % 3
local h  = a4 % -2


test.eq(b.N,     a.N)
test.eq(b2.N,    a.N)
test.eq(c.N,     a.N)
test.eq(d.N,     a.N)
test.eq(d.type,  a.type)
test.eq(b.type,  b2.type)
test.eq(e.type:baseType(), L.float)

test.aeq(a.data, {1, 2, 3.29})
test.aeq(b.data, {3*1, 3*2, 3*3.29})
test.aeq(b.data, b2.data)


--------------------
-- Expected fails --
--------------------
function div_fail ()
	return 4.5 / ai
end

function add_fail ()
	return 2 + a
end

function add_fail2 ()
	return a + 2
end

function sub_fail ()
	return 2 - a
end

function sub_fail2 ()
	return a - 2
end

function mult_fail ()
	return a * b
end

function power_fail ()
	return a ^ 3
end

function type_fail ()
	return ai / true
end

function type_fail2 ()
	return a + a4
end

function type_fail3 ()
	return a4 + a
end

function type_fail4 ()
	return a - a4
end

function type_fail5 ()
	return a4 - a
end

test.fail_function(div_fail,   "divide")
test.fail_function(add_fail,   "add")
test.fail_function(add_fail2,  "add")
test.fail_function(sub_fail,   "subtract")
test.fail_function(sub_fail2,  "subtract")
test.fail_function(mult_fail,  "multiply")
test.fail_function(power_fail, "arithmetic") -- Lua error here, for now

test.fail_function(type_fail,  "numeric")
test.fail_function(type_fail2, "lengths")
test.fail_function(type_fail3, "lengths")
test.fail_function(type_fail4, "lengths")
test.fail_function(type_fail5, "lengths")


--------------------------
-- Kernel vector tests: --
--------------------------
local LMesh = terralib.require "compiler.lmesh"
local mesh = LMesh.Load("examples/mesh.lmesh")

------------------
-- Should pass: --
------------------
local k = liszt_kernel (v : mesh.vertices)
	var x       = {5, 5, 5}
	v.position += x + {0, 1, 1}
end
k(mesh.vertices)

local s = L.NewScalar(L.vector(L.float, 3), {0.0, 0.0, 0.0})
local sum_position = liszt_kernel(v : mesh.vertices)
	s += v.position
end
sum_position(mesh.vertices)

local f = s:value() / mesh.vertices._size
test.fuzzy_aeq(f.data, {5, 6, 6})
