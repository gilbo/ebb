--[[ Note: this test file is not at all comprehensive for making sure that field reads/writes
	 translate correctly to terra code.  Right now, all it does it make sure that the codegen
	 produces something that can compile.
]]

import "compiler/liszt"
require 'tests/test'

mesh = LoadMesh("examples/mesh.lmesh")


----------------
-- check args --
----------------
function fail_topo1()
	local f = mesh:field(L.vector(L.float, 3), 'abc', 4)
end
function fail_topo2()
	local f = mesh:field(4, L.int, 3)
end

test.fail_function(fail_topo1, "topological")
test.fail_function(fail_topo2, "topological")

function fail_type1()
	local f = mesh:field(L.vertex, 'table', {})
end
function fail_type2()
	local f = mesh:field(L.face, 3)
end
test.fail_function(fail_type1, "data type")
test.fail_function(fail_type2, "data type")

function fail_init1()
	local f = mesh:field(L.face, L.vector(L.float, 3), true)
end
function fail_init2()
	local f = mesh:field(L.face, L.float, {1, 3, 4})
end
test.fail_function(fail_init1, "Initializer is not of type")
test.fail_function(fail_init2, "Initializer is not of type")


------------------
-- Test Codegen --
------------------
pos    = mesh:fieldWithLabel(L.vertex, L.vector(L.float, 3), "position")
field  = mesh:field(L.face,   L.float, 1.0)
field2 = mesh:field(L.face,   L.float, 2.5)
field3 = mesh:field(L.face,   L.float, 6.0)
field4 = mesh:field(L.cell,   L.bool,  false)
field5 = mesh:field(L.vertex, L.vector(L.float, 4), {0.0, 0.0, 0.0, 0.0})

local a = global(int, 6)
local b = Vector.new(L.float, {1, 3, 4, 5})

local reduce1 = liszt_kernel (f)
	field(f) = field(f) - 3 + 1/6 * a
end

local reduce2 = liszt_kernel (f)
	field2(f) = field2(f) * 3 * 7 / 3
end

local reduce3 = liszt_kernel (f)
	field3(f) = field3(f) 
end

local read1 = liszt_kernel (f)
	var tmp = field3(f) + 5
end

local write1 = liszt_kernel(f)
	field3(f) = 0.0
end
local write2 = liszt_kernel (f)
	field5(f) = b
end

local write3 = liszt_kernel (f)
	field4(f) = true
end

local check3 = liszt_kernel (f)
	assert(field4(f))
end

local write4 = liszt_kernel (f)
	field4(f) = false
end

local check4 = liszt_kernel(f)
	assert(not field4(f))
end

mesh.faces:map(reduce1)
mesh.faces:map(reduce2)
mesh.faces:map(reduce3)
mesh.faces:map(read1)
mesh.faces:map(write1)
mesh.faces:map(write2)

mesh.faces:map(write3)
mesh.faces:map(check3)

mesh.faces:map(write4)
mesh.faces:map(check4)

local  f = mesh:scalar(L.float, 0.0)
local bv = mesh:scalar(L.bool, true)
local f4 = mesh:scalar(L.vector(L.float, 4), {0, 0, 0, 0})

local function check_write ()
	-- should initialize each field element to {1, 3, 4, 5}
	mesh.faces:map(write2)

	f4:setTo({0, 0, 0, 0})
	mesh.faces:map(
		liszt_kernel (f)
			f4 = f4 + field5(f)
		end
	)
	local avg = f4:value() / mesh.faces:size()
	test.fuzzy_aeq(avg.data, {1, 3, 4, 5 })
end
--check_write()

