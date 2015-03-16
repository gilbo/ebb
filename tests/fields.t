--[[ Note: this test file is not at all comprehensive for making sure that field reads/writes
	 translate correctly to terra code.  Right now, all it does it make sure that the codegen
	 produces something that can compile.
]]

import "compiler.liszt"
require "tests.test"

local assert = L.assert
local LMesh = L.require "domains.lmesh"
local mesh = LMesh.Load("examples/mesh.lmesh")

local V      = mesh.vertices
local F      = mesh.faces

----------------
-- check args --
----------------
function fail_type1()
	local f = V:NewField('field', 'asdf')
end
function fail_type2()
	local f = V:NewField('field', bool)
end

test.fail_function(fail_type1, "type")
test.fail_function(fail_type2, "type")


-------------------------------
-- Create/initialize fields: --
-------------------------------
F:NewField('field1', L.float)
F:NewField('field2', L.float)
F:NewField('field3', L.float)
F:NewField('field4', L.bool)
F:NewField('field5', L.vector(L.float, 4))

F.field1:LoadConstant(1)
F.field2:LoadConstant(2.5)
F.field3:LoadConstant(6)
F.field4:LoadConstant(false)
F.field5:LoadConstant({ 0, 0, 0, 0 })


-----------------
-- Local vars: --
-----------------
local a = 6
local b = L.Constant(L.vec4f, {1, 3, 4, 5})


---------------------
-- Test functions: --
---------------------
local reduce1 = liszt (f : F)
	f.field1 -= 3 - 1/6 * a
end

local reduce2 = liszt (f : F)
	f.field2 *= 3 * 7 / 3
end

local read1 = liszt (f : F)
	var tmp = f.field3 + 5
	assert(tmp == 11)
end

local write1 = liszt(f : F)
	f.field3 = 0.0f
end

local write2 = liszt (f : F)
	f.field5 = b
end

local reduce3 = liszt (f : F)
	f.field5 += {1.0f,1.0f,1.0f,1.0f}
end

local check2 = liszt (f : F)
	assert(f.field5[0] == 2)
	assert(f.field5[1] == 4)
	assert(f.field5[2] == 5)
	assert(f.field5[3] == 6)
end

local write3 = liszt (f : F)
	f.field4 = true
end

local check3 = liszt (f : F)
	assert(f.field4)
end

local write4 = liszt (f : F)
	f.field4 = false
end

local check4 = liszt(f : F)
	assert(not f.field4)
end


-- execute!
F:map(reduce1)
F:map(reduce2)

F:map(read1)
F:map(write1)
F:map(write2)
F:map(reduce3)
F:map(check2)

F:map(write3)
F:map(check3)

F:map(write4)
F:map(check4)



--------------------------------------
-- Now the same thing with globals: --
--------------------------------------
local  f = L.Global(L.float, 0.0)
local bv = L.Global(L.bool, true)
local f4 = L.Global(L.vector(L.float, 4), {0, 0, 0, 0})

local function check_write ()
	-- should initialize each field element to {2, 4, 5, 6}
	F:map(write2)
	F:map(reduce3)

	f4:set({0, 0, 0, 0})
	local sum_positions = liszt (f : F)
		f4 += f.field5
	end
	F:map(sum_positions)

	local f4t = f4:get()
	local fs  = F:Size()
	local avg = { f4t[1]/fs, f4t[2]/fs, f4t[3]/fs, f4t[4]/fs }
	test.fuzzy_aeq(avg, {2, 4, 5, 6})
end
check_write()
