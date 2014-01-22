--[[ Note: this test file is not at all comprehensive for making sure that field reads/writes
	 translate correctly to terra code.  Right now, all it does it make sure that the codegen
	 produces something that can compile.
]]

import "compiler.liszt"
require "tests.test"

local assert = L.assert
local LMesh = terralib.require "compiler.lmesh"
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

F.field1:LoadFromCallback(terra (mem : &float, i : uint) mem[0] = 1     end)
F.field2:LoadFromCallback(terra (mem : &float, i : uint) mem[0] = 2.5   end)
F.field3:LoadFromCallback(terra (mem : &float, i : uint) mem[0] = 6     end)
F.field4:LoadFromCallback(terra (mem : &bool,  i : uint) mem[0] = false end)
F.field5:LoadFromCallback(
	terra (mem: &float, i : uint)
		mem[0] = 0
		mem[1] = 0
		mem[2] = 0
		mem[3] = 0
	end
)


-----------------
-- Local vars: --
-----------------
local a = 6
local b = L.NewVector(L.float, {1, 3, 4, 5})


-------------------
-- Test kernels: --
-------------------
local reduce1 = liszt_kernel (f in F)
	f.field1 -= 3 - 1/6 * a
end

local reduce2 = liszt_kernel (f in F)
	f.field2 *= 3 * 7 / 3
end

local read1 = liszt_kernel (f in F)
	var tmp = f.field3 + 5
	assert(tmp == 11)
end

local write1 = liszt_kernel(f in F)
	f.field3 = 0.0
end

local write2 = liszt_kernel (f in F)
	f.field5 = b
end

local reduce3 = liszt_kernel (f in F)
	f.field5 += {1.0,1.0,1.0,1.0}
end

local check2 = liszt_kernel (f in F)
	assert(f.field5[0] == 2)
	assert(f.field5[1] == 4)
	assert(f.field5[2] == 5)
	assert(f.field5[3] == 6)
end

local write3 = liszt_kernel (f in F)
	f.field4 = true
end

local check3 = liszt_kernel (f in F)
	assert(f.field4)
end

local write4 = liszt_kernel (f in F)
	f.field4 = false
end

local check4 = liszt_kernel(f in F)
	assert(not f.field4)
end


-- execute!
reduce1()
reduce2()

read1()
write1()
write2()
reduce3()
check2()

write3()
check3()

write4()
check4()



--------------------------------------
-- Now the same thing with scalars: --
--------------------------------------
local  f = L.NewScalar(L.float, 0.0)
local bv = L.NewScalar(L.bool, true)
local f4 = L.NewScalar(L.vector(L.float, 4), {0, 0, 0, 0})

local function check_write ()
	-- should initialize each field element to {2, 4, 5, 6}
	write2()
	reduce3()

	f4:setTo({0, 0, 0, 0})
	local sum_positions = liszt_kernel (f in F)
		f4 += f.field5
	end
	sum_positions()

	local avg = f4:value() / F._size
	test.fuzzy_aeq(avg.data, {2, 4, 5, 6})
end
check_write()
