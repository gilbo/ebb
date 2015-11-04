--[[
  Note: this test file is not at all comprehensive for making sure
        that field reads/writes translate correctly to terra code.
        Right now, all it does it make sure that the codegen
        produces something that can compile.
]]

import "ebb"
require "tests.test"

local assert = L.assert
local ioOff = require 'ebb.domains.ioOff'
local mesh  = ioOff.LoadTrimesh('tests/octa.off')

local V      = mesh.vertices
local T      = mesh.triangles

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
T:NewField('field1', L.float)
T:NewField('field2', L.float)
T:NewField('field3', L.float)
T:NewField('field4', L.bool)
T:NewField('field5', L.vector(L.float, 4))

T.field1:Load(1)
T.field2:Load(2.5)
T.field3:Load(6)
T.field4:Load(false)
T.field5:Load({ 0, 0, 0, 0 })


-----------------
-- Local vars: --
-----------------
local a = 6
local b = L.Constant(L.vec4f, {1, 3, 4, 5})


---------------------
-- Test functions: --
---------------------
local reduce1 = ebb (t : T)
  t.field1 -= 3 - 1/6 * a
end

local reduce2 = ebb (t : T)
  t.field2 *= 3 * 7 / 3
end

local read1 = ebb (t : T)
  var tmp = t.field3 + 5
  assert(tmp == 11)
end

local write1 = ebb(t : T)
  t.field3 = 0.0f
end

local write2 = ebb (t : T)
  t.field5 = b
end

local reduce3 = ebb (t : T)
  t.field5 += {1.0f,1.0f,1.0f,1.0f}
end

local check2 = ebb (t : T)
  assert(t.field5[0] == 2)
  assert(t.field5[1] == 4)
  assert(t.field5[2] == 5)
  assert(t.field5[3] == 6)
end

local write3 = ebb (t : T)
  t.field4 = true
end

local check3 = ebb (t : T)
  assert(t.field4)
end

local write4 = ebb (t : T)
  t.field4 = false
end

local check4 = ebb(t : T)
  assert(not t.field4)
end


-- execute!
T:foreach(reduce1)
T:foreach(reduce2)

T:foreach(read1)
T:foreach(write1)
T:foreach(write2)
T:foreach(reduce3)
T:foreach(check2)
T:foreach(write2)

T:foreach(write3)
T:foreach(check3)

T:foreach(write4)
T:foreach(check4)



--------------------------------------
-- Now the same thing with globals: --
--------------------------------------
local  f = L.Global(L.float, 0.0)
local bv = L.Global(L.bool, true)
local f4 = L.Global(L.vector(L.float, 4), {0, 0, 0, 0})

local function check_write ()
  -- should initialize each field element to {2, 4, 5, 6}
  T:foreach(write2)
  T:foreach(reduce3)

  f4:set({0, 0, 0, 0})
  local sum_positions = ebb (t : T)
    f4 += t.field5
  end
  T:foreach(sum_positions)

  local f4t = f4:get()
  local fs  = T:Size()
  local avg = { f4t[1]/fs, f4t[2]/fs, f4t[3]/fs, f4t[4]/fs }
  test.fuzzy_aeq(avg, {2, 4, 5, 6})
end
check_write()
