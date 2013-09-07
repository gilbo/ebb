package.path = package.path .. ";./tests/?.lua;?.lua"
require "test"
import "compiler/liszt"


------------------
-- Should pass: --
------------------
local a  = Vector.new(float, {1,     2, 3.29})
local z  = Vector.new(float, {4, 5.392,    6})

local a4 = Vector.new(float, {3.4, 4.3, 5, 6.153})
local ai = Vector.new(int, {2, 3, 4})

local b  = 3 * a
local b2 = a * 3
local c  = a + b
local d  = a + ai


--------------------
-- Expected fails --
--------------------
function use_uninitialized()
	local uninit = Vector.new(float, 3)
	local x = uninit * 2
end

test.fail_function(use_uninitialized, "uninitialized")
