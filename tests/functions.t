--DISABLE-PARTITIONED
-- The MIT License (MIT)
-- 
-- Copyright (c) 2015 Stanford University.
-- All rights reserved.
-- 
-- Permission is hereby granted, free of charge, to any person obtaining a
-- copy of this software and associated documentation files (the "Software"),
-- to deal in the Software without restriction, including without limitation
-- the rights to use, copy, modify, merge, publish, distribute, sublicense,
-- and/or sell copies of the Software, and to permit persons to whom the
-- Software is furnished to do so, subject to the following conditions:
-- 
-- The above copyright notice and this permission notice shall be included
-- in all copies or substantial portions of the Software.
-- 
-- THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
-- IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
-- FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
-- AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
-- LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
-- FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
-- DEALINGS IN THE SOFTWARE.
import 'ebb'
local L = require 'ebblib'
require "tests/test"

local cells = L.NewRelation { size = 10, name = 'cells' }
cells:NewField('position', L.vec2d):Load(function(i)
  local x = i%5
  local y = math.floor(i/5)
  return {x,y}
end)


local assert, length, print, dot = L.assert, L.length, L.print, L.dot

-------------------
-- Test function --
-------------------
local square = ebb(x)
  return x*x
end

local t1 = ebb(c : cells)
  assert(square(7) == 49)
end
cells:foreach(t1)

-------------------------------
--Test stacked function calls--
-------------------------------
local sum = ebb(x, y, z)
    return x + y + z
end
local t2 = ebb(c : cells)
    assert(sum(square(1), square(2), square(3)) == 14)
end
cells:foreach(t2)

------------------------------------------
--Test functions that behave like fields--
------------------------------------------
cells:NewFieldReadFunction('scaledposition', ebb(c)
    return 2*c.position 
end)
local t3 = ebb(c : cells)
    assert(c.scaledposition == 2 * c.position)
end
cells:foreach(t3)

-- save backup positions
cells:NewField('orig_pos', L.vec2d):Load(cells.position)

cells:NewFieldWriteFunction('scaledposition', ebb(c, val)
    c.position = val/2.0
end)
local writefunc = ebb(c : cells)
    c.scaledposition = {3.2, 5.6}
    assert(c.position == {1.6,2.8})
end
cells:foreach(writefunc)

cells:NewFieldReduceFunction('scaledposition', '+', ebb(c, val)
    c.position += val/2.0
end)
local reducefunc = ebb(c : cells)
    c.scaledposition += {0.8,0.4}
    assert(c.position == {2,3})
end
cells:foreach(reducefunc)

-- restore 
cells:Copy { from='orig_pos', to='position' }

--------------------------------------
--Combine normal and field functions--
--------------------------------------
local norm = ebb(v)
    return dot(v, v)
end
local t4 = ebb(c : cells)
    var lensq = norm(c.scaledposition)
    var expected = 4.0 * length(c.position) * length(c.position)
    assert(square(lensq - expected) < 0.00005)
end
cells:foreach(t4)

------------------------------------------
--Test Functions with more than one line--
------------------------------------------
local sub1_but_non_neg = ebb(num)
  var result = num - 1
  if result < 0 then result = 0 end
  return result
end
local t5 = ebb (c : cells)
    assert(sub1_but_non_neg(2) == 1)
    assert(sub1_but_non_neg(0) == 0)
end
cells:foreach(t5)

----------------------------
--Test Dynamic Scoping Bug--
----------------------------
-- (symbols fixed this bug w/o alpha-renaming)
local dyn_prod = ebb(d, x) -- the two parameters is important
  return d*x
end
local dyn_outer = ebb(d)
  return dyn_prod(1, d+d)
end
local dyn_kern = ebb (c : cells)
    var r = dyn_outer(2)
    assert(r == 4)
end
cells:foreach(dyn_kern)


------------------
--Empty Function--
------------------
local empty_f = ebb() end
local t6 = ebb (c : cells)
  empty_f()
end
cells:foreach(t6)
-- Two failure cases to exercise case of no return value
test.fail_function(function()
  local ebb t(c : cells)
    var x = empty_f()
  end
  cells:foreach(t)
end, 'can only assign numbers, bools, or keys')
test.fail_function(function()
  local ebb t(c : cells)
    var x = 2 + empty_f()
  end
  cells:foreach(t)
end, "invalid types for operator '%+'")

--------------
--Just Print--
--------------
local print_42 = ebb() L.print(42) end
local do_print = ebb ( c : cells )
  print_42()
end
cells:foreach(do_print)


--------------
--Return val--
--------------
local get_3 = ebb() return 3 end
local check_3 = ebb ( c : cells )
  L.assert(get_3() == 3)
end
cells:foreach(check_3)


------------------------------
--check that recursion fails--
------------------------------
test.fail_function(function()
  local recurse = ebb() return recurse() end
end, "variable 'recurse' is not defined")


-----------------------------------------------------
--check that return values are prohibited if mapped--
-----------------------------------------------------
test.fail_function(function()
  local ebb t(c : cells)
    return 3
  end
  cells:foreach(t)
end, 'Functions executed over relations should not return values')


-----------------------------------------------
--Test improper placement of return statement--
-----------------------------------------------
--test.fail_function(function()
--  local bad_ret = ebb(num)
--    if num-1 < 0 then return 0 else return num-1 end
--  end
--end, 'asdfasf')
-- can't test parse error, since it isn't trapped

