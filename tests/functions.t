import "compiler.liszt"
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
local square = liszt(x)
  return x*x
end

local t1 = liszt(c : cells)
  assert(square(7) == 49)
end
cells:foreach(t1)

-------------------------------
--Test stacked function calls--
-------------------------------
local sum = liszt(x, y, z)
    return x + y + z
end
local t2 = liszt(c : cells)
    assert(sum(square(1), square(2), square(3)) == 14)
end
cells:foreach(t2)

-------------------------------------------
--Test function that behaves like a field--
-------------------------------------------
cells:NewFieldFunction('scaledposition', liszt(c)
    return 2*c.position 
end)
local t3 = liszt(c : cells)
    assert(c.scaledposition == 2 * c.position)
end
cells:foreach(t3)

--------------------------------------
--Combine normal and field functions--
--------------------------------------
local norm = liszt(v)
    return dot(v, v)
end
local t4 = liszt(c : cells)
    var lensq = norm(c.scaledposition)
    var expected = 4.0 * length(c.position) * length(c.position)
    assert(square(lensq - expected) < 0.00005)
end
cells:foreach(t4)

------------------------------------------
--Test Functions with more than one line--
------------------------------------------
local sub1_but_non_neg = liszt(num)
  var result = num - 1
  if result < 0 then result = 0 end
  return result
end
local t5 = liszt (c : cells)
    assert(sub1_but_non_neg(2) == 1)
    assert(sub1_but_non_neg(0) == 0)
end
cells:foreach(t5)

----------------------------
--Test Dynamic Scoping Bug--
----------------------------
-- (symbols fixed this bug w/o alpha-renaming)
local dyn_prod = liszt(d, x) -- the two parameters is important
  return d*x
end
local dyn_outer = liszt(d)
  return dyn_prod(1, d+d)
end
local dyn_kern = liszt (c : cells)
    var r = dyn_outer(2)
    assert(r == 4)
end
cells:foreach(dyn_kern)


------------------
--Empty Function--
------------------
local empty_f = liszt() end
local t6 = liszt (c : cells)
  empty_f()
end
cells:foreach(t6)
-- Two failure cases to exercise case of no return value
test.fail_function(function()
  local liszt t(c : cells)
    var x = empty_f()
  end
  cells:foreach(t)
end, 'can only assign numbers, bools, or keys')
test.fail_function(function()
  local liszt t(c : cells)
    var x = 2 + empty_f()
  end
  cells:foreach(t)
end, "invalid types for operator '%+'")

--------------
--Just Print--
--------------
local print_42 = liszt() L.print(42) end
local do_print = liszt ( c : cells )
  print_42()
end
cells:foreach(do_print)


--------------
--Return val--
--------------
local get_3 = liszt() return 3 end
local check_3 = liszt ( c : cells )
  L.assert(get_3() == 3)
end
cells:foreach(check_3)


------------------------------
--check that recursion fails--
------------------------------
test.fail_function(function()
  local recurse = liszt() return recurse() end
end, "variable 'recurse' is not defined")


-----------------------------------------------------
--check that return values are prohibited if mapped--
-----------------------------------------------------
test.fail_function(function()
  local liszt t(c : cells)
    return 3
  end
  cells:foreach(t)
end, 'Functions executed over relations should not return values')


-----------------------------------------------
--Test improper placement of return statement--
-----------------------------------------------
--test.fail_function(function()
--  local bad_ret = liszt(num)
--    if num-1 < 0 then return 0 else return num-1 end
--  end
--end, 'asdfasf')
-- can't test parse error, since it isn't trapped

