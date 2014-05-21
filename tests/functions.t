import "compiler.liszt"
require "tests/test"

local cells = L.NewRelation(10, 'cells')
cells:NewField('position', L.vec2d):Load(function(i)
  local x = i%5
  local y = math.floor(i/5)
  return {x,y}
end)


local assert, length, print, dot = L.assert, L.length, L.print, L.dot

-------------------
-- Test function --
-------------------
local square = liszt function(x)
  return x*x
end

local t1 = liszt kernel(c : cells)
  assert(square(7) == 49)
end
t1(cells)

-------------------------------
--Test stacked function calls--
-------------------------------
local sum = liszt function(x, y, z)
    return x + y + z
end
local t2 = liszt kernel(c : cells)
    assert(sum(square(1), square(2), square(3)) == 14)
end
t2(cells)

-------------------------------------------
--Test function that behaves like a field--
-------------------------------------------
cells:NewFieldFunction('scaledposition', liszt function(c)
    return 2*c.position 
end)
local t3 = liszt kernel(c : cells)
    assert(c.scaledposition == 2 * c.position)
end
t3(cells)

--------------------------------------
--Combine normal and field functions--
--------------------------------------
local norm = liszt function(v)
    return dot(v, v)
end
local t4 = liszt kernel(c : cells)
    var lensq = norm(c.scaledposition)
    var expected = 4.0 * length(c.position) * length(c.position)
    assert(square(lensq - expected) < 0.00005)
end
t4(cells)

------------------------------------------
--Test Functions with more than one line--
------------------------------------------
local sub1_but_non_neg = liszt function(num)
  var result = num - 1
  if result < 0 then result = 0 end
  return result
end
local t5 = liszt kernel (c : cells)
    assert(sub1_but_non_neg(2) == 1)
    assert(sub1_but_non_neg(0) == 0)
end
t5(cells)

----------------------------
--Test Dynamic Scoping Bug--
----------------------------
-- (symbols fixed this bug w/o alpha-renaming)
local dyn_prod = liszt function(d, x) -- the two parameters is important
  return d*x
end
local dyn_outer = liszt function(d)
  return dyn_prod(1, d+d)
end
local dyn_kern = liszt kernel (c : cells)
    var r = dyn_outer(2)
    assert(r == 4)
end
dyn_kern(cells)


------------------
--Empty Function--
------------------
local empty_f = liszt function() end
local t6 = liszt kernel (c : cells)
  empty_f()
end
t6(cells)
-- Two failure cases to exercise case of no return value
test.fail_function(function()
  liszt kernel (c : cells)
    var x = empty_f()
  end
end, 'can only assign numbers, bools, or rows')
test.fail_function(function()
  liszt kernel (c : cells)
    var x = 2 + empty_f()
  end
end, "invalid types for operator '%+'")

--------------
--Just Print--
--------------
local print_42 = liszt function() L.print(42) end
local do_print = liszt kernel ( c : cells )
  print_42()
end
do_print(cells)


--------------
--Return val--
--------------
local get_3 = liszt function() return 3 end
local check_3 = liszt kernel ( c : cells )
  L.assert(get_3() == 3)
end
check_3(cells)


------------------------------
--check that recursion fails--
------------------------------
test.fail_function(function()
  local recurse = liszt function() return recurse() end
end, "variable 'recurse' is not defined")



-----------------------------------------------
--Test improper placement of return statement--
-----------------------------------------------
--test.fail_function(function()
--  local bad_ret = liszt function(num)
--    if num-1 < 0 then return 0 else return num-1 end
--  end
--end, 'asdfasf')
-- can't test parse error, since it isn't trapped

