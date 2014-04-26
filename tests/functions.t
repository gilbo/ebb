import "compiler.liszt"
require "tests/test"

local N = 5
local cells = L.NewRelation(N*N, 'cells')
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

(liszt kernel(c : cells)
  assert(square(7) == 49)
end)(cells)

-------------------------------
--Test stacked function calls--
-------------------------------
local sum = liszt function(x, y, z)
    return x + y + z
end)
(liszt kernel(c : cells)
    assert(sum(square(1), square(2), square(3)) == 14)
end)(cells)

----------------------------------------
--Test function that behaves like a field--
----------------------------------------
cells:NewFieldFunction('scaledposition', liszt function(c)
    return 2*c.position 
end))
(liszt kernel(c : cells)
    assert(c.scaledposition == 2 * c.position)
end)(cells)

--------------------------------------
--Combine normal and field functions--
--------------------------------------
local norm = liszt function(v)
    return dot(v, v)
end)
(liszt kernel(c : cells)
    var lensq = norm(c.scaledposition)
    var expected = 4.0 * length(c.position) * length(c.position)
    assert(square(lensq - expected) < 0.00005)
end)(cells)

------------------------------------------
--Test Functions with more than one line--
------------------------------------------
local sub1_but_non_neg = liszt function(num)
  var result = num - 1
  if result < 0 then result = 0 end
  return result
end)
(liszt kernel (c : cells)
    assert(sub1_but_non_neg(2) == 1)
    assert(sub1_but_non_neg(0) == 0)
end)(cells)


-----------------------------------------------
--Test improper placement of return statement--
-----------------------------------------------
--local bad_ret = liszt function(num)
--  if num-1 < 0 then return 0 else return num-1 end
--end

