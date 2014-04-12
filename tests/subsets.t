import "compiler.liszt"
require "tests/test"


local cells = L.NewRelation(5*5, 'cells')

cells:NewField('value', L.double):Load(function(i)
  if i/5 == 0 or i/5 == 4 or i%5 == 0 or i%5 == 4 then
    return 0
  else
    return 1
  end
end)

cells:NewSubsetFromFunction('boundary', function(i)
  return i/5 == 0 or i/5 == 4 or i%5 == 0 or i%5 == 4
end)
cells:NewSubsetFromFunction('interior', function(i)
  return not (i/5 == 0 or i/5 == 4 or i%5 == 0 or i%5 == 4)
end)


local test_boundary = liszt kernel ( c : cells )
  L.assert(c.value == 0)
end

local test_interior = liszt kernel ( c : cells )
  L.assert(c.value == 1)
end

test_boundary(cells.boundary)
test_interior(cells.interior)




