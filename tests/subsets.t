import "compiler.liszt"
require "tests/test"

-- Need a high N to test the index-based representation as well as boolmask
local N = 40
local cells = L.NewRelation(N*N, 'cells')

local function yidx(i) return math.floor(i/N) end

cells:NewField('value', L.double):Load(function(i)
  if yidx(i) == 0 or yidx(i) == N-1 or i%N == 0 or i%N == N-1 then
    return 0
  else
    return 1
  end
end)

cells:NewSubsetFromFunction('boundary', function(i)
  return yidx(i) == 0 or yidx(i) == N-1 or i%N == 0 or i%N == N-1
end)
cells:NewSubsetFromFunction('interior', function(i)
  return not (yidx(i) == 0 or yidx(i) == N-1 or i%N == 0 or i%N == N-1)
end)


local test_boundary = liszt kernel ( c : cells )
  L.assert(c.value == 0)
end

local test_interior = liszt kernel ( c : cells )
  L.assert(c.value == 1)
end

test_boundary(cells.boundary)
test_interior(cells.interior)




