--DISABLE-TEST
import "compiler.liszt"
require "tests/test"

local grid = L.require 'domains.grid'

local N = 4
local grid = Grid.New2dUniformGrid(N,N, {0,0}, N, N)

------------------------------------------------------------------------------

grid.cells:NewField('t', L.double):Load(function(i)
  if i == 0 then return N*N else return 0 end)
grid.cells:NewField('new_t', L.double):Load(0)

-- 4 neighbors
local dirs = {'left', 'right', 'up', 'down'}
local diffuse_names = liszt kernel ( c : grid.cells )
  if not c.is_bnd then
    var sum = 0
    for dir in dirs do
      sum += c.[dir].t - c.t
    end
  end
end

-- 8 neighbors
local diffuse_nums = liszt kernel ( c : grid.cells )
  if not c.is_bnd
    var sum = 0
    for i=-1,2 do
      for j=-1,2 do
        sum += c(i,j).t - c.t
      end
    end
  end
end

diffuse_names(grid.cells)
diffuse_nums(grid.cells)

------------------------------------------------------------------------------

local cutoff = L.NewGlobal(L.int, 2)

test.fail_function(function()
  liszt kernel (c : grid.cells)
    if not c.is_bnd
      var sum = 0
      for i=-1,cutoff do
        for j=-1,cutoff do
          sum += c(i,j).t - c.t
        end
      end
    end
  end
end, "cannot index using non-constant mapping")

















