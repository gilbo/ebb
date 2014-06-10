--DISABLE-TEST
import "compiler.liszt"
require "tests/test"

local Grid = L.require 'domains.grid'

local N = 4
local grid = Grid.NewGrid2d {
  size    = {N,N},
  origin  = {0,0},
  width   = N,
  height  = N,
}

------------------------------------------------------------------------------

grid.cells:NewField('t', L.double):Load(function(i)
  if i == 0 then return N*N else return 0 end end)
grid.cells:NewField('new_t', L.double):Load(0)

-- 4 neighbors
local dirs = {'left', 'right', 'up', 'down'}
local diffuse_names = liszt kernel ( c : grid.cells )
  if not c.in_boundary then
    c.new_t = 0
    for dir in dirs do
      c.new_t += c.[dir].t - c.t
    end
  end
end

-- 8 neighbors
local diffuse_nums = liszt kernel ( c : grid.cells )
  if not c.in_boundary then
    c.new_t = 0
    for i=-1,2 do
      for j=-1,2 do
        c.new_t += c(i,j).t - c.t
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
    if not c.in_boundary then
      c.new_t = 0
      for i=-1,cutoff do
        for j=-1,cutoff do
          c.new_t += c(i,j).t - c.t
        end
      end
    end
  end
end, "cannot index using non-constant mapping")

















