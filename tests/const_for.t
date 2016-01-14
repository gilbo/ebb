--DISABLE-TEST
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

local Grid = require 'ebb.domains.grid'

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
local diffuse_names = ebb ( c : grid.cells )
  if not c.in_boundary then
    c.new_t = 0
    for dir in dirs do
      c.new_t += c.[dir].t - c.t
    end
  end
end

-- 8 neighbors
local diffuse_nums = ebb ( c : grid.cells )
  if not c.in_boundary then
    c.new_t = 0
    for i=-1,2 do
      for j=-1,2 do
        c.new_t += c(i,j).t - c.t
      end
    end
  end
end

grid.cells:foreach(diffuse_names)
grid.cells:foreach(diffuse_nums)

------------------------------------------------------------------------------

local cutoff = L.NewGlobal(L.int, 2)

test.fail_function(function()
  local ebb nonconst(c : grid.cells)
    if not c.in_boundary then
      c.new_t = 0
      for i=-1,cutoff do
        for j=-1,cutoff do
          c.new_t += c(i,j).t - c.t
        end
      end
    end
  end
  grid.cells:foreach(nonconst)
end, "cannot index using non-constant mapping")

















