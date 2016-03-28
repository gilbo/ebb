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

-- Need a high N to test the index-based representation as well as boolmask
local N = 40
--[[
local cells = L.NewRelation { size = N*N, name = 'cells' }

local function yidx(i) return math.floor(i/N) end

cells:NewField('value', L.double):Load(function(i)
  if yidx(i) == 0 or yidx(i) == N-1 or i%N == 0 or i%N == N-1 then
    return 0
  else
    return 1
  end
end)

-- subsets of non-grid relations are currently disabled
--[[
cells:NewSubsetFromFunction('boundary', function(i)
  return yidx(i) == 0 or yidx(i) == N-1 or i%N == 0 or i%N == N-1
end)
cells:NewSubsetFromFunction('interior', function(i)
  return not (yidx(i) == 0 or yidx(i) == N-1 or i%N == 0 or i%N == N-1)
end)


local test_boundary = ebb ( c : cells )
  L.assert(c.value == 0)
end

local test_interior = ebb ( c : cells )
  L.assert(c.value == 1)
end

cells.boundary:foreach(test_boundary)
cells.interior:foreach(test_interior)
--]]


-- Now run the same test, but with a truly grid structured relation

local gridcells  = L.NewRelation { dims = {N,N}, name = 'gridcells' }
gridcells:SetPartitions {2,2}


gridcells:NewField('value', L.double):Load(function(xi,yi)
  if xi == 0 or xi == N-1 or yi == 0 or yi == N-1 then
    return 0
  else
    return 1
  end
end)


gridcells:NewSubset('boundary', {
  rectangles = { { {0,0},     {0,N-1}   },
                 { {N-1,N-1}, {0,N-1}   },
                 { {0,N-1},   {0,0}     },
                 { {0,N-1},   {N-1,N-1} } }
})
gridcells:NewSubset('interior', { {1,N-2}, {1,N-2} })

local ebb grid_test_boundary ( c : gridcells )
  L.assert(c.value == 0)
end
local ebb grid_test_interior ( c : gridcells )
  L.assert(c.value == 1)
end

gridcells.boundary:foreach(grid_test_boundary)
gridcells.interior:foreach(grid_test_interior)




