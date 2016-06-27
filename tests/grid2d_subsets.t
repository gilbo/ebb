-- The MIT License (MIT)
-- 
-- Copyright (c) 2016 Stanford University.
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


local N = 16
local grid  = L.NewRelation {
  dims = {N,N},
  name = 'gridcells'
}
grid:SetPartitions{2,2}

grid:NewField('s1v', L.double):Load(0.0)
grid:NewField('s2v', L.double):Load(0.0)

local ebb init_cells( c : grid )
  c.s1v = 1.0
  if L.xid(c) == 0 or L.xid(c) == N-1 or
     L.yid(c) == 0 or L.yid(c) == N-1 then c.s1v = 0.0 end

  var cx  = L.xid(c) < N/2
  var cy  = L.yid(c) < N/2
  if cx == cy then c.s2v = 1.0 else c.s2v = 0.0 end
end
grid:foreach(init_cells)


grid:NewSubset('interior', { {1,N-2}, {1,N-2} })
grid:NewSubset('checker', {
  rectangles = {
    { {0,N/2-1}, {0,N/2-1} },
    { {N/2,N-1}, {N/2,N-1} },
  },
})

local ebb check_1( c : grid )
  L.assert(c.s1v == 1.0)
  c.s1v = 0.0
end
local ebb check_2( c : grid )
  L.assert(c.s2v == 1.0)
  c.s2v = 0.0
end

local ebb check_all_0( c : grid )
  L.assert(c.s1v == 0.0)
  L.assert(c.s2v == 0.0)
end

grid.interior:foreach(check_1)
grid.checker:foreach(check_2)
grid:foreach(check_all_0)

