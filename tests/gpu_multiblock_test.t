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

-- This test actually doesn't need to run on the GPU,
-- it's just designed to stress test GPU execution in particular
import 'ebb'
local L = require 'ebblib'

-- A 17x17 grid will, at current settings,
-- will force ebb to run the generated kernel
-- on multiple GPU blocks
local N=17

local Grid  = require 'ebb.domains.grid'
local grid = Grid.NewGrid2d{size           = {N, N},
                            origin         = {0, 0},
                            width          = {1, 1},
                            boundary_depth = {1, 1},
                            periodic_boundary = {true, true},
                            partitions     = {2, 2}, }

function main ()
	-- declare a global to store the computed centroid of the grid
	local com = L.Global(L.vector(L.float, 2), {0, 0})

	-- compute centroid
	local sum_pos = ebb(c : grid.cells)
		com += c.center
	end
	grid.cells:foreach(sum_pos)

	local center = com:get()
  center[1] = center[1] / grid.cells:Size()
  center[2] = center[2] / grid.cells:Size()

  -- test with a fudge factor
  assert( math.abs(center[1]-0.5) < 1e-5 and
          math.abs(center[2]-0.5) < 1e-5 )
end

main()