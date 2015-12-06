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
------------------------------------------------------------------------------

import 'ebb'
local L = require 'ebblib'

local GridLib   = require 'ebb.domains.grid'
local vdb       = require 'ebb.lib.vdb'
local N = 40
local grid = GridLib.NewGrid2d {
  size   = {N, N},
  origin = {0, 0},
  width  = {N, N},
  periodic_boundary = {true,true},
}

-- define constants, globals and fields
local timestep    = L.Constant(L.double, 0.45)
local conduction  = L.Constant(L.double, 1.0)
local max_diff    = L.Global(L.double, 0.0)

grid.cells:NewField('t', L.double):Load(function(x_idx, y_idx)
  if x_idx == 4 and y_idx == 6 then return 1000 else return 0 end
end)
grid.cells:NewField('new_t', L.double):Load(0)

-- compute diffusion
local ebb update_temperature ( c : grid.cells )
  var avg   = (1.0/4.0) * ( c(1,0).t + c(-1,0).t + c(0,1).t + c(0,-1).t )
  var diff  = avg - c.t
  c.new_t   = c.t + timestep * conduction * diff
end

-- measure statistic
local ebb measure_max_diff ( c : grid.cells )
  var avg   = (1.0/4.0) * ( c(1,0).t + c(-1,0).t + c(0,1).t + c(0,-1).t )
  var diff  = avg - c.t
  max_diff  max= L.fabs(diff)
end

local ebb visualize ( c : grid.cells )
  vdb.color({ 0.5 * c.t + 0.5, 0.5-c.t, 0.5-c.t })
  var p2 = c.center
  vdb.point({ p2[0], p2[1], 0 })
end

-- simulation loop
for i=1,360 do
  grid.cells.interior:foreach(update_temperature)
  grid.cells:Swap('t', 'new_t')

  vdb.vbegin()
  vdb.frame()
    grid.cells:foreach(visualize)
  vdb.vend()

  if i % 10 == 0 then -- measure statistics every 10 steps
    max_diff:set(0)
    grid.cells.interior:foreach(measure_max_diff)
    print( 'iteration #'..tostring(i), 'max gradient: ', max_diff:get() )
  end
end
