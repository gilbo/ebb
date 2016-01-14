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

-- In this example, we'll see how to get lower level access to the data that
-- Ebb is storing and computing on.  This data access is provided through
-- metadata objects that we call data layout descriptors (DLDs).  These
-- descriptors communicate information about how data is stored in memory
-- so that Ebb-external code can directly manipulate the data without
-- incurring memory copy overheads.

import 'ebb'
local L = require 'ebblib'

local GridLib   = require 'ebb.domains.grid'

local DLD       = require 'ebb.lib.dld'
local vdb       = require 'ebb.lib.vdb'
-- In addition to libraries we've already seen, we also require the DLD
-- library.

local N = 40

local grid = GridLib.NewGrid2d {
  size   = {N, N},
  origin = {0, 0},
  width  = {N, N},
  periodic_boundary = {true,true},
}

local timestep    = L.Constant(L.double, 0.45)
local conduction  = L.Constant(L.double, 1.0)

grid.cells:NewField('t', L.double):Load(0)
grid.cells:NewField('new_t', L.double):Load(0)

local function init_temperature(x_idx, y_idx)
  if x_idx == 4 and y_idx == 6 then return 400
                               else return 0 end
end
grid.cells.t:Load(init_temperature)

local ebb visualize ( c : grid.cells )
  vdb.color({ 0.5 * c.t + 0.5, 0.5-c.t, 0.5-c.t })
  var p2 = c.center
  vdb.point({ p2[0], p2[1], 0 })
end

local ebb update_temperature ( c : grid.cells )
  var avg   = (1.0/4.0) * ( c(1,0).t + c(-1,0).t + c(0,1).t + c(0,-1).t )
  var diff  = avg - c.t
  c.new_t   = c.t + timestep * conduction * diff
end
-- Similarly to tutorial 09, we simplify the grid-based heat-diffusion code
-- by assuming periodic boundaries.

-- Below we define a function to shuffle the temperature data in a way that
-- would not be allowed by an Ebb `foreach()` call.  This function is defined
-- using Terra, which is a C-alternative language embedded in Lua.  When
-- using Ebb, Terra is always available to you. We could equally well
-- use a function defined in C, and do exactly that in a later tutorial.

-- Using DLDs, we can incorporate pre-existing linear solvers, an FFT,
-- or other computations that are (i) legacy code, (ii) specially optimized,
-- or (iii) use computation patterns that Ebb does not support or allow.

terra tile_shuffle( dld : &DLD.C_DLD )
  var t_ptr   = [&double](dld.address)
  var xstride = dld.dim_stride[0]
  var ystride = dld.dim_stride[1]

  for y=0,20 do
    for x=0,20 do
      var t1_idx  =      x * xstride   +      y * ystride
      var t2_idx  = (x+20) * xstride   + (y+20) * ystride

      var temp      = t_ptr[t1_idx]
      t_ptr[t1_idx] = t_ptr[t2_idx]
      t_ptr[t2_idx] = temp
    end
  end
end
-- This Terra function reads out the address of the temperature data and
-- the appropriate strides to let it iterate over that data.  Because we're
-- writing this function for the very specific case of a 40x40 2d grid of
-- double values, we can simplify the code tremendously.  The function
-- swaps the top left quadrant with the bottom right.  This function
-- doesn't have any physical analogy, but it's a simple example of a
-- computation that we can't encode in Ebb due to phase-checking restrictions.

-- While using these external computations imposes less restrictions, we also
-- lose Ebb's ability to automatically parallelize the code.  The
-- `tile_shuffle` function depends on data being CPU-resident, and computes
-- the swap as a purely sequential computation.  If we want to swap in
-- parallel on a GPU, then we would have to write an entirely separate
-- function.


for i=1,360 do
  grid.cells:foreach(update_temperature)
  grid.cells:Swap('t', 'new_t')

  if i % 200 == 199 then
    local t_dld = grid.cells.t:GetDLD()

    assert( t_dld.version[1] == 1 and t_dld.version[2] == 0 )
    assert( t_dld.base_type      == DLD.DOUBLE )
    assert( t_dld.location       == DLD.CPU )
    assert( t_dld.type_stride    == 8 )
    assert( t_dld.type_dims[1] == 1 and t_dld.type_dims[2] == 1 )
    assert( t_dld.dim_size[1] == 40 and
            t_dld.dim_size[2] == 40 and
            t_dld.dim_size[3] == 1 )

    tile_shuffle( t_dld:toTerra() )
  end

  vdb.vbegin()
  vdb.frame()
    grid.cells:foreach(visualize)
  vdb.vend()

  if i % 10 == 0 then print( 'iteration #'..tostring(i) ) end
end
-- Finally, we modify our simulation loop to swap the tiles on the 200th
-- iteration.  This swap proceeds by requesting a Lua form of the DLD object
-- via `:GetDLD()`, asserting that a number of values are what we expect them
-- to be, and finally calling the tile_shuffle function with a Terra version
-- of the DLD object.

