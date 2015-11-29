-- In this tutorial, we'll repeat the tile-swapping variant of the
-- heat-diffusion simulation from tutorial 10, but instead of implementing
-- the tile swap computation in Terra, we'll use traditional C-code.

-- That C-code is in file `11___c_code.c` and `11___c_header.h`.  It needs
-- to be separately compiled into a shared object/library.  We've included
-- some sample compilation commands in a makefile in this directory.

import "ebb"

local GridLib   = require 'ebb.domains.grid'

local DLD       = require 'ebb.lib.dld'
local vdb       = require 'ebb.lib.vdb'

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
-- The bulk of this simulation is exactly identical to the previous tutorial
-- 10 on DLDs.


local PN    = require 'ebb.lib.pathname'
local here  = tostring(PN.scriptdir():abspath())

terralib.linklibrary(here..'/11___c_obj.so')
local C_Import = terralib.includecstring('#include "11___c_header.h"',
                                         {'-I'..here,
                                          '-I'..here..'/../../include'})
-- Here we use the Terra API calls `linklibrary()` and `includecstring()` to
-- first dynamically link the shared object into the process and then import
-- the header file's interface.  To do so, we need to make sure Terra knows
-- where to look for the shared object and for all the header files being
-- included.  Rather than hard-code fragile paths, we again use the pathname
-- library to dynamically determine the correct filesystem paths based on
-- the location of this script file.

-- Once these calls have run, the C_Import table will contain a Terra/C
-- function `tile_shuffle()`, that we can call to invoke the C code.  For
-- more information on how this whole process works, take a look at the Terra
-- documentation.


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
    assert( t_dld.dim_size[3] == 1 )

    C_Import.tile_shuffle( t_dld:toTerra() )
  end

  vdb.vbegin()
  vdb.frame()
    grid.cells:foreach(visualize)
  vdb.vend()

  if i % 10 == 0 then print( 'iteration #'..tostring(i) ) end
end
-- The simulation loop is then nearly identical to the simulation loop from
-- tutorial 10.  Since Terra is designed to be ABI compatible with C, passing
-- the Terra version of the DLD struct is the same thing as passing a
-- C-struct version of the DLD metadata.

