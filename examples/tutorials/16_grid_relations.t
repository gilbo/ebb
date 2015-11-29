-- In this tutorial, we show how to create grid-structured domains and
-- connect them.  Grid-structured data is especially important for
-- simulations.  Grids also offer special opportunities for optimization
-- by exploiting the regular addressing to eliminate memory accesses.
-- Consequently, Ebb provides special mechanisms for indicating that certain
-- relations are grid-structured, and for connecting those relations to
-- themselves and other gridded relations.

-- To illustrate these features, we're going to write two coupled simulations
-- on grids of different scales.  The lower-resolution grid will simulate
-- a heat diffusion, while the higher-resolution grid will simulate the wave
-- equation.  The particular simulation isn't physically derived, but
-- will show both how to write code for grids and how to construct multi-grid
-- structures.


import "ebb"

local vdb   = require('ebb.lib.vdb')
-- We'll start the program in the usual way


local N = 50

local hi_cells = L.NewRelation {
  name      = 'hi_cells',
  dims      = { N, N },
  periodic  = { true, true },
}
local lo_cells = L.NewRelation {
  name      = 'lo_cells',
  dims      = { N/2, N/2 },
  periodic  = { true, true },
}
-- Instead of declaring the size of a relation, we can specify `dims`, a Lua
-- list of 2 or 3 numbers, giving the number of grid entries we want in each
-- dimension.  If we want the grid relation to be considered periodic, then
-- we can additionally specify a `periodic` parameter.  Notice that raw grid
-- structured relations do not need an `origin` or `width` specified.  Those
-- are parameters of the standard library grid, which provides a set of
-- standard functionality on top of the raw grid relations.


hi_cells:NewField('t', L.double):Load(function(xi,yi)
  if xi == 4 and yi == 10 then return 400 else return 0 end
end)
lo_cells:NewField('t', L.double):Load(0)
hi_cells:NewField('t_prev', L.double):Load(hi_cells.t)
lo_cells:NewField('t_prev', L.double):Load(lo_cells.t)
hi_cells:NewField('t_next', L.double):Load(hi_cells.t)
lo_cells:NewField('t_next', L.double):Load(lo_cells.t)
-- Here we define the necessary simulation variables.  Rather than explicitly
-- encode velocity, we choose to instead store the previous field value.
-- By copying, we effectively choose to initialize everything with 0 velocity.


local ebb shift_right_example( c : hi_cells )
  var left_c = L.Affine(hi_cells, {{1,0, -1},
                                   {0,1,  0}}, c)
  c.t_next = left_c.t
end
hi_cells:foreach(shift_right_example)
-- This computation doesn't accomplish anything for our simulation, but it
-- does demonstrate how we can use the special `L.Affine(...)` function to
-- access neighboring elements in a grid.  The first argument to `L.Affine()`
-- specifies which grid-structured relation we're performing a lookup into.
-- The second argument specifies an _affine transformation_ of the third
-- argument's "coordinates." (since the third argument is an key from
-- some grid strucured relation)  This second argument must be a constant
-- matrix, which we can interpret as follows:  Let `out` be the key returned
-- from the `L.Affine` call and `in` be the third argument input key.
-- (here `c`)  Then, abusing notation slightly
-- `out.x = 1 * in.x + 0 * in.y + (-1)` and
-- `out.y = 0 * in.x + 1 * in.y + 0`.  That is, `left_c` is just the cell
-- displaced by `-1` in the x-direction.


hi_cells:NewFieldMacro('__apply_macro', L.Macro(function(c, x, y)
  return ebb `L.Affine(hi_cells, {{1,0, x},
                                  {0,1, y}}, c)
end))
lo_cells:NewFieldMacro('__apply_macro', L.Macro(function(c, x, y)
  return ebb `L.Affine(lo_cells, {{1,0, x},
                                  {0,1, y}}, c)
end))
local ebb shift_right_example_2( c : hi_cells )
  var left_c = c(-1,0)
  c.t_next = left_c.t
end
hi_cells:foreach(shift_right_example_2)
-- Usually, we won't type out the entire call to `L.Affine`.  Instead we'll
-- use macros. (introduced in tutorial 15)  Ebb provides a special syntax
-- overloading feature when a macro is installed with the name
-- `'__apply_macro'`.  In this case, "function calls" to keys from the
-- relation are redirected to the macro, which is supplied with the arguments
-- as additional parameters.  This syntax is especially valuable for
-- giving a shorthand for relative offsets in grid data.


lo_cells:NewFieldMacro('hi', L.Macro(function(c)
  return ebb `L.Affine(hi_cells, {{2,0, 0},
                                  {0,2, 0}}, c)
end))
local ebb down_sample( c : lo_cells )
  var sum_t = c.hi(0,0).t + c.hi(0,1).t
            + c.hi(1,0).t + c.hi(1,1).t
  c.t = sum_t / 4.0
end
local ebb up_sample( c : lo_cells )
  var d_t = c.t_next - c.t
  c.hi(0,0).t_next += d_t
  c.hi(0,1).t_next += d_t
  c.hi(1,0).t_next += d_t
  c.hi(1,1).t_next += d_t
end
-- The second macro we define here lets us access the higher-resolution
-- grid from the lower resolution grid.  Using this connection, we can define
-- routines to down-sample the current t-field, and also to up-sample and
-- apply the diffusion results.  Notice how various access macros can be
-- chained together.  We first access the high-resolution grid with `c.hi`,
-- but then can immediately use the offset macro to locally navigate to
-- the other 3 cells covered by the low resolution cell.


local timestep    = L.Constant(L.double, 0.25)
local conduction  = L.Constant(L.double, 0.5)
local friction    = L.Constant(L.double, 0.95)

local ebb diffuse_lo( c : lo_cells )
  var avg = (   c(1,0).t + c(-1,0).t
              + c(0,1).t + c(0,-1).t ) / 4.0
  var d_t = avg - c.t
  c.t_next = c.t + timestep * conduction * d_t
end
local ebb wave_hi( c : hi_cells )
  var avg = (   c(1,0).t + c(-1,0).t
              + c(0,1).t + c(0,-1).t ) / 4.0

  var spatial_d_t   = avg - c.t
  var temporal_d_t  = (c.t - c.t_prev)

  c.t_next = c.t + friction * temporal_d_t
                 + timestep * conduction * spatial_d_t
end
-- Now, we define the simulaiton at each resolution.


local sum_t     = L.Global(L.double, 0)
local max_diff  = L.Global(L.double, 0)
local ebb measure_sum( c : hi_cells )
  sum_t += c.t
end
local ebb measure_diff( c : hi_cells )
  var diff = L.fmax( L.fmax( L.fabs(c.t - c(0,0).t),
                             L.fabs(c.t - c(0,1).t) ),
                     L.fmax( L.fabs(c.t - c(1,0).t),
                             L.fabs(c.t - c(1,1).t) ))
  max_diff max= diff
end

local ebb visualize_hi( c : hi_cells )
  vdb.color({ 0.5 * c.t + 0.5, 0.5-c.t, 0.5-c.t })
  var c = { L.xid(c), L.yid(c), 0 }
  vdb.point(c)
end
-- To define the visualization function, we use the debugging functions
-- `L.xid()` and `L.yid()` which recover the numeric ids identifying which
-- specific cell we're in.  We also define some debugging stats for the
-- console.  In particular, we expect that since we defined our simulation
-- carefully, we should preserve the sum of `t` and see a gradual decrease
-- in the gradient as the diffusion behavior eventually dominates.


for i=1,200 do
  lo_cells:foreach(down_sample)

  lo_cells:foreach(diffuse_lo)
  hi_cells:foreach(wave_hi)

  lo_cells:foreach(up_sample)

  -- step forward
  hi_cells:Swap('t_prev','t')
  hi_cells:Swap('t','t_next')
  lo_cells:Swap('t_prev','t')
  lo_cells:Swap('t','t_next')

  vdb.vbegin()
  vdb.frame()
    hi_cells:foreach(visualize_hi)
  vdb.vend()

  if i % 10 == 0 then -- measure statistics every 10 steps
    max_diff:set(0)
    sum_t:set(0)
    hi_cells:foreach(measure_sum)
    hi_cells:foreach(measure_diff)
    print( 'iteration #'..tostring(i),
           'max gradient: ', max_diff:get()..'   ',
           'sum_t:',         sum_t:get() )
  end
end
-- Our simulation loop down-samples the field, runs both simulations,
-- and then up-samples the diffusion results to combine with the wave
-- simulation step.  Then we cycle our buffers, visualize and collect
-- statistics.







