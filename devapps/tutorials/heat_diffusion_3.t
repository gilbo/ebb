import "ebb"


-- In this tutorial, we're going to write heat diffusion yet again.
-- This time however, we're going to do heat diffusion on a grid.
-- The basic structure will be the same, but we'll get to see
-- a pretty different data structure this time around.

local Grid = require 'ebb.domains.grid'
local cmath = terralib.includecstring '#include <math.h>'

-- We don't have to load grid data from a file.
-- Instead, we supply the necessary parameters
local N = 50
local grid = Grid.NewGrid2d({
    size   = {N, N},
    origin = {-N/2, -N/2},
    width  = {N, N},
})
local cellw = grid:xCellWidth() / 2

------------------------------------------------------------------------------

-- Again, we declare constants and fields;
-- we declare our data fields on the cells of the grid.

local timestep    = L.Constant(L.double, 0.45)
local conduction  = L.Constant(L.double, 1.0)

grid.cells:NewField('temperature', L.double)      :Load(0)
grid.cells:NewField('d_temperature', L.double)  :Load(0)

------------------------------------------------------------------------------

-- Here we define the compute kernels again.
local ebb compute_update ( c : grid.cells )
  -- Notice that now, instead of a for loop to loop over edges/neighbors,
  -- we're using relative offsets from the "centered" cell.
  -- This expression sums the temperatures in cells to the
  --    left, right, top and bottom
  var sum_t = c(1,0).temperature + c(-1,0).temperature
            + c(0,1).temperature + c(0,-1).temperature
  var avg_t = sum_t / 4.0
  c.d_temperature = timestep * conduction * (avg_t - c.temperature)
end

local ebb apply_update ( c : grid.cells )
  c.temperature += c.d_temperature
end

------------------------------------------------------------------------------

-- But hold on a second.  If we map the `compute_update` function
-- over the entire grid, then we're going to try to access cells
-- that don't exist past the edges.  We only want to map the update
-- over the "interior" cells, and then impose a boundary condition
-- on the remaining "boundary" cells.

-- To handle boundary conditions in Ebb, we use subsets
-- Our grid has two subsets: `interior` and `boundary`

-- Let's define a Dirichlet boundary condition that will
-- create a sinusoidal heat pattern
local PI_N = 4.0 * math.pi / N
local ebb dirichlet_condition ( c : grid.cells )
  -- c.center is the position of the cell center
  -- using the coordinate system we defined when we created
  -- the grid.
  var pos = c.center
  var wave = L.cos(pos[0] * PI_N) * L.cos(pos[1] * PI_N)
  c.temperature = wave + 1.0
end

-- We'll actually only need to set the Dirichlet condition once and
-- then leave it alone.  Notice we are mapping over the subset here
grid.cells.boundary:foreach(dirichlet_condition)


------------------------------------------------------------------------------


-- WARNING / EXTRA VDB
local vdb  = require('ebb.lib.vdb')
local cold = L.Constant(L.vec3f,{0.5,0.5,0.5})
local hot  = L.Constant(L.vec3f,{1.0,0.0,0.0})
local ebb debug_quad_draw ( c : grid.cells )
  var scale = L.float(0.5 * c.temperature)
  var pos   = c.center

  vdb.color((1.0-scale)*cold + scale*hot)
  vdb.triangle({ pos[0] - cellw, pos[1] - cellw, 0 },
               { pos[0] + cellw, pos[1] - cellw, 0 },
               { pos[0] - cellw, pos[1] + cellw, 0 })
  vdb.triangle({ pos[0] + cellw, pos[1] + cellw, 0 },
               { pos[0] - cellw, pos[1] + cellw, 0 },
               { pos[0] + cellw, pos[1] - cellw, 0 })
end
-- END EXTRA VDB CODE


------------------------------------------------------------------------------

for i = 1,500 do
  -- Note that we're mapping the main computation over the grid's interior
  grid.cells.interior:foreach(compute_update)
  grid.cells.interior:foreach(apply_update)

  -- EXTRA: VDB (For visualization)
  vdb.vbegin()
    vdb.frame() -- this call clears the canvas for a new frame
    grid.cells:foreach(debug_quad_draw)
  vdb.vend()
  -- END EXTRA
end


