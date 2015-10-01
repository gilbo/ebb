
-- This test actually doesn't need to run on the GPU,
-- it's just designed to stress test GPU execution in particular
import 'ebb.liszt'

-- A 17x17 grid will, at current settings,
-- will force liszt to run the generated kernel
-- on multiple GPU blocks
local N=17

local Grid  = require 'ebb.domains.grid'
local grid = Grid.NewGrid2d{size           = {N, N},
                            origin         = {0, 0},
                            width          = {1, 1},
                            boundary_depth = {1, 1},
                            periodic_boundary = {true, true} }

function main ()
	-- declare a global to store the computed centroid of the grid
	local com = L.Global(L.vector(L.float, 2), {0, 0})

	-- compute centroid
	local sum_pos = liszt(c : grid.cells)
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