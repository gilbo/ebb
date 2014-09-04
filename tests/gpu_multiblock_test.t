import 'compiler.liszt'
L.default_processor = L.GPU

-- A 17x17 grid will, at current settings, force liszt to run the generated kernel
-- on multiple blocks
local N=17

local Grid  = L.require 'domains.grid'
local grid = Grid.NewGrid2d{size           = {N, N},
                            origin         = {0, 0},
                            width          = {1, 1},
                            boundary_depth = {1, 1},
                            periodic_boundary = {true, true} }

function main ()
	-- declare a global to store the computed centroid of the grid
	local com = L.NewGlobal(L.vector(L.float, 2), {0, 0})

	-- compute centroid
	local sum_pos = liszt kernel(c : grid.cells)
		com += c.center
	end
	sum_pos(grid.cells)

	local center = com:get() / grid.cells:Size()

	-- output
	print("center is: (" .. center.data[1] .. ", " .. center.data[2] .. ')')
end

main()