import "compiler.liszt"
local Grid  = L.require 'domains.grid'

L.default_processor = L.GPU

local xn = 2
local yn = xn

local grid = Grid.NewGrid2d{size           = {xn, yn},
                            origin         = {0, 0},
                            width          = 1,
                            height         = 1,
                            boundary_depth = {1, 1},
                            periodic_boundary = {true, true} }

function main ()
	-- declare a global to store the computed centroid of the grid
	local com = L.NewGlobal(L.vector(L.float, 3), {0, 0, 0})

	-- compute centroid
	local sum_pos = liszt kernel(c : grid.cells)
		com += L.vec3f(c.center)
	end
	sum_pos(grid.cells)

	local center = com:get() / grid.cells:Size()

	-- output
	print("center of mass is: (" .. center.data[1] .. ", " .. center.data[2] .. ', ' .. center.data[3] .. ")")
end

main()