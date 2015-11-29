import "compiler.liszt"
local Grid  = L.require 'domains.grid'

L.SetDefaultProcessor(L.GPU)

local xn = 17
local yn = xn

local grid = Grid.NewGrid2d{size           = {xn, yn},
                            origin         = {0, 0},
                            width          = {1, 1},
                            boundary_depth = {1, 1},
                            periodic_boundary = {true, true} }

function main ()
	-- declare a global to store the computed centroid of the grid
	local com = L.NewGlobal(L.vector(L.float, 2), {0, 0})

	-- compute centroid
	local liszt sum_pos (c : grid.cells)
		com += c.center
	end
	grid.cells:foreach(sum_pos)

	local center = com:get() / grid.cells:Size()

	-- output
	print("center is: (" .. center.data[1] .. ", " .. center.data[2] .. ')')
end

main()
