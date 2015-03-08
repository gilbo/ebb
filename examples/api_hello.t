import 'compiler.liszt'

local Grid = L.require('domains.grid')

local N = 5 -- 5x5 cell 2d grid
local width = 2.0 -- with size 2.0 x 2.0, and bottom-left at 0,0
local grid = Grid.NewGrid2d{
    size   = {N,N},
    origin = {0,0},
    width  = width,
    height = width,
}

-- load via a function...
grid.cells:NewField('temperature', L.double):Load(function(i)
  if i == 0 then return N*N else return 0 end
end)
-- load a constant
grid.cells:NewField('d_temperature', L.double):Load(0)

local K = L.Global(L.double, 1.0)

local liszt compute_diffuse ( c : grid.cells )
  if not c.in_boundary then
    var sum_diff = c( 1,0).temperature - c.temperature
                 + c(-1,0).temperature - c.temperature
                 + c(0, 1).temperature - c.temperature
                 + c(0,-1).temperature - c.temperature

    c.d_temperature = K * sum_diff
  end
end

local liszt apply_diffuse ( c : grid.cells )
  c.temperature += c.d_temperature
end

for i = 1, 1000 do
  grid.cells:map(compute_diffuse)
  grid.cells:map(apply_diffuse)
end
