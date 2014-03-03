import "compiler.liszt"

local Grid = terralib.require 'compiler.grid'

local grid = Grid.New2dUniformGrid(20,20)

grid.cells:NewField('density', L.float)
grid.cells.density:LoadConstant(0)

grid.cells:NewField('velocity', L.vector(L.float, 2))
grid.cells.velocity:LoadConstant(L.NewVector(L.float, {0,0}))

grid.cells:NewField('temp_density', L.float)
grid.cells.temp_density:LoadConstant(0)


local diffuse = liszt kernel(c : grid.cells)
  var sq_density = c.density * c.density

  c.temp_density = sq_density
end

local diffuse2 = liszt_kernel(c : grid.cells)
  c.density = c.temp_density
end

for i=1,100 do
  diffuse(grid.cells)
  diffuse2(grid.cells)
end