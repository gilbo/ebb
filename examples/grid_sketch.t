import "compiler.liszt"

local Grid = terralib.require 'compiler.grid'
local cmath = terralib.includecstring '#include <math.h>'

local N = 3
local grid = Grid.New2dUniformGrid(N, N)

grid.cells:NewField('density', L.float)
grid.cells.density:LoadConstant(1)

grid.cells:NewField('density_prev', L.float)
grid.cells.density:LoadConstant(1)

grid.cells:NewField('density_temp', L.float)
grid.cells.density_temp:LoadConstant(0)

grid.cells:NewField('velocity', L.vector(L.float, 2))
grid.cells.velocity:LoadConstant(L.NewVector(L.float, {0,0}))

grid.cells:NewField('velocity_prev', L.vector(L.float, 2))
grid.cells.velocity:LoadConstant(L.NewVector(L.float, {0,0}))

grid.cells:NewField('velocity_temp', L.vector(L.float, 2))
grid.cells.velocity_temp:LoadConstant(L.NewVector(L.float, {0,0}))

grid.cells:NewField('position', L.vector(L.float, 2))

-- TODO: Init position

local a     = L.NewScalar(L.float, 0)
local dt0   = L.NewScalar(L.float, 0)
local h     = L.NewScalar(L.float, 0)

local diff  = L.NewScalar(L.float, 1)
local visc  = L.NewScalar(L.float, 0.01)
local dt    = L.NewScalar(L.float, 0.01)

-----------------------------------------------------------------------------
--[[                             ADD SOURCE                              ]]--
-----------------------------------------------------------------------------

local addsource_density = liszt_kernel(c : grid.cells)
    c.density = c.density_prev
end

local addsource_velocity = liszt_kernel(c : grid.cells)
    c.velocity = c.velocity_prev
end

-----------------------------------------------------------------------------
--[[                             VELSTEP                                 ]]--
-----------------------------------------------------------------------------



-----------------------------------------------------------------------------
--[[                             DENSTEP                                 ]]--
-----------------------------------------------------------------------------
-----------------------------------------------------------------------------
--[[                             DIFFUSE                                 ]]--
-----------------------------------------------------------------------------

local function diffuse_preprocess(val)
    a = dt * val * N * N
end

local diffuse_density = liszt_kernel(c : grid.cells)
-- TODO: Deal with boundary conditions
    c.density_temp =
        ( c.density_prev +
          a * ( c.left.density + c.right.density +
                c.top.density  + c.bot.density
              )
        ) / (1 + 4 * a)
end

local diffuse_density_update = liszt_kernel(c : grid.cells)
    c.density = c.density_temp
end

-----------------------------------------------------------------------------
--[[                             ADVECT                                  ]]--
-----------------------------------------------------------------------------

local function advect_preprocess()
    dt0 = dt * N
end

local advect_density = liszt_kernel(c : grid.cells)
    var x = c.position[1] - dt0 * c.velocity[1]

    if x < 0.5 then
        x = 0.5
    end

    if x > N + 0.5 then
        x = N + 0.5
    end

    var i0 = cmath.floor(x) + 1
    var i1 = i0 + 1

    var y = c.position[2] - dt0 * c.velocity[2]

    if y < 0.5 then
        y = 0.5
    end

    if y > N + 0.5 then
        y = N + 0.5
    end

    var j0 = cmath.floor(y) + 1
    var j1 = j0 + 1

    var s1 = x - i0
    var s0 = 1 - s1
    var t1 = y - j0
    var t0 = 1 - t1

    -- TODO: Implement casting for this
    c.density_temp =
        ( s0 * ( t0 * c.density_prev.locate(i0, j0) +
                 t1 * c.density_prev.locate(i0, j1)
               ) +
          s1 * ( t0 * c.density_prev.locate(i1, j0) +
                 t1 * c.density_prev.locate(i1, j1)
               )
        )
end

local advect_density_update = liszt_kernel(c : grid.cells)
    c.density = c.density_temp
end

-----------------------------------------------------------------------------
--[[                             PROJECT                                 ]]--
-----------------------------------------------------------------------------

local function project_preprocess()
    h = 1 / N
end

local project_1 = liszt_kernel(c : grid.cells)
    c.velocity_prev[2] =
        -0.5 * h * ( c.right.velocity[1] - c.left.velocity[1] +
                     c.right.velocity[2] - c.left.velocity[2]
                   )
end

local project_2 = liszt_kernel(c : grid.cells)
    c.velocity[1] =
        0.25 * ( c.velocity_prev[2] +
                 c.left.velocity_prev[1] + c.right.velocity_prev[1] +
                 c.top.velocity_prev[1]  + c.bot.velocity_prev[1])
end

local project_3 = liszt_kernel(c : grid.cells)
    c.velocity_temp[1] =
        c.velocity[1] - 0.5 * ( c.right.velocity_prev[1] -
                                c.left.velocity_prev[1]
                              ) / h
    c.velocity_temp[2] =
        c.velocity[2] - 0.5 * ( c.bot.velocity_prev[1] -
                                c.top.velocity_prev[1]
                              ) / h
end

local project_update = liszt_kernel(c : grid.cells)
    c.velocity = c.velocity_temp
end

-----------------------------------------------------------------------------
--[[                             MAIN LOOP                               ]]--
-----------------------------------------------------------------------------

grid.cells:print()

for i = 1, 1000 do
    -- velocity step
    addsource_velocity(grid.cells)
    
    -- density step
    addsource_density(grid.cells)
    
    diffuse_preprocess(diff)
    diffuse_density(grid.cells) -- TODO: Repeat this 20 times
    diffuse_density_update(grid.cells)

    advect_preprocess()
    advect_density(grid.cells)
    advect_density_update(grid.cells)

    project_preprocess()
    project_1(grid.cells)
    project_2(grid.cells) -- TODO: Repeat this 20 times
    project_3(grid.cells)
    project_update(grid.cells)
end

grid.cells:print()

