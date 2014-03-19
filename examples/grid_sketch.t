import "compiler.liszt"

local Grid = terralib.require 'compiler.grid'
local cmath = terralib.includecstring '#include <math.h>'

local N = 3
local grid = Grid.New2dUniformGrid(N, N, {0.0, 0.0}, N, N)



--grid.cells:NewField('density', L.float)
--grid.cells.density:LoadConstant(1)
--
--grid.cells:NewField('density_prev', L.float)
--grid.cells.density_prev:LoadConstant(1)
--
--grid.cells:NewField('density_temp', L.float)
--grid.cells.density_temp:LoadConstant(0)
--
--grid.cells:NewField('density_prev_temp', L.float)
--grid.cells.density_prev_temp:LoadConstant(0)

grid.cells:NewField('velocity', L.vector(L.float, 2))
grid.cells.velocity:LoadConstant(L.NewVector(L.float, {0,0}))

grid.cells:NewField('velocity_prev', L.vector(L.float, 2))
grid.cells.velocity_prev:LoadConstant(L.NewVector(L.float, {0,0}))

--grid.cells:NewField('velocity_temp', L.vector(L.float, 2))
--grid.cells.velocity_temp:LoadConstant(L.NewVector(L.float, {0,0}))
--
--grid.cells:NewField('velocity_prev_temp', L.vector(L.float, 2))
--grid.cells.velocity_prev_temp:LoadConstant(L.NewVector(L.float, {0,0}))


local dt0   = L.NewGlobal(L.float, 0)
local h     = L.NewGlobal(L.float, 0)

local diff  = L.NewGlobal(L.float, 1)
local visc  = L.NewGlobal(L.float, 0.01)
local dt    = L.NewGlobal(L.float, 0.01)

-----------------------------------------------------------------------------
--[[                           PREPROCESSORS                             ]]--
-----------------------------------------------------------------------------

local function diffuse_preprocess(val)
    a:setTo(dt:value() * val * N * N)
end

local function advect_preprocess()
    dt0:setTo(dt:value() * N)
end

local function project_preprocess()
    h:setTo(1 / N)
end

-----------------------------------------------------------------------------
--[[                             UPDATES                                 ]]--
-----------------------------------------------------------------------------

--local density_update = liszt_kernel(c : grid.cells)
--    c.density = c.density_temp
--end
--
--local density_prev_update = liszt_kernel(c : grid.cells)
--    c.density_prev = c.density_prev_temp
--end

local velocity_zero = liszt_kernel(c : grid.cells)
    c.velocity = {0,0}
end

local velocity_swap = liszt_kernel(c : grid.cells)
    c.velocity_prev = c.velocity
end

--local velocity_update = liszt_kernel(c : grid.cells)
--    c.velocity = c.velocity_temp
--end
--
--local velocity_prev_update = liszt_kernel(c : grid.cells)
--    c.velocity_prev = c.velocity_prev_temp
--end

-----------------------------------------------------------------------------
--[[                             VELSTEP                                 ]]--
-----------------------------------------------------------------------------
-----------------------------------------------------------------------------
--[[                             ADD SOURCE                              ]]--
-----------------------------------------------------------------------------

--local addsource_velocity = liszt_kernel(c : grid.cells)
--    c.velocity_temp = c.velocity + dt * c.velocity_prev
--end

-----------------------------------------------------------------------------
--[[                             DIFFUSE                                 ]]--
-----------------------------------------------------------------------------

local a     = L.NewGlobal(L.float, 0)
-- Execute this 20 times
local lin_solve_diffuse_kernel = liszt_kernel(c : grid.cells)
    c.x = x0 + a * (c.left.x + c.right.x + c.top.x + c.bottom.x) / (1+4*a)
end

local function diffuse_velocity(grid)
    local diffusion_constant = 1.0
    a:setTo(dt:value() * diffusion_constant * N * N)
    for i=1,20 do
        lin_solve_diffuse_kernel(grid.cells)
        --set_boundary(grid.cells)
    end
end

local diffuse_velocity = liszt_kernel(c : grid.cells)
    
end

local diffuse_velocity_prev = liszt_kernel(c : grid.cells)
    var sum_vel = ( c.left.velocity_prev + c.right.velocity_prev +
                    c.top.velocity_prev  + c.bot.velocity_prev )
    c.velocity_prev_temp = ( c.velocity + a * sum_vel) / (1 + 4 * a)
end

-----------------------------------------------------------------------------
--[[                             ADVECT                                  ]]--
-----------------------------------------------------------------------------

local cell_w = grid:cellWidth()
local cell_h = grid:cellHeight()
local advect_density = liszt_kernel(c : grid.cells)
    var offset          = - c.velocity
    var xfrac           = offset[0] - cell_w * cmath.floor(offset[0] / cell_w)
    var yfrac           = offset[1] - cell_h * cmath.floor(offset[1] / cell_h)
    var prev_position   = c.center + dt0 * offset

    -- locate dual cell (TODO: REMOVE THIS INTO ANOTHER KERNEL)
    var dc = grid.dual_locate(prev_position)

    -- interpolate from corners of dual cell
    var x1 = L.float(xfrac)
    var y1 = L.float(yfrac)
    var x0 = L.float(1.0 - xfrac)
    var y0 = L.float(1.0 - yfrac)
    var dens = x0 * y0 * dc.topleft.density_prev
             + x1 * y0 * dc.topright.density_prev
             + x0 * y1 * dc.bottomleft.density_prev
             + x1 * y1 * dc.bottomright.density_prev

    c.density = dens
end

-----------------------------------------------------------------------------
--[[                             PROJECT                                 ]]--
-----------------------------------------------------------------------------

local project_velocity_prev_1 = liszt_kernel(c : grid.cells)
    c.velocity_temp = { c.velocity[1],
        L.float(-0.5) * h *
            ( c.right.velocity_prev[1] - c.left.velocity_prev[1] +
              c.right.velocity_prev[2] - c.left.velocity_prev[2] )
    }
end

local project_velocity_prev_2 = liszt_kernel(c : grid.cells)
    c.velocity_temp = {
        L.float(0.25) * ( c.velocity[2] +
            c.left.velocity[1] + c.right.velocity[1] +
            c.top.velocity[1]  + c.bot.velocity[1]
        ),
        c.velocity[2]
    }
end

local project_velocity_prev_3 = liszt_kernel(c : grid.cells)
    c.velocity_prev_temp = {
        c.velocity_prev[1] - L.float(0.5) *
            ( c.right.velocity[1] - c.left.velocity[1] ) / h,
        c.velocity_prev[2] - L.float(0.5) *
            ( c.bot.velocity[1] - c.top.velocity[1] ) / h
    }
end

local project_velocity_1 = liszt_kernel(c : grid.cells)
    c.velocity_prev_temp = {
        c.velocity_prev[1],
        -L.float(0.5) * h *
            ( c.right.velocity[1] - c.left.velocity[1] +
              c.right.velocity[2] - c.left.velocity[2] )
    }
end

local project_velocity_2 = liszt_kernel(c : grid.cells)
    c.velocity_prev_temp = {
        L.float(0.25) * ( c.velocity_prev[2] +
            c.left.velocity_prev[1] + c.right.velocity_prev[1] +
            c.top.velocity_prev[1]  + c.bot.velocity_prev[1]
        ),
        c.velocity_prev[2]
    }
end

local project_velocity_3 = liszt_kernel(c : grid.cells)
    c.velocity_temp = {
        c.velocity[1] - L.float(0.5) *
            ( c.right.velocity_prev[1] - c.left.velocity_prev[1] ) / h,
        c.velocity[2] - L.float(0.5) *
            ( c.bot.velocity_prev[1] - c.top.velocity_prev[1] ) / h
    }
end

-----------------------------------------------------------------------------
--[[                             DENSTEP                                 ]]--
-----------------------------------------------------------------------------
-----------------------------------------------------------------------------
--[[                             ADD SOURCE                              ]]--
-----------------------------------------------------------------------------

local addsource_density = liszt_kernel(c : grid.cells)
    c.density_temp = c.density + dt * c.density_prev
end

-----------------------------------------------------------------------------
--[[                             DIFFUSE                                 ]]--
-----------------------------------------------------------------------------

local diffuse_density_prev = liszt_kernel(c : grid.cells)
    c.density_prev_temp =
        ( c.density +
          a * ( c.left.density_prev + c.right.density_prev +
                c.top.density_prev  + c.bot.density_prev
              )
        ) / (1 + 4 * a)
end

-----------------------------------------------------------------------------
--[[                             ADVECT                                  ]]--
-----------------------------------------------------------------------------

local advect_density = liszt_kernel(c : grid.cells)
--[[
    var x = c.idx[1] - dt0 * c.velocity[1]
--    var x = 0

    if x < 0.5 then
--        x = 0.5
    end

    if x > N + 0.5 then
--        x = N + 0.5
    end

    var i0 = cmath.floor(x) + 1
    var i1 = i0 + 1

    var y = c.idx[2] - dt0 * c.velocity[2]
--    var y = 0

    if y < 0.5 then
--        y = 0.5
    end

    if y > N + 0.5 then
--        y = N + 0.5
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
]]
end

-----------------------------------------------------------------------------
--[[                             MAIN LOOP                               ]]--
-----------------------------------------------------------------------------

grid.cells:print()

for i = 1, 1000 do
    -- velocity step
    addsource_velocity(grid.cells)
    velocity_update(grid.cells)

    diffuse_preprocess(diff:value())
    diffuse_velocity_prev(grid.cells)

    project_preprocess()
    project_velocity_prev_1(grid.cells)
    velocity_update(grid.cells)

    for j = 1, 20 do
        project_velocity_prev_2(grid.cells) -- TODO: Repeat this 20 times
        velocity_update(grid.cells)
    end

    project_velocity_prev_3(grid.cells)
    velocity_update(grid.cells)
    velocity_prev_update(grid.cells)   

    project_preprocess()
    project_velocity_1(grid.cells)
    velocity_prev_update(grid.cells)

    for j = 1, 20 do
        project_velocity_2(grid.cells) -- TODO: Repeat this 20 times
        velocity_prev_update(grid.cells)
    end

    project_velocity_3(grid.cells)
    velocity_update(grid.cells)
    velocity_prev_update(grid.cells)   

    -- density step
    addsource_density(grid.cells)
    density_update(grid.cells)

    diffuse_preprocess(diff:value())
    
    for j = 1, 20 do
        diffuse_density_prev(grid.cells) -- TODO: Repeat this 20 times
        density_prev_update(grid.cells)
    end

    advect_preprocess()
    advect_density(grid.cells)
    -- maybe no need to do this.. for now
    --density_update(grid.cells)
end

grid.cells:print()

