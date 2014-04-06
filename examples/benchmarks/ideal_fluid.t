import "compiler.liszt"

local Grid  = terralib.require 'compiler.grid'
local cmath = terralib.includecstring [[
#include <math.h>
#include <stdlib.h>
#include <time.h>



float rand_float()
{
      float r = (float)rand() / (float)RAND_MAX;
      return r;
}
]]
cmath.srand(cmath.time(nil));
local vdb   = terralib.require 'compiler.vdb'

local N = 150
local grid = Grid.New2dUniformGrid(N, N, {-N/2.0, -1.0}, N, N)

local viscosity     = 0.08
local dt            = L.NewGlobal(L.float, 0.01)


grid.cells:NewField('velocity', L.vec2f)
grid.cells.velocity:LoadConstant(L.NewVector(L.float, {0,0}))

grid.cells:NewField('velocity_prev', L.vec2f)
grid.cells.velocity_prev:LoadConstant(L.NewVector(L.float, {0,0}))



-----------------------------------------------------------------------------
--[[                        BOUNDARY CONDITIONS                          ]]--
-----------------------------------------------------------------------------

local velocity_zero = liszt_kernel(c : grid.cells)
    c.velocity = {0,0}
end


grid.cells:NewKernel('copy_boundary_kernel', liszt kernel (c)
fields
    dst
    src
end
    if c.is_left_bnd then
        c.[dst] = c.right.[src]
    elseif c.is_right_bnd then
        c.[dst] = c.left.[src]
    elseif c.is_up_bnd then
        c.[dst] = c.down.[src]
    elseif c.is_down_bnd then
        c.[dst] = c.up.[src]
    end
end)
function grid.copy_boundary(x)
    grid.cells.boundary:CopyField('boundary_temp', x)
    grid.cells.boundary.copy_boundary_kernel(x, 'boundary_temp')
    grid.cells.boundary:ReleaseField('boundary_temp')
end

grid.cells:NewKernel('invert_boundary_kernel', liszt kernel (c)
fields
    dst
    src
end
    var v : L.vec2f
    if c.is_left_bnd then
        v = c.right.[src]
        v[0] = -v[0]
    elseif c.is_right_bnd then
        v = c.left.[src]
        v[0] = -v[0]
    elseif c.is_up_bnd then
        v = c.down.[src]
        v[1] = -v[1]
    elseif c.is_down_bnd then
        v = c.up.[src]
        v[1] = -v[1]
    end
    c.[dst] = v
end)
function grid.invert_boundary(x)
    grid.cells.boundary:CopyField('boundary_temp', x)
    grid.cells.boundary.copy_boundary_kernel(x, 'boundary_temp')
    grid.cells.boundary:ReleaseField('boundary_temp')
end

-----------------------------------------------------------------------------
--[[                         LINEAR SYSTEM SOLVE                         ]]--
-----------------------------------------------------------------------------


-- linear system solve
local edge_weight = L.NewGlobal(L.float, 0.0)
local diag_weight = L.NewGlobal(L.float, 0.0)

grid.cells:NewKernel('jacobi_step', liszt kernel (c)
fields
    x_temp
    x
    b
in
    var edge_sum = edge_weight *
        (c.left.[x] + c.right.[x] + c.up.[x] + c.down.[x])

    c.[x_temp] = (c.[b] - edge_sum) / diag_weight
end)

function grid.lin_solve_cells(x, b, ew, dw, bnd_conditions)
    grid.cells:CopyField(x, b)
    grid.cells:CopyField('lin_solve_temp', x)
    edge_weight:setTo(ew)
    diag_weight:setTo(dw)

    -- INIT ?

    for i = 1,20 do
        grid.cells.interior.jacobi_step(x,b)
        grid.cells:SwapFields(x, 'lin_solve_temp')

        -- enforce boundary conditions
        grid[bnd_conditions](x)
    end

    grid.cells:FreeField('lin_solve_temp')
end


-----------------------------------------------------------------------------
--[[                             DIFFUSE                                 ]]--
-----------------------------------------------------------------------------

local function diffuse_velocity(grid)
    -- Why the N*N term?  I don't get that...
    local laplacian_weight  = dt:value() * viscosity * N * N
    local diagonal          = 1.0 + 4.0 * laplacian_weight
    local edge              = -laplacian_weight

    -- swap current velocity into previous and solve for new current
    -- velocity
    grid.cells:SwapFields('velocity', 'velocity_prev')
    grid.lin_solve_cells('velocity', 'velocity_prev',
                         edge, diagonal, 'invert_boundary')
end

-----------------------------------------------------------------------------
--[[                             ADVECT                                  ]]--
-----------------------------------------------------------------------------

local cell_w = grid:cellWidth()
local cell_h = grid:cellHeight()

local advect_dt = L.NewGlobal(L.float, 0.0)
grid.cells:NewField('lookup_pos', L.vec2f):Load(L.NewVector(L.float, {0,0}))
grid.cells:NewField('lookup_from', grid.dual_cells):Load(0)

grid.cells:NewKernel('advect_where_from', liszt_kernel(c)
    var offset      = - c.velocity_prev
    -- Make sure all our lookups are appropriately confined
    c.lookup_pos    = grid.snap_to_grid(c.center + advect_dt * offset)
end)

grid.cells:NewKernel('advect_point_locate', liszt_kernel(c)
    c.lookup_from   = grid.dual_locate(c.lookup_pos)
end)

grid.cells:NewKernel('advect_interpolate_velocity', liszt kernel(c)
    if not c.is_bnd then
        var dc      = c.lookup_from
        var frac    = c.lookup_pos - dc.center
        -- figure out fractional position in the dual cell in range [0.0, 1.0]
        var xfrac   = frac[0] / cell_w + 0.5 
        var yfrac   = frac[1] / cell_h + 0.5

        -- interpolation constants
        var x1      = L.float(xfrac)
        var y1      = L.float(yfrac)
        var x0      = L.float(1.0 - xfrac)
        var y0      = L.float(1.0 - yfrac)

        c.velocity  = x0 * y0 * dc.upleft.velocity_prev
                    + x1 * y0 * dc.upright.velocity_prev
                    + x0 * y1 * dc.downleft.velocity_prev
                    + x1 * y1 * dc.downright.velocity_prev
    end
end)

local function advect_velocity(grid)
    -- Why N?
    advect_dt:setTo(dt:value() * N)

    grid.cells:SwapFields('velocity', 'velocity_prev')

    grid.cells.advect_where_from(grid.cells)
    grid.cells.advect_point_locate()
    grid.cells.interior.advect_interpolate_velocity()

    grid.invert_boundary('velocity')
end

-----------------------------------------------------------------------------
--[[                             PROJECT                                 ]]--
-----------------------------------------------------------------------------


grid.cells:NewField('divergence', L.float):Load(0)
grid.cells:NewField('p', L.float):Load(0)
grid.cells:NewKernel('compute_divergence', liszt kernel (c)
    -- why the factor of N?
    var vx_dx = c.right.velocity[0] - c.left.velocity[0]
    var vy_dy = c.up.velocity[1]   - c.down.velocity[1]
    c.divergence = L.float(-(0.5/N)*(vx_dx + vy_dy))
end)

grid.cells:NewKernel('compute_projection', liszt kernel (c)
    var grad = L.vec2f(0.5 * N * { c.right.p - c.left.p,
                                   c.up.p   - c.down.p })
    c.velocity = c.velocity_prev - grad
end)

local function project_velocity(grid)
    -- Why the N*N term?  I don't get that...
    --local laplacian_weight  = dt:value() * viscosity * N * N
    local diagonal          =  4.0
    local edge              = -1.0

    grid.cells.interior.compute_divergence()
    grid.copy_boundary('divergence')

    grid.lin_solve_cells('p', 'divergence', edge, diagonal, 'copy_boundary')

    grid.cells:SwapFields('velocity', 'velocity_prev')
    grid.cells.interior.compute_projection()
    grid.invert_boundary('velocity')
end







--[[

local N_particles = (N-1)*(N-1)
local particles = L.NewRelation(N_particles, 'particles')

particles:NewField('dual_cell', grid.dual_cells)
    :Load(function(i) return i end)

particles:NewField('next_pos', L.vec2f):Load(L.NewVector(L.float, {0,0}))
particles:NewField('pos', L.vec2f):Load(L.NewVector(L.float, {0,0}))
(liszt kernel (p : particles) -- init...
    p.pos = p.dual_cell.center
end)(particles)

local locate_particles = liszt kernel (p : particles)
    p.dual_cell = grid.dual_locate(p.pos)
end

local compute_particle_velocity = liszt kernel (p : particles)
    var dc      = p.dual_cell
    var frac    = p.pos - dc.center
    -- figure out fractional position in the dual cell in range [0.0, 1.0]
    var xfrac   = frac[0] / cell_w + 0.5 
    var yfrac   = frac[1] / cell_h + 0.5

    -- interpolation constants
    var x1      = L.float(xfrac)
    var y1      = L.float(yfrac)
    var x0      = L.float(1.0 - xfrac)
    var y0      = L.float(1.0 - yfrac)

    p.next_pos  = p.pos + N *
        ( x0 * y0 * dc.upleft.velocity
        + x1 * y0 * dc.upright.velocity
        + x0 * y1 * dc.downleft.velocity
        + x1 * y1 * dc.downright.velocity )
end

local update_particle_pos = liszt kernel (p : particles)
    var r = L.vec2f({ cmath.rand_float() - 0.5, cmath.rand_float() - 0.5 })
    var pos = p.next_pos + dt * r
    p.pos = grid.snap_to_grid(pos)
end

]]--


-----------------------------------------------------------------------------
--[[                             MAIN LOOP                               ]]--
-----------------------------------------------------------------------------

--grid.cells:print()

local source_strength = 100.0
local source_velocity = liszt kernel (c : grid.cells)
    if cmath.fabs(c.center[0]) < 1.75 and
       cmath.fabs(c.center[1]) < 1.75 and
       not c.is_bnd
    then
        c.velocity += dt * source_strength * { 0.0, 1.0 }
    end
end

local draw_grid = liszt kernel (c : grid.cells)
    var color = {1.0, 1.0, 1.0}
    vdb.color(color)
    var p : L.vec3f = { c.center[0],   c.center[1],   0.0 }
    var vel = c.velocity
    var v = L.vec3f({ vel[0], vel[1], 0.0 })
    --if not c.is_bnd then
    vdb.line(p, p+v*N)
end

local draw_particles = liszt kernel (p : particles)
    var color = {1.0,1.0,0.0}
    vdb.color(color)
    var pos : L.vec3f = { p.pos[0], p.pos[1], 0.0 }
    vdb.point(pos)
end

for i = 1, 1000 do
    -- injecting velocity
    if math.floor(i / 70) % 2 == 0 then
        source_velocity(grid.cells)
        velocity_swap(grid.cells)
        velocity_update_bnd(grid.cells)
    end

    diffuse_velocity(grid)
    project_velocity(grid)

    advect_velocity(grid)
    project_velocity(grid)

    --compute_particle_velocity(particles)
    --update_particle_pos(particles)
    --locate_particles(particles)

    vdb.vbegin()
        vdb.frame()
        --draw_grid(grid.cells)
        draw_particles(particles)
    vdb.vend()

end

--grid.cells:print()
