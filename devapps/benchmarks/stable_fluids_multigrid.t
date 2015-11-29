import "ebb"
local L = require "ebblib"
--L.SetDefaultProcessor(L.GPU)

local Grid  = require 'ebb.domains.grid'
local cmath = terralib.includecstring [[
#include <math.h>
#include <stdlib.h>
#include <time.h>

int nancheck(float val) {
    return isnan(val);
}

float rand_float()
{
      float r = (float)rand() / (float)RAND_MAX;
      return r;
}
]]
cmath.srand(cmath.time(nil));
local vdb   = require 'ebb.lib.vdb'

local N = 128
local PERIODIC = true
local INSERT_DELETE = false -- exercise these features in periodic mode...
local period = {false,false}
local origin = {-N/2.0, -1.0}
if PERIODIC then
    period = {true,true}
    origin = {-N/2, -N/2}
end
local grid = Grid.NewGrid2d {
    size   = {N, N},
    origin = origin,
    width  = {N, N},
    periodic_boundary = period,
}

local XORIGIN       = origin[1]
local YORIGIN       = origin[2]
local viscosity     = 0.08
local dt            = L.Global(L.float, 0.01)



grid.cells:NewField('velocity', L.vec2f)
grid.cells.velocity:Load({0,0})

grid.cells:NewField('velocity_prev', L.vec2f)
grid.cells.velocity_prev:Load({0,0})

-----------------------------------------------------------------------------
--[[                        MULTI-GRID DOMAIN                            ]]--
-----------------------------------------------------------------------------

local MultiGrid = require 'ebb.domains.multigrid'
local pyramid = MultiGrid.NewMultiGrid2d {
    base_rel = grid.cells,
    top_resolution = 16,
}

for mgrid in pyramid:levelIter() do
    mgrid.cells:NewField('velocity', L.vec2f):Load({0,0})
    mgrid.cells:NewField('vel_shadow', L.vec2f):Load({0,0})
    mgrid.cells:NewField('velocity_prev', L.vec2f):Load({0,0})
end

local ebb lift_up_velocity ( c )
    c.velocity      = c.down_cell.velocity
    c.velocity_prev = c.down_cell.velocity_prev
    --c.vel_shadow
--    var sum = c.down_cell(0,0).velocity + c.down_cell(1,0).velocity
--            + c.down_cell(0,1).velocity + c.down_cell(1,1).velocity
--    var avg = sum / 4.0f
--    c.velocity      = avg
--    c.vel_shadow    = avg
--    var sum_prev =
--            c.down_cell(0,0).velocity_prev + c.down_cell(1,0).velocity_prev
--          + c.down_cell(0,1).velocity_prev + c.down_cell(1,1).velocity_prev
--    var avg_prev = sum / 4.0f
--    c.velocity_prev = avg_prev
end

-- linearly interpolate values down
local ebb pull_down_velocity ( c )
    var upc = c.up_cell

    -- interpolation weights
    var xw : L.vec2f
    var yw : L.vec2f
    if L.xid(c)%2 == 0 then xw = {1.0f,0.0f} else xw = {0.5f,0.5f} end
    if L.yid(c)%2 == 0 then yw = {1.0f,0.0f} else yw = {0.5f,0.5f} end

    var velocity =
        xw[0]*yw[0]*upc(0,0).velocity + xw[1]*yw[0]*upc(1,0).velocity +
        xw[0]*yw[1]*upc(0,1).velocity + xw[1]*yw[1]*upc(1,1).velocity

    c.velocity = velocity
end

-----------------------------------------------------------------------------
--[[                             UPDATES                                 ]]--
-----------------------------------------------------------------------------

grid.cells:NewField('vel_shadow', L.vec2f):Load({0,0})
local ebb neumann_shadow_update (c)
        if c.xneg_depth > 0 then
        var v = c(1,0).velocity
        c.vel_shadow = { -v[0],  v[1] }
    elseif c.xpos_depth > 0 then
        var v = c(-1,0).velocity
        c.vel_shadow = { -v[0],  v[1] }
    elseif c.yneg_depth > 0 then
        var v = c(0,1).velocity
        c.vel_shadow = {  v[0], -v[1] }
    elseif c.ypos_depth > 0 then
        var v = c(0,-1).velocity
        c.vel_shadow = {  v[0], -v[1] }
    end
end
local ebb neumann_cpy_update (c)
    c.velocity = c.vel_shadow
end
local function vel_neumann_bnd(cells)
    cells.boundary:foreach(neumann_shadow_update)
    cells.boundary:foreach(neumann_cpy_update)
end

-----------------------------------------------------------------------------
--[[                             DIFFUSE                                 ]]--
-----------------------------------------------------------------------------

local diffuse_diagonal = L.Global(L.float, 0.0)
local diffuse_edge     = L.Global(L.float, 0.0)

-- One Jacobi-Iteration
local ebb diffuse_lin_solve_jacobi_step ( c )
    var edge_sum = diffuse_edge * ( c(-1,0).velocity + c(1,0).velocity +
                                    c(0,-1).velocity + c(0,1).velocity )
    c.vel_shadow = (c.velocity_prev - edge_sum) / diffuse_diagonal
end

print(pyramid:nLevels())

local function diffuse_multi_solve(edge, diagonal)
    diffuse_diagonal:set(diagonal)
    diffuse_edge:set(edge)

    local Nlevels = pyramid:nLevels()

    -- Base level, 2 iterations
    --for k=1,2 do
    --    grid.cells.interior:foreach(diffuse_lin_solve_jacobi_step)
    --    grid.cells:Swap('velocity','vel_shadow')
    --end
    -- Up steps, 2 iterations each
    for i=1,Nlevels do
        local mgrid = pyramid:level(i)
        mgrid.cells.interior:foreach(lift_up_velocity)
        --for k=1,2 do
        --    mgrid.cells:foreach(diffuse_lin_solve_jacobi_step)
        --    mgrid.cells:Swap('velocity','vel_shadow')
        --end
    end
    -- Down steps, 4 iterations each
    for i=1,Nlevels-1 do
        local mgrid = pyramid:level(Nlevels-i)
        mgrid.cells:foreach(pull_down_velocity)
        for k=1,i*3 do
            mgrid.cells:foreach(diffuse_lin_solve_jacobi_step)
            mgrid.cells:Swap('velocity','vel_shadow')
        end
    end
    -- Base level, 4 iterations
    grid.cells.interior:foreach(pull_down_velocity)
    for k=1,400 do
        grid.cells:foreach(diffuse_lin_solve_jacobi_step)
        grid.cells:Swap('velocity','vel_shadow')
    end
end

-- Should be called with velocity and velocity_prev both set to
-- the previous velocity field value...
local function diffuse_lin_solve(edge, diagonal)
    diffuse_diagonal:set(diagonal)
    diffuse_edge:set(edge)

    local domain = grid.cells.interior
    if PERIODIC then domain = grid.cells end

    -- do 20 Jacobi iterations
    for i=1,400 do
        domain:foreach(diffuse_lin_solve_jacobi_step)
        grid.cells:Swap('velocity','vel_shadow')
        if not PERIODIC then vel_neumann_bnd(grid.cells) end
    end
end

local function diffuse_velocity(grid)
    -- Why the N*N term?  I don't get that...
    local laplacian_weight  = dt:get() * viscosity * N * N
    local diagonal          = 1.0 + 4.0 * laplacian_weight
    local edge              = -laplacian_weight

    grid.cells:Copy{from='velocity',to='velocity_prev'}
    diffuse_lin_solve(edge, diagonal)
    --diffuse_multi_solve(edge, diagonal)
end

-----------------------------------------------------------------------------
--[[                             ADVECT                                  ]]--
-----------------------------------------------------------------------------

local cell_w = grid:xCellWidth()
local cell_h = grid:yCellWidth()

local advect_dt = L.Global(L.float, 0.0)
grid.cells:NewField('lookup_pos', L.vec2f):Load({0,0})
grid.cells:NewField('lookup_from', grid.dual_cells):Load({0,0})

local epsilon = 1.0e-5 * math.max(cell_w, cell_h)
local min_x = grid:xOrigin() + cell_w/2 + epsilon
local max_x = grid:xOrigin() + grid:xWidth() - cell_w/2 - epsilon
local min_y = grid:yOrigin() + cell_h/2 + epsilon
local max_y = grid:yOrigin() + grid:yWidth() - cell_h/2 - epsilon
local snap_to_grid = ebb(p)
    var pxy : L.vec2f = p
    if      pxy[0] < min_x then pxy[0] = L.float(min_x)
    elseif  pxy[0] > max_x then pxy[0] = L.float(max_x) end
    if      pxy[1] < min_y then pxy[1] = L.float(min_y)
    elseif  pxy[1] > max_y then pxy[1] = L.float(max_y) end
    return  pxy
end
if PERIODIC then
    min_x = grid:xOrigin()
    max_x = grid:xOrigin() + grid:xWidth()
    min_y = grid:yOrigin()
    max_y = grid:yOrigin() + grid:yWidth()
    local d_x = grid:xWidth()
    local d_y = grid:yWidth()
    local ebb wrap_func(val, lower, upper)
        var diff    = upper-lower
        var temp    = val - lower
        temp        = L.float(cmath.fmod(temp, diff))
        if temp < 0 then
            temp    = temp+diff
        end
        return temp + lower
    end
    snap_to_grid = ebb(p)
        var pxy : L.vec2f = p
        pxy[0] = L.float(wrap_func(pxy[0], min_x, max_x))
        pxy[1] = L.float(wrap_func(pxy[1], min_y, max_y))
        return pxy
    end
end

local ebb advect_where_from(c : grid.cells)
    var offset      = - c.velocity_prev
    -- Make sure all our lookups are appropriately confined
    c.lookup_pos    = snap_to_grid(c.center + advect_dt * offset)
end

local ebb advect_interpolate_velocity(c : grid.cells)
    -- lookup cell (this is the bottom left corner)
    var dc      = c.lookup_from

    -- figure out fractional position in the dual cell in range [0.0, 1.0]
    var xfrac   = cmath.fmod((c.lookup_pos[0] - XORIGIN)/cell_w + 0.5, 1.0)
    var yfrac   = cmath.fmod((c.lookup_pos[1] - YORIGIN)/cell_h + 0.5, 1.0)

    -- interpolation constants
    var x1      = L.float(xfrac)
    var y1      = L.float(yfrac)
    var x0      = L.float(1.0 - xfrac)
    var y0      = L.float(1.0 - yfrac)

    -- velocity interpolation
    var lc      = dc.vertex.cell(-1,-1)
    c.velocity  = x0 * y0 * lc(0,0).velocity_prev
                + x1 * y0 * lc(1,0).velocity_prev
                + x0 * y1 * lc(0,1).velocity_prev
                + x1 * y1 * lc(1,1).velocity_prev
end

local function advect_velocity(grid)
    -- Why N?
    advect_dt:set(dt:get() * N)

    grid.cells:Swap('velocity','velocity_prev')
    grid.cells:foreach(advect_where_from)
    grid.locate_in_duals(grid.cells, 'lookup_pos', 'lookup_from')
    if PERIODIC then
        grid.cells:foreach(advect_interpolate_velocity)
    else
        grid.cells.interior:foreach(advect_interpolate_velocity)
        vel_neumann_bnd(grid.cells)
    end
end

-----------------------------------------------------------------------------
--[[                             PROJECT                                 ]]--
-----------------------------------------------------------------------------

local project_diagonal = L.Global(L.float, 0.0)
local project_edge     = L.Global(L.float, 0.0)
grid.cells:NewField('divergence', L.float):Load(0)
grid.cells:NewField('p', L.float):Load(0)
grid.cells:NewField('p_temp', L.float):Load(0)

local ebb project_lin_solve_jacobi_step (c : grid.cells)
    var edge_sum = project_edge * ( c(-1,0).p + c(1,0).p +
                                    c(0,-1).p + c(0,1).p )
    c.p_temp = (c.divergence - edge_sum) / project_diagonal
end

-- Neumann condition
local ebb pressure_shadow_update (c : grid.cells)
        if c.xneg_depth > 0 then
        c.p_temp = c(1,0).p
    elseif c.xpos_depth > 0 then
        c.p_temp = c(-1,0).p
    elseif c.yneg_depth > 0 then
        c.p_temp = c(0,1).p
    elseif c.ypos_depth > 0 then
        c.p_temp = c(0,-1).p
    end
end
local ebb pressure_cpy_update (c : grid.cells)
    c.p = c.p_temp
end
local function pressure_neumann_bnd(cells)
    cells.boundary:foreach(pressure_shadow_update)
    cells.boundary:foreach(pressure_cpy_update)
end


local function project_lin_solve(edge, diagonal)
    project_diagonal:set(diagonal)
    project_edge:set(edge)

    local domain = grid.cells.interior
    if PERIODIC then domain = grid.cells end

    -- do 20 Jacobi iterations
    for i=1,20 do
        domain:foreach(project_lin_solve_jacobi_step)
        grid.cells:Swap('p','p_temp')
        if not PERIODIC then
            pressure_neumann_bnd(grid.cells)
        end
    end
end

local ebb compute_divergence (c : grid.cells)
    -- why the factor of N?
    var vx_dx = c(1,0).velocity[0] - c(-1,0).velocity[0]
    var vy_dy = c(0,1).velocity[1] - c(0,-1).velocity[1]
    c.divergence = L.float(-(0.5/N)*(vx_dx + vy_dy))
end

local ebb compute_projection (c : grid.cells)
    var grad = L.vec2f(0.5 * N * { c(1,0).p - c(-1,0).p,
                                   c(0,1).p - c(0,-1).p })
    c.velocity = c.velocity_prev - grad
end

local function project_velocity(grid)
    local diagonal          =  4.0
    local edge              = -1.0

    local domain = grid.cells.interior
    if PERIODIC then domain = grid.cells end

    domain:foreach(compute_divergence)
    if PERIODIC then
        grid.cells:Copy{from='divergence', to='p'}
    else
        grid.cells:Swap('divergence','p') -- move divergence into p to do bnd
        pressure_neumann_bnd(grid.cells)
        grid.cells:Copy{from='p',to='divergence'} -- then copy it back
    end

    project_lin_solve(edge, diagonal)

    grid.cells:Swap('velocity','velocity_prev')
    domain:foreach(compute_projection)

    if not PERIODIC then
        vel_neumann_bnd(grid.cells)
    end
end


-----------------------------------------------------------------------------
--[[                            PARTICLES                                ]]--
-----------------------------------------------------------------------------

local PARTICLE_LEN = N - (PERIODIC and 0 or 1)
local N_particles = PARTICLE_LEN * PARTICLE_LEN
local mode = 'PLAIN'
if INSERT_DELETE then mode = 'ELASTIC' end
local particles = L.NewRelation {
    mode = mode,
    size = N_particles,
    name = 'particles',
}

particles:NewField('dual_cell', grid.dual_cells):Load(function(i)
    local xid = math.floor(i%PARTICLE_LEN)
    local yid = math.floor(i/PARTICLE_LEN)
    if PERIODIC then    return {xid,yid}
    else                return {(xid+1),(yid+1)}
    end
end)

particles:NewField('next_pos', L.vec2f):Load({0,0})
particles:NewField('pos', L.vec2f):Load({0,0})
particles:foreach(ebb (p : particles) -- init...
    p.pos = p.dual_cell.vertex.cell(-1,-1).center +
            L.vec2f({cell_w/2.0, cell_h/2.0})
end)

local ebb locate_particles (p : particles)
    p.dual_cell = grid.dual_locate(p.pos)
end

local ebb compute_particle_velocity (p : particles)
    -- lookup cell (this is the bottom left corner)
    var dc      = p.dual_cell

    -- figure out fractional position in the dual cell in range [0.0, 1.0]
    var xfrac   = cmath.fmod((p.pos[0] - XORIGIN)/cell_w + 0.5, 1.0)
    var yfrac   = cmath.fmod((p.pos[1] - YORIGIN)/cell_h + 0.5, 1.0)

    -- interpolation constants
    var x1      = L.float(xfrac)
    var y1      = L.float(yfrac)
    var x0      = L.float(1.0 - xfrac)
    var y0      = L.float(1.0 - yfrac)

    -- velocity interpolation
    var lc = dc.vertex.cell(-1,-1)
    p.next_pos  = p.pos + N *
        ( x0 * y0 * lc(0,0).velocity_prev
        + x1 * y0 * lc(1,0).velocity_prev
        + x0 * y1 * lc(0,1).velocity_prev
        + x1 * y1 * lc(1,1).velocity_prev )
end

local ebb particle_snap(p, pos)
    p.pos = snap_to_grid(pos)
end
if PERIODIC and INSERT_DELETE then
    min_x = grid:xOrigin()
    max_x = grid:xOrigin() + grid:xWidth()
    min_y = grid:yOrigin()
    max_y = grid:yOrigin() + grid:yWidth()

    particle_snap = ebb(p, pos)
        p.pos = pos
        if pos[0] > max_x or pos[0] < min_x or
           pos[1] > max_y or pos[1] < min_y then
            delete p
        end
    end
end

local ebb update_particle_pos (p : particles)
    var r = L.vec2f({ cmath.rand_float() - 0.5, cmath.rand_float() - 0.5 })
    var pos = p.next_pos + L.float(dt) * r
    particle_snap(p, pos)
end


-----------------------------------------------------------------------------
--[[                             MAIN LOOP                               ]]--
-----------------------------------------------------------------------------

--grid.cells:print()

local source_strength = L.Constant(L.float, 100.0)
local source_velocity = ebb (c : grid.cells)
    if cmath.fabs(c.center[0]) < 1.75 and
       cmath.fabs(c.center[1]) < 1.75
    then
        if not c.in_boundary then
            c.velocity += L.float(dt) * source_strength * { 0.0f,  1.0f }
        else
            c.velocity += L.float(dt) * source_strength * { 0.0f, -1.0f }
        end
    end
end
if PERIODIC then
    source_strength = L.Constant(L.float, 5.0)
    local optional_insertion = ebb(c) end -- no-op
    if INSERT_DELETE then
        optional_insertion = ebb(c)
            var create_particle = cmath.rand_float() < 0.01f
            if create_particle then
                var offset = L.vec2f({
                    cell_w * (rand_float() - 0.5),
                    cell_h * (rand_float() - 0.5)
                })
                var pos = c.center + offset
                insert {
                    dual_cell = c.vertex.dual_cell, -- to be overwritten soon
                    pos = pos,
                    next_pos = pos
                } into particles
            end
        end
    end
    source_velocity = ebb (c : grid.cells)
        if cmath.fabs(c.center[0]) < 1.75 and
           cmath.fabs(c.center[1]) < 1.75
        then
            c.velocity += L.float(dt) * source_strength * { 0.0f, 1.0f }
        end

        optional_insertion(c)
    end
end

local ebb draw_grid (c : grid.cells)
    var color = {1.0, 1.0, 1.0}
    vdb.color(color)
    var p : L.vec3f = { c.center[0],   c.center[1],   0.0f }
    var vel = c.velocity
    var v = L.vec3f({ vel[0], vel[1], 0.0f })
    vdb.line(p, p+v*N*10)
end

local ebb draw_particles (p : particles)
    var color = {1.0f,1.0f,0.0f}
    vdb.color(color)
    var pos : L.vec3f = { p.pos[0], p.pos[1], 0.0f }
    vdb.point(pos)
end

local STEPS = 500 -- use 500 on my local machine
for i = 0, STEPS-1 do
    if math.floor(i / 70) % 2 == 0 then
        grid.cells:foreach(source_velocity)
    end

    diffuse_velocity(grid)
    project_velocity(grid)

    advect_velocity(grid)
    project_velocity(grid)

    grid.locate_in_duals(particles, 'pos', 'dual_cell')
    particles:foreach(compute_particle_velocity)
    particles:foreach(update_particle_pos)

    if i % 4 == 0 then
        vdb.vbegin()
            vdb.frame()
            --grid.cells:foreach(draw_grid)
            particles:foreach(draw_particles)
        vdb.vend()
        --print(particles:ConcreteSize(), particles:Size())
    end
    --print('push key')
    --io.read()
end

--grid.cells:print()

