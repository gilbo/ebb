import "compiler.liszt"
--L.default_processor = L.GPU

local Grid  = L.require 'domains.grid'
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
--cmath.srand(cmath.time(nil));
--local rand_float = cmath.rand_float
local rand_float = L.rand
--if L.default_processor == L.GPU then
--    rand_float = terra() return 0.5f end
--end
local vdb   = L.require 'lib.vdb'

local N = 150
local PERIODIC = true
local INSERT_DELETE = true -- exercise these features in periodic mode...
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
--[[                             UPDATES                                 ]]--
-----------------------------------------------------------------------------

grid.cells:NewField('vel_shadow', L.vec2f):Load({0,0})
local liszt neumann_shadow_update (c : grid.cells)
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
local liszt neumann_cpy_update (c : grid.cells)
    c.velocity = c.vel_shadow
end
local function vel_neumann_bnd(cells)
    cells.boundary:foreach(neumann_shadow_update)
    cells.boundary:foreach(neumann_cpy_update)
end

-----------------------------------------------------------------------------
--[[                             VELSTEP                                 ]]--
-----------------------------------------------------------------------------

-----------------------------------------------------------------------------
--[[                             DIFFUSE                                 ]]--
-----------------------------------------------------------------------------

local diffuse_diagonal = L.Global(L.float, 0.0)
local diffuse_edge     = L.Global(L.float, 0.0)

-- One Jacobi-Iteration
local liszt diffuse_lin_solve_jacobi_step (c : grid.cells)
    var edge_sum = diffuse_edge * ( c(-1,0).velocity + c(1,0).velocity +
                                    c(0,-1).velocity + c(0,1).velocity )
    c.vel_shadow = (c.velocity_prev - edge_sum) / diffuse_diagonal
end

-- Should be called with velocity and velocity_prev both set to
-- the previous velocity field value...
local function diffuse_lin_solve(edge, diagonal)
    diffuse_diagonal:set(diagonal)
    diffuse_edge:set(edge)

    local domain = grid.cells.interior
    if PERIODIC then domain = grid.cells end

    -- do 20 Jacobi iterations
    for i=1,20 do
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
local snap_to_grid = liszt(p)
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
    local liszt wrap_func(val, lower, upper)
        var diff    = upper-lower
        var temp    = val - lower
        temp        = L.float(L.fmod(temp, diff))
        if temp < 0 then
            temp    = temp+diff
        end
        return temp + lower
    end
    snap_to_grid = liszt(p)
        var pxy : L.vec2f = p
        pxy[0] = L.float(wrap_func(pxy[0], min_x, max_x))
        pxy[1] = L.float(wrap_func(pxy[1], min_y, max_y))
        return pxy
    end
end

local liszt advect_where_from(c : grid.cells)
    var offset      = - c.velocity_prev
    -- Make sure all our lookups are appropriately confined
    c.lookup_pos    = snap_to_grid(c.center + advect_dt * offset)
end

local liszt advect_point_locate(c : grid.cells)
    c.lookup_from   = grid.dual_locate(c.lookup_pos)
end

local liszt advect_interpolate_velocity(c : grid.cells)
    -- lookup cell (this is the bottom left corner)
    var dc      = c.lookup_from

    -- figure out fractional position in the dual cell in range [0.0, 1.0]
    var xfrac   = L.fmod((c.lookup_pos[0] - XORIGIN)/cell_w + 0.5, 1.0)
    var yfrac   = L.fmod((c.lookup_pos[1] - YORIGIN)/cell_h + 0.5, 1.0)

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
    grid.cells:foreach(advect_point_locate)
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

local liszt project_lin_solve_jacobi_step (c : grid.cells)
    var edge_sum = project_edge * ( c(-1,0).p + c(1,0).p +
                                    c(0,-1).p + c(0,1).p )
    c.p_temp = (c.divergence - edge_sum) / project_diagonal
end

-- Neumann condition
local liszt pressure_shadow_update (c : grid.cells)
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
local liszt pressure_cpy_update (c : grid.cells)
    c.p = c.p_temp
end
local function pressure_neumann_bnd(cells)
    cells.boundary:foreach(pressure_shadow_update)
    cells.boundary:foreach(pressure_cpy_update)
end


-- Should be called with velocity and velocity_prev both set to
-- the previous velocity field value...
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

local liszt compute_divergence (c : grid.cells)
    -- why the factor of N?
    var vx_dx = c(1,0).velocity[0] - c(-1,0).velocity[0]
    var vy_dy = c(0,1).velocity[1] - c(0,-1).velocity[1]
    c.divergence = L.float(-(0.5/N)*(vx_dx + vy_dy))
end

local liszt compute_projection (c : grid.cells)
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
        grid.cells:foreach(pressure_neumann_bnd)
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
particles:foreach(liszt (p : particles) -- init...
    p.pos = p.dual_cell.vertex.cell(-1,-1).center +
            L.vec2f({cell_w/2.0, cell_h/2.0})
end)

local liszt locate_particles (p : particles)
    p.dual_cell = grid.dual_locate(p.pos)
end

local liszt compute_particle_velocity (p : particles)
    -- lookup cell (this is the bottom left corner)
    var dc      = p.dual_cell

    -- figure out fractional position in the dual cell in range [0.0, 1.0]
    var xfrac   = L.fmod((p.pos[0] - XORIGIN)/cell_w + 0.5, 1.0)
    var yfrac   = L.fmod((p.pos[1] - YORIGIN)/cell_h + 0.5, 1.0)

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

local liszt particle_snap(p, pos)
    p.pos = snap_to_grid(pos)
end
if PERIODIC and INSERT_DELETE then
    min_x = grid:xOrigin()
    max_x = grid:xOrigin() + grid:xWidth()
    min_y = grid:yOrigin()
    max_y = grid:yOrigin() + grid:yWidth()

    particle_snap = liszt(p, pos)
        p.pos = pos
        if pos[0] > max_x or pos[0] < min_x or
           pos[1] > max_y or pos[1] < min_y then
            delete p
        end
    end
end

local liszt update_particle_pos (p : particles)
    var r = L.vec2f({ rand_float() - 0.5, rand_float() - 0.5 })
    var pos = p.next_pos + L.float(dt) * r
    particle_snap(p, pos)
end


-----------------------------------------------------------------------------
--[[                             MAIN LOOP                               ]]--
-----------------------------------------------------------------------------

--grid.cells:print()

local source_strength = L.Constant(L.float, 100.0)
local source_velocity = liszt (c : grid.cells)
    if L.fabs(c.center[0]) < 1.75 and
       L.fabs(c.center[1]) < 1.75
    then
        if not c.in_boundary then
            c.velocity += L.float(dt) * source_strength * { 0.0f, 1.0f }
        else
            c.velocity += L.float(dt) * source_strength * { 0.0f, -1.0f }
        end
    end
end
if PERIODIC then
    source_strength = L.Constant(L.float, 5.0)
    local optional_insertion = liszt(c) end -- no-op
    if INSERT_DELETE then
        optional_insertion = liszt(c)
            var create_particle = rand_float() < 0.01
            if create_particle then
                var pos = c.center + L.vec2f({
                    cell_w * (rand_float() - 0.5),
                    cell_h * (rand_float() - 0.5)
                })
                insert {
                    dual_cell = grid.dual_locate(pos),
                    pos = pos,
                    next_pos = pos
                } into particles
            end
        end
    end
    source_velocity = liszt (c : grid.cells)
        if L.fabs(c.center[0]) < 1.75 and
           L.fabs(c.center[1]) < 1.75
        then
            c.velocity += L.float(dt) * source_strength * { 0.0f, 1.0f }
        end

        optional_insertion(c)
    end
end

local liszt draw_grid (c : grid.cells)
    var color = {1.0, 1.0, 1.0}
    vdb.color(color)
    var p : L.vec3f = { c.center[0],   c.center[1],   0.0f }
    var vel = c.velocity
    var v = L.vec3f({ vel[0], vel[1], 0.0f })
    vdb.line(p, p+v*N*10)
end

local liszt draw_particles (p : particles)
    var color = {1.0,1.0,0.0}
    vdb.color(color)
    var pos : L.vec3f = { p.pos[0], p.pos[1], 0.0f }
    vdb.point(pos)
end

local STEPS = 500 -- use 500 on my local machine
for i = 0, STEPS-1 do
    print(i)
    if math.floor(i / 70) % 2 == 0 then
        grid.cells:foreach(source_velocity)
    end
    diffuse_velocity(grid)
    project_velocity(grid)

    advect_velocity(grid)
    project_velocity(grid)

    particles:foreach(compute_particle_velocity)
    particles:foreach(update_particle_pos)
    particles:foreach(locate_particles)

    if L.default_processor == L.CPU and i % 5 == 0 then
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

