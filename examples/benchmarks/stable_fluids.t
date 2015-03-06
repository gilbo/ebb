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
cmath.srand(cmath.time(nil));
local vdb   = L.require 'lib.vdb'

local N = 150
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
--[[                             UPDATES                                 ]]--
-----------------------------------------------------------------------------

grid.cells:NewField('vel_shadow', L.vec2f):Load({0,0})
local neumann_shadow_update = liszt kernel (c : grid.cells)
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
local neumann_cpy_update = liszt kernel (c : grid.cells)
    c.velocity = c.vel_shadow
end
local function vel_neumann_bnd(cells)
    neumann_shadow_update(cells.boundary)
    neumann_cpy_update(cells.boundary)
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
local diffuse_lin_solve_jacobi_step = liszt kernel (c : grid.cells)
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
        diffuse_lin_solve_jacobi_step(domain)
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
local snap_to_grid = liszt function(p)
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
    local wrap_func = liszt function(val, lower, upper)
        var diff    = upper-lower
        var temp    = val - lower
        temp        = L.float(cmath.fmod(temp, diff))
        if temp < 0 then
            temp    = temp+diff
        end
        return temp + lower
    end
    snap_to_grid = liszt function(p)
        var pxy : L.vec2f = p
        pxy[0] = L.float(wrap_func(pxy[0], min_x, max_x))
        pxy[1] = L.float(wrap_func(pxy[1], min_y, max_y))
        return pxy
    end
end

local advect_where_from = liszt kernel(c : grid.cells)
    var offset      = - c.velocity_prev
    -- Make sure all our lookups are appropriately confined
    c.lookup_pos    = snap_to_grid(c.center + advect_dt * offset)
end

local advect_point_locate = liszt kernel(c : grid.cells)
    c.lookup_from   = grid.dual_locate(c.lookup_pos)
end

local advect_interpolate_velocity = liszt kernel(c : grid.cells)
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
    advect_where_from(grid.cells)
    advect_point_locate(grid.cells)
    if PERIODIC then
        advect_interpolate_velocity(grid.cells)
    else
        advect_interpolate_velocity(grid.cells.interior)
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

local project_lin_solve_jacobi_step = liszt kernel (c : grid.cells)
    var edge_sum = project_edge * ( c(-1,0).p + c(1,0).p +
                                    c(0,-1).p + c(0,1).p )
    c.p_temp = (c.divergence - edge_sum) / project_diagonal
end

-- Neumann condition
local pressure_shadow_update = liszt kernel (c : grid.cells)
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
local pressure_cpy_update = liszt kernel (c : grid.cells)
    c.p = c.p_temp
end
local function pressure_neumann_bnd(cells)
    pressure_shadow_update(cells.boundary)
    pressure_cpy_update(cells.boundary)
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
        project_lin_solve_jacobi_step(domain)
        grid.cells:Swap('p','p_temp')
        if not PERIODIC then
            pressure_neumann_bnd(grid.cells)
        end
    end
end

local compute_divergence = liszt kernel (c : grid.cells)
    -- why the factor of N?
    var vx_dx = c(1,0).velocity[0] - c(-1,0).velocity[0]
    var vy_dy = c(0,1).velocity[1] - c(0,-1).velocity[1]
    c.divergence = L.float(-(0.5/N)*(vx_dx + vy_dy))
end

local compute_projection = liszt kernel (c : grid.cells)
    var grad = L.vec2f(0.5 * N * { c(1,0).p - c(-1,0).p,
                                   c(0,1).p - c(0,-1).p })
    c.velocity = c.velocity_prev - grad
end

local function project_velocity(grid)
    local diagonal          =  4.0
    local edge              = -1.0

    local domain = grid.cells.interior
    if PERIODIC then domain = grid.cells end

    compute_divergence(domain)
    if PERIODIC then
        grid.cells:Copy{from='divergence', to='p'}
    else
        grid.cells:Swap('divergence','p') -- move divergence into p to do bnd
        pressure_neumann_bnd(grid.cells)
        grid.cells:Copy{from='p',to='divergence'} -- then copy it back
    end

    project_lin_solve(edge, diagonal)

    grid.cells:Swap('velocity','velocity_prev')
    compute_projection(domain)

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
(liszt kernel (p : particles) -- init...
    p.pos = p.dual_cell.vertex.cell(-1,-1).center +
            L.vec2f({cell_w/2.0, cell_h/2.0})
end)(particles)

local locate_particles = liszt kernel (p : particles)
    p.dual_cell = grid.dual_locate(p.pos)
end

local compute_particle_velocity = liszt kernel (p : particles)
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

local particle_snap = liszt function(p, pos)
    p.pos = snap_to_grid(pos)
end
if PERIODIC and INSERT_DELETE then
    min_x = grid:xOrigin()
    max_x = grid:xOrigin() + grid:xWidth()
    min_y = grid:yOrigin()
    max_y = grid:yOrigin() + grid:yWidth()

    particle_snap = liszt function(p, pos)
        p.pos = pos
        if pos[0] > max_x or pos[0] < min_x or
           pos[1] > max_y or pos[1] < min_y then
            delete p
        end
    end
end

local update_particle_pos = liszt kernel (p : particles)
    var r = L.vec2f({ cmath.rand_float() - 0.5, cmath.rand_float() - 0.5 })
    var pos = p.next_pos + L.float(dt) * r
    particle_snap(p, pos)
end


-----------------------------------------------------------------------------
--[[                             MAIN LOOP                               ]]--
-----------------------------------------------------------------------------

--grid.cells:print()

local source_strength = 100.0
local source_velocity = liszt kernel (c : grid.cells)
    if cmath.fabs(c.center[0]) < 1.75 and
       cmath.fabs(c.center[1]) < 1.75
    then
        if not c.in_boundary then
            c.velocity += L.float(dt) * source_strength * { 0.0, 1.0 }
        else
            c.velocity += L.float(dt) * source_strength * { 0.0, -1.0 }
        end
    end
end
if PERIODIC then
    source_strength = 5.0
    local optional_insertion = liszt function(c) end -- no-op
    if INSERT_DELETE then
        optional_insertion = liszt function(c)
            var create_particle = cmath.rand_float() < 0.01
            if create_particle then
                var pos = c.center + L.vec2f({
                    cell_w * (cmath.rand_float() - 0.5),
                    cell_h * (cmath.rand_float() - 0.5)
                })
                insert {
                    dual_cell = grid.dual_locate(pos),
                    pos = pos,
                    next_pos = pos
                } into particles
            end
        end
    end
    source_velocity = liszt kernel (c : grid.cells)
        if cmath.fabs(c.center[0]) < 1.75 and
           cmath.fabs(c.center[1]) < 1.75
        then
            c.velocity += L.float(dt) * source_strength * { 0.0, 1.0 }
        end

        optional_insertion(c)
    end
end

local draw_grid = liszt kernel (c : grid.cells)
    var color = {1.0, 1.0, 1.0}
    vdb.color(color)
    var p : L.vec3f = { c.center[0],   c.center[1],   0.0 }
    var vel = c.velocity
    var v = L.vec3f({ vel[0], vel[1], 0.0 })
    vdb.line(p, p+v*N*10)
end

local draw_particles = liszt kernel (p : particles)
    var color = {1.0,1.0,0.0}
    vdb.color(color)
    var pos : L.vec3f = { p.pos[0], p.pos[1], 0.0 }
    vdb.point(pos)
end

local STEPS = 500 -- use 500 on my local machine
for i = 0, STEPS-1 do
    if math.floor(i / 70) % 2 == 0 then
        source_velocity(grid.cells)
    end

    diffuse_velocity(grid)
    project_velocity(grid)

    advect_velocity(grid)
    project_velocity(grid)

    compute_particle_velocity(particles)
    update_particle_pos(particles)
    locate_particles(particles)


    if i % 5 == 0 then
        vdb.vbegin()
            vdb.frame()
            --draw_grid(grid.cells)
            draw_particles(particles)
        vdb.vend()
        --print(particles:ConcreteSize(), particles:Size())
    end
    --print('push key')
    --io.read()
end

--grid.cells:print()

