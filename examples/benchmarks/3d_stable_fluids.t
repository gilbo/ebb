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

local N = 32
local PERIODIC = false
local period = {false,false,false}
local origin = {-N/2.0, -1.0, -N/2.0}
if PERIODIC then
    period = {true,true,true}
    origin = {-N/2, -N/2, -N/2}
end
local grid = Grid.NewGrid3d {
    size   = {N, N, N},
    origin = origin,
    width  = {N, N, N},
    periodic_boundary = period,
}

local XORIGIN       = origin[1]
local YORIGIN       = origin[2]
local ZORIGIN       = origin[3]
local viscosity     = 0.08
local dt            = L.Global(L.float, 0.01)

grid.cells:NewField('velocity', L.vec3f):Load({0,0,0})
grid.cells:NewField('velocity_prev', L.vec3f):Load({0,0,0})


-----------------------------------------------------------------------------
--[[                             UPDATES                                 ]]--
-----------------------------------------------------------------------------

grid.cells:NewField('vel_shadow', L.vec3f):Load({0,0,0})

local liszt neumann_shadow_update (c : grid.cells)
        if c.xneg_depth > 0 then
        var v = c(1,0,0).velocity
        c.vel_shadow = { -v[0],  v[1],  v[2] }
    elseif c.xpos_depth > 0 then
        var v = c(-1,0,0).velocity
        c.vel_shadow = { -v[0],  v[1],  v[2] }
    elseif c.yneg_depth > 0 then
        var v = c(0,1,0).velocity
        c.vel_shadow = {  v[0], -v[1],  v[2] }
    elseif c.ypos_depth > 0 then
        var v = c(0,-1,0).velocity
        c.vel_shadow = {  v[0], -v[1],  v[2] }
    elseif c.zneg_depth > 0 then
        var v = c(0,0,1).velocity
        c.vel_shadow = {  v[0],  v[1], -v[2] }
    elseif c.zpos_depth > 0 then
        var v = c(0,0,-1).velocity
        c.vel_shadow = {  v[0],  v[1], -v[2] }
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
--[[                             DIFFUSE                                 ]]--
-----------------------------------------------------------------------------

local diffuse_diagonal = L.Global(L.float, 0.0)
local diffuse_edge     = L.Global(L.float, 0.0)

-- One Jacobi-Iteration
local liszt diffuse_lin_solve_jacobi_step (c : grid.cells)
    var edge_sum = diffuse_edge * ( c(-1,0,0).velocity + c(1,0,0).velocity +
                                    c(0,-1,0).velocity + c(0,1,0).velocity +
                                    c(0,0,-1).velocity + c(0,0,1).velocity )
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
    local diagonal          = 1.0 + 6.0 * laplacian_weight
    local edge              = -laplacian_weight

    grid.cells:Copy{from='velocity',to='velocity_prev'}
    diffuse_lin_solve(edge, diagonal)
end

-----------------------------------------------------------------------------
--[[                             ADVECT                                  ]]--
-----------------------------------------------------------------------------

local xcwidth = grid:xCellWidth()
local ycwidth = grid:yCellWidth()
local zcwidth = grid:zCellWidth()

local advect_dt = L.Global(L.float, 0.0)
grid.cells:NewField('lookup_pos', L.vec3f):Load({0,0,0})
grid.cells:NewField('lookup_from', grid.dual_cells):Load({0,0,0})

local epsilon = 1.0e-5 * math.max(xcwidth, math.max(ycwidth, zcwidth))
local min_x = grid:xOrigin() + xcwidth/2 + epsilon
local max_x = grid:xOrigin() + grid:xWidth() - xcwidth/2 - epsilon
local min_y = grid:yOrigin() + ycwidth/2 + epsilon
local max_y = grid:yOrigin() + grid:yWidth() - ycwidth/2 - epsilon
local min_z = grid:zOrigin() + zcwidth/2 + epsilon
local max_z = grid:zOrigin() + grid:zWidth() - zcwidth/2 - epsilon
local liszt wrap(val, lower, upper)
    var diff : L.float = upper - lower
    var temp = L.float(cmath.fmod(val - lower, diff))
    if temp < 0 then
        temp += diff
    end
    return temp + lower
end
local liszt clamp(val, lower, upper)
    var result : L.float = val
    result max= L.float(lower)
    result min= L.float(upper)
    return result
end
local snap_to_grid = liszt(p)
    var pxyz : L.vec3f = p
    pxyz[0] = clamp(pxyz[0], min_x, max_x)
    pxyz[1] = clamp(pxyz[1], min_y, max_y)
    pxyz[2] = clamp(pxyz[2], min_z, max_z)
    return pxyz
end
if PERIODIC then
    min_x = grid:xOrigin()
    max_x = grid:xOrigin() + grid:xWidth()
    min_y = grid:yOrigin()
    max_y = grid:yOrigin() + grid:yWidth()
    min_z = grid:zOrigin()
    max_z = grid:zOrigin() + grid:zWidth()
    snap_to_grid = liszt(p)
        var pxyz : L.vec3f = p
        pxyz[0] = wrap(pxyz[0], min_x, max_x)
        pxyz[1] = wrap(pxyz[1], min_y, max_y)
        pxyz[2] = wrap(pxyz[2], min_z, max_z)
        return pxyz
    end
end

local liszt advect_where_from (c : grid.cells)
    var offset      = - c.velocity_prev
    -- Make sure all our lookups are appropriately confined
    c.lookup_pos    = snap_to_grid(c.center + advect_dt * offset)
end

local liszt advect_point_locate (c : grid.cells)
    c.lookup_from   = grid.dual_locate(c.lookup_pos)
end

local liszt advect_interpolate_velocity (c : grid.cells)
    var dc      = c.lookup_from

    -- figure out fractional position in the dual cell in range [0.0, 1.0]
    var xfrac   = cmath.fmod((c.lookup_pos[0] - XORIGIN)/xcwidth + 0.5, 1.0)
    var yfrac   = cmath.fmod((c.lookup_pos[1] - YORIGIN)/ycwidth + 0.5, 1.0)
    var zfrac   = cmath.fmod((c.lookup_pos[2] - ZORIGIN)/zcwidth + 0.5, 1.0)

    -- interpolation constants
    var xc = L.vec2f({ 1-xfrac, xfrac })
    var yc = L.vec2f({ 1-yfrac, yfrac })
    var zc = L.vec2f({ 1-zfrac, zfrac })
    --var x1      = L.float(xfrac)
    --var y1      = L.float(yfrac)
    --var z1      = L.float(zfrac)
    --var x0      = L.float(1.0 - xfrac)
    --var y0      = L.float(1.0 - yfrac)
    --var z0      = L.float(1.0 - zfrac)

    -- lookup cell
    var lc = dc.vertex.cell(-1,-1,-1)
    c.velocity = {0,0,0}
    for i=0,2 do for j=0,2 do for k=0,2 do
        c.velocity += xc[i] * yc[j] * zc[k] * lc(i,j,k).velocity_prev
    end end end
    --c.velocity  = x0 * y0 * lc(0,0,0).velocity_prev
    --            + x1 * y0 * lc(1,0,0).velocity_prev
    --            + x0 * y1 * lc(0,1,0).velocity_prev
    --            + x1 * y1 * lc(1,1,0).velocity_prev
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
    var edge_sum = project_edge * ( c(-1,0,0).p + c(1,0,0).p +
                                    c(0,-1,0).p + c(0,1,0).p +
                                    c(0,0,-1).p + c(0,0,1).p )
    c.p_temp = (c.divergence - edge_sum) / project_diagonal
end

-- Neumann condition
local liszt pressure_shadow_update (c : grid.cells)
        if c.xneg_depth > 0 then
        c.p_temp = c(1,0,0).p
    elseif c.xpos_depth > 0 then
        c.p_temp = c(-1,0,0).p
    elseif c.yneg_depth > 0 then
        c.p_temp = c(0,1,0).p
    elseif c.ypos_depth > 0 then
        c.p_temp = c(0,-1,0).p
    elseif c.zneg_depth > 0 then
        c.p_temp = c(0,0,1).p
    elseif c.zpos_depth > 0 then
        c.p_temp = c(0,0,-1).p
    end
end
local pressure_cpy_update = liszt (c : grid.cells)
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
        if not PERIODIC then pressure_neumann_bnd(grid.cells) end
    end
end

local liszt compute_divergence (c : grid.cells)
    -- why the factor of N?
    var vx_dx = c(1,0,0).velocity[0] - c(-1,0,0).velocity[0]
    var vy_dy = c(0,1,0).velocity[1] - c(0,-1,0).velocity[1]
    var vz_dz = c(0,0,1).velocity[2] - c(0,0,-1).velocity[2]
    c.divergence = L.float(-(0.5/N)*(vx_dx + vy_dy + vz_dz))
end

local liszt compute_projection (c : grid.cells)
    var grad = L.vec3f(0.5 * N * { c(1,0,0).p - c(-1,0,0).p,
                                   c(0,1,0).p - c(0,-1,0).p,
                                   c(0,0,1).p - c(0,0,-1).p })
    c.velocity = c.velocity_prev - grad
end

local function project_velocity(grid)
    local diagonal          =  6.0
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

    if not PERIODIC then vel_neumann_bnd(grid.cells) end
end


-----------------------------------------------------------------------------
--[[                            PARTICLES                                ]]--
-----------------------------------------------------------------------------

local PARTICLE_LEN = N - (PERIODIC and 0 or 1)
local PARTICLE_LEN = math.floor(N / 2)
local PARTICLE_OFF = math.floor((N-PARTICLE_LEN)/2)
local N_particles = PARTICLE_LEN * PARTICLE_LEN * PARTICLE_LEN
local particles = L.NewRelation {
    mode = 'ELASTIC',
    size = N_particles,
    name = 'particles',
}

particles:NewField('dual_cell', grid.dual_cells):Load(function(i)
    local xid = i%PARTICLE_LEN
    local topx = (i-xid)/PARTICLE_LEN
    local yid = topx%PARTICLE_LEN
    local topy = (topx-yid)/PARTICLE_LEN
    local zid = topy
    xid = xid + PARTICLE_OFF
    zid = zid + PARTICLE_OFF
    yid = yid + 2

    if PERIODIC then return {xid, yid, zid}
    else             return {xid+1, yid+1, zid+1}
    end
end)

particles:NewField('next_pos', L.vec3f):Load({0,0,0})
particles:NewField('pos', L.vec3f):Load({0,0,0})
particles:foreach(liszt (p : particles) -- init...
    p.pos = p.dual_cell.vertex.cell(-1,-1,-1).center +
            L.vec3f({xcwidth/2.0, ycwidth/2.0, zcwidth/2.0})
end)

local liszt locate_particles (p : particles)
    p.dual_cell = grid.dual_locate(p.pos)
end

local liszt compute_particle_velocity (p : particles)
    var dc      = p.dual_cell

    -- figure out fractional position in the dual cell in range [0.0, 1.0]
    var xfrac   = cmath.fmod((p.pos[0] - XORIGIN)/xcwidth + 0.5, 1.0)
    var yfrac   = cmath.fmod((p.pos[1] - YORIGIN)/ycwidth + 0.5, 1.0)
    var zfrac   = cmath.fmod((p.pos[2] - ZORIGIN)/zcwidth + 0.5, 1.0)

    -- interpolation constants
    var xc = L.vec2f({ 1-xfrac, xfrac })
    var yc = L.vec2f({ 1-yfrac, yfrac })
    var zc = L.vec2f({ 1-zfrac, zfrac })

    -- lookup cell
    var lc = dc.vertex.cell(-1,-1,-1)
    p.next_pos = p.pos
    for i=0,2 do for j=0,2 do for k=0,2 do
        p.next_pos += N * xc[i] * yc[j] * zc[k] * lc(i,j,k).velocity_prev
    end end end
end

local liszt update_particle_pos (p : particles)
    var r = L.vec3f({
        cmath.rand_float() - 0.5,
        cmath.rand_float() - 0.5,
        cmath.rand_float() - 0.5
    })
    var pos = p.next_pos + L.float(dt) * r
    p.pos = snap_to_grid(pos)
end


-----------------------------------------------------------------------------
--[[                             MAIN LOOP                               ]]--
-----------------------------------------------------------------------------

--grid.cells:print()

local source_strength = 100.0
local source_velocity = liszt (c : grid.cells)
    if cmath.fabs(c.center[0]) < 1.75 and
       cmath.fabs(c.center[1]) < 1.75 and
       cmath.fabs(c.center[2]) < 1.75
    then
        if not c.in_boundary then
            c.velocity += L.float(dt) * source_strength * { 0.0, 1.0, 0.0 }
        else
            c.velocity += L.float(dt) * source_strength * { 0.0, -1.0, 0.0 }
        end
    end
end
if PERIODIC then
    source_strength = 10.0
    source_velocity = liszt (c : grid.cells)
        if cmath.fabs(c.center[0]) < 1.75 and
           cmath.fabs(c.center[1]) < 1.75 and
           cmath.fabs(c.center[2]) < 1.75
        then
            c.velocity += L.float(dt) * source_strength * { 0.0, 1.0, 0.0 }
        end
    end
end

local liszt draw_grid (c : grid.cells)
    var color = {1.0, 1.0, 1.0}
    vdb.color(color)
    var p : L.vec3f = c.center
    var v : L.vec3f = c.velocity
    --if not c.in_boundary then
    vdb.line(p, p+v*N*10)
end

local liszt draw_particles (p : particles)
    var color = {1.0,1.0,0.0}
    vdb.color(color)
    var pos : L.vec3f = p.pos
    vdb.point(pos)
end

local STEPS = 500 -- use 500 on my local machine
for i = 1, STEPS do
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


    --if i % 5 == 0 then
        vdb.vbegin()
            vdb.frame()
            --grid.cells:foreach(draw_grid)
            particles:foreach(draw_particles)
        vdb.vend()
    --end
    --print('push key')
    --io.read()
end

--grid.cells:print()

