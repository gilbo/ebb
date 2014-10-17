import 'compiler.liszt'

--
-- Constants
--

local PN    = L.require 'lib.pathname'
local Grid  = L.require 'domains.grid'

local cmath = terralib.includecstring [[
#include <math.h>
#include <stdlib.h>
#include <time.h>

int nancheck(float val) {
    return isnan(val);
}

float randFloat()
{
      float r = (float)rand() / (float)RAND_MAX;
      return r;
}
]]

local N = 5
local dt = 0.09
local visc = 0.0025
local origin = {0.0, 0.0}
local XORIGIN       = origin[1]
local YORIGIN       = origin[2]

local grid = Grid.NewGrid2d {
    size   = {N, N},
    origin = origin,
    width  = {N, N},
}

local min_x = grid:xOrigin()
local max_x = grid:xOrigin() + grid:xWidth()
local min_y = grid:yOrigin()
local max_y = grid:yOrigin() + grid:yWidth()
local cell_w = grid:xCellWidth()
local cell_h = grid:yCellWidth()

--
-- Fields
--
grid.cells:NewField('coord', L.vec2i):Load(L.NewVector(L.int, {0,0}))
grid.cells:NewField('dv', L.vec2f):Load(L.NewVector(L.float, {0,0}))
grid.cells:NewField('vx', L.vec2f):Load(L.NewVector(L.float, {0,0}))
grid.cells:NewField('vy', L.vec2f):Load(L.NewVector(L.float, {0,0}))
grid.cells:NewField('advectPos', L.vec2f):Load(L.NewVector(L.float, {0,0}))
grid.cells:NewField('advectFrom', grid.dual_cells):Load(0)

--
-- Helper functions
--

local wrapFunc = liszt function(val, lower, upper)
    var diff    = upper - lower
    var temp    = val - lower
    temp        = L.float(cmath.fmod(temp, diff))
    if temp < 0 then
        temp    = temp + diff
    end
    return temp + lower
end

local snapToGrid = liszt function(p)
    var pxy : L.vec2f
    pxy[0] = L.float(wrapFunc(p[0], min_x, max_x))
    pxy[1] = L.float(wrapFunc(p[1], min_y, max_y))
    return pxy
end

--
-- Advection kernels
--

local advectWhereFrom = liszt kernel(c : grid.cells)
    c.advectPos = snapToGrid(c.center + dt * N * -c.dv)
end

local advectPointLocate = liszt kernel(c : grid.cells)
    c.advectFrom = grid.dual_locate(c.advectPos)
end

local advectInterpolateVelocity = liszt kernel(c : grid.cells)
    -- lookup cell (this is the bottom left corner)
    var dc      = c.advectFrom

    -- figure out fractional position in the dual cell in range [0.0, 1.0]
    var xfrac   = cmath.fmod((c.advectPos[0] - XORIGIN)/cell_w + 0.5, 1.0)
    var yfrac   = cmath.fmod((c.advectPos[1] - YORIGIN)/cell_h + 0.5, 1.0)

    -- interpolation constants
    var x1      = L.float(xfrac)
    var y1      = L.float(yfrac)
    var x0      = L.float(1.0 - xfrac)
    var y0      = L.float(1.0 - yfrac)

    -- velocity interpolation
    var lc      = dc.vertex.cell(-1,-1)
    var velocity = x0 * y0 * lc(0,0).dv
                 + x1 * y0 * lc(1,0).dv
                 + x0 * y1 * lc(0,1).dv
                 + x1 * y1 * lc(1,1).dv
	
	c.vx[0] = velocity[0]
	c.vx[1] = 0.0

	c.vy[0] = velocity[1]
	c.vy[1] = 0.0
end

local function advectVelocity(grid)
	advectWhereFrom(grid.cells)
    advectPointLocate(grid.cells)
    advectInterpolateVelocity(grid.cells)
end

--
-- Diffusion kernel
--

local diffuseProject = liszt kernel(c : grid.cells)
	var xIndex = c.coord[0]
	var yIndex = c.coord[1]
	var xFFT = c.vx
    var yFFT = c.vy

    -- Compute the index of the wavenumber based on the
    -- data order produced by a standard NN FFT.
    var iix = xIndex
    var iiy = yIndex
	if yIndex > N / 2 then
		iiy = yIndex - N
	end
	
    -- Velocity diffusion
    var kk = L.float(iix * iix + iiy * iiy)
    var diff = L.float(1.0f / (1.0f + visc * dt * kk))
    xFFT[0] *= diff
    xFFT[1] *= diff
    yFFT[0] *= diff
    yFFT[1] *= diff

    -- Velocity projection
    if kk > 0.0f then
        var rkk = 1.f / kk
        -- Real portion of velocity projection
        var rkp = (iix * xFFT[0] + iiy * yFFT[0])
        -- Imaginary portion of velocity projection
        var ikp = (iix * xFFT[1] + iiy * yFFT[1])
        xFFT[0] -= rkk * rkp * iix
        xFFT[1] -= rkk * ikp * iix
        yFFT[0] -= rkk * rkp * iiy
        yFFT[1] -= rkk * ikp * iiy
    end

    c.vx = xFFT
    c.vy = yFFT
end

local updateVelocity = liszt kernel(c : grid.cells)
    var scale = 1.0f / (N * N)
    c.dv[0] = c.vx[0] * scale
	c.dv[1] = c.vy[0] * scale
end

--
-- Particles
--

local N_particles = N * N
local particles = L.NewRelation(N_particles, 'particles')

particles:NewField('dual_cell', grid.dual_cells):Load(function(i)
    local xid = math.floor(i%N)
    local yid = math.floor(i/N)
    return (xid+1) + (N+1)*(yid+1)
end)

particles:NewField('nextPos', L.vec2f):Load(L.NewVector(L.float, {0,0}))
particles:NewField('pos', L.vec2f):Load(L.NewVector(L.float, {0,0}))
(liszt kernel (p : particles) -- init...
    p.pos = p.dual_cell.vertex.cell(-1,-1).center +
            L.vec2f({cell_w/2.0, cell_h/2.0})
end)(particles)

local locateParticles = liszt kernel (p : particles)
    p.dual_cell = grid.dual_locate(p.pos)
end

local computeParticleVelocity = liszt kernel (p : particles)
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
    p.nextPos  = p.pos + N *
        ( x0 * y0 * lc(0,0).dv
        + x1 * y0 * lc(1,0).dv
        + x0 * y1 * lc(0,1).dv
        + x1 * y1 * lc(1,1).dv )
end

local updateParticlePos = liszt kernel (p : particles)
    var r = L.vec2f({ cmath.randFloat() - 0.5, cmath.randFloat() - 0.5 })
    var pos = p.nextPos + L.float(dt) * r
end

for i = 1, 1000 do
	advectVelocity(grid)
    --diffuseProject(grid)
    --updateVelocity(grid)
    
	computeParticleVelocity(particles)
    updateParticlePos(particles)
    locateParticles(particles)
end

grid.cells:print()

