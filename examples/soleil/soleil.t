import "compiler.liszt"
local Grid  = L.require 'domains.grid'
local cmath = terralib.includecstring [[
#include <math.h>
#include <stdlib.h>
#include <time.h>

double rand_double() {
      double r = (double)rand();
      return r;
}

double rand_unity() {
    double r = (double)rand()/(double)RAND_MAX;
    return r;
}

]]

cmath.srand(cmath.time(nil));
local vdb   = L.require 'lib.vdb'

-----------------------------------------------------------------------------
--[[                            CONSTANT VARIABLES                       ]]--
-----------------------------------------------------------------------------

local pi = 2.0*cmath.acos(0)
local twoPi = 2.0*pi

-----------------------------------------------------------------------------
--[[                            NAMESPACES                               ]]--
-----------------------------------------------------------------------------

local Flow = {};
local Particles = {};
local TimeIntegrator = {};
local Statistics = {};
local IO = {};

-----------------------------------------------------------------------------
--[[                             OPTIONS                                 ]]--
-----------------------------------------------------------------------------

local grid_options = {
    xnum = 64,
    ynum = 64,
    origin = {0.0, 0.0},
    width = twoPi,
    height = twoPi
}

local spatial_stencil = {
--  Splitting parameter (for skew
    split = 0.5,
----  Order 2
--    order = 2,
--    size = 2,
--    numInterpolateCoeffs = 2,
--    interpolateCoeffs = L.NewVector(L.double, {0, 0.5}),
--    numFirstDerivativeCoeffs = 2,
--    firstDerivativeCoeffs = L.NewVector(L.double, {0, 0.5}),
--    firstDerivativeModifiedWaveNumber = 1.0,
--    secondDerivativeModifiedWaveNumber = 4.0,
----  Order 6
    order = 6,
    size = 6,
    numInterpolateCoeffs = 4,
    interpolateCoeffs = L.NewVector(L.double, {0, 37/60, -8/60, 1/60}),
    numFirstDerivativeCoeffs = 4,
    firstDerivativeCoeffs = L.NewVector(L.double, {0.0,45.0/60.0,-9.0/60.0, 1.0/60.0}),
    firstDerivativeModifiedWaveNumber = 1.59,
    secondDerivativeModifiedWaveNumber = 6.04
}

-- Define offsets for boundary conditions in flow solver
-- Periodic
local xoffsetLeft  = grid_options.xnum
local xoffsetRight = grid_options.xnum
local yoffsetDown  = grid_options.ynum
local yoffsetUp    = grid_options.ynum
local signDouble   = 1

-- Time integrator
TimeIntegrator.coeff_function       = {1/6, 1/3, 1/3, 1/6}
TimeIntegrator.coeff_time           = {0.5, 0.5, 1, 1}
TimeIntegrator.simTime              = 0
TimeIntegrator.final_time           = 100.00001
TimeIntegrator.timeStep             = 0
TimeIntegrator.cfl                  = 1.2
TimeIntegrator.outputEveryTimeSteps = 63
TimeIntegrator.deltaTime            = L.NewGlobal(L.double, 0.01)

local fluid_options = {
    gasConstant = 20.4128,
    gamma = 1.4,
    dynamic_viscosity_ref = 0.0044,
    dynamic_viscosity_temp_ref = 1.0,
    prandtl = 0.7
}

local flow_options = {
    bodyForce = L.NewGlobal(L.vec2d, {0,-0.001})
}

local particles_options = {
    num = 1000,
    convective_coefficient = L.NewGlobal(L.double, 0.7), -- W m^-2 K^-1
    heat_capacity = L.NewGlobal(L.double, 0.7), -- J Kg^-1 K^-1
    pos_max = 6.2,
    initialTemperature = 20,
    density = 1000,
    diameter_m = 0.01,
    diameter_a = 0.001,
    bodyForce = L.NewGlobal(L.vec2d, {0,-0.001}),
    emissivity = 0.5,
    absorptivity = 0.5 -- Equal to emissivity in thermal equilibrium
                       -- (Kirchhoff law of thermal radiation)
}

local radiation_options = {
    radiationIntensity = 1000.0
}

-- IO
IO.outputFileNamePrefix = "../soleilOutput/output"

-----------------------------------------------------------------------------
--[[                       FLOW/ PARTICLE RELATIONS                      ]]--
-----------------------------------------------------------------------------


-- Declare and initialize grid and related fields

local bnum = spatial_stencil.order/2
local bw   = grid_options.width/grid_options.xnum * bnum
local gridOriginX = grid_options.origin[1]
local gridOriginY = grid_options.origin[2]
local originWithGhosts = grid_options.origin
originWithGhosts[1] = originWithGhosts[1] - 
                      bnum * grid_options.width/grid_options.xnum
originWithGhosts[2] = originWithGhosts[2] - 
                      bnum * grid_options.height/grid_options.xnum
print(originWithGhosts[1],originWithGhosts[2])

local grid = Grid.NewGrid2d{size           = {grid_options.xnum + 2*bnum,
                                              grid_options.ynum + 2*bnum},
                            origin         = originWithGhosts,
                            width          = grid_options.width + 2*bw,
                            height         = grid_options.height + 2*bw,
                            boundary_depth = {bnum, bnum} }

-- Conserved variables
grid.cells:NewField('rho', L.double):
LoadConstant(0)
grid.cells:NewField('rhoVelocity', L.vec2d):
LoadConstant(L.NewVector(L.double, {0, 0}))
grid.cells:NewField('rhoEnergy', L.double):
LoadConstant(0)

-- Primitive variables
grid.cells:NewField('centerCoordinates', L.vec2d):
LoadConstant(L.NewVector(L.double, {0, 0}))
grid.cells:NewField('velocity', L.vec2d):
LoadConstant(L.NewVector(L.double, {0, 0}))
grid.cells:NewField('velocityGradientX', L.vec2d):
LoadConstant(L.NewVector(L.double, {0, 0}))
grid.cells:NewField('velocityGradientY', L.vec2d):
LoadConstant(L.NewVector(L.double, {0, 0}))
grid.cells:NewField('temperature', L.double):
LoadConstant(0)
grid.cells:NewField('pressure', L.double):
LoadConstant(0)
grid.cells:NewField('rhoEnthalpy', L.double):
LoadConstant(0)
grid.cells:NewField('kineticEnergy', L.double):
LoadConstant(0)
grid.cells:NewField('sgsEnergy', L.double):
LoadConstant(0)
grid.cells:NewField('sgsEddyViscosity', L.double):
LoadConstant(0)
grid.cells:NewField('sgsEddyKappa', L.double):
LoadConstant(0)
grid.cells:NewField('convectiveSpectralRadius', L.double):
LoadConstant(0)
grid.cells:NewField('viscousSpectralRadius', L.double):
LoadConstant(0)
grid.cells:NewField('heatConductionSpectralRadius', L.double):
LoadConstant(0)

-- Fields for boundary treatment
grid.cells:NewField('rhoBoundary', L.double):
LoadConstant(0)
grid.cells:NewField('rhoVelocityBoundary', L.vec2d):
LoadConstant(L.NewVector(L.double, {0, 0}))
grid.cells:NewField('rhoEnergyBoundary', L.double):
LoadConstant(0)
grid.cells:NewField('velocityBoundary', L.vec2d):
LoadConstant(L.NewVector(L.double, {0, 0}))
grid.cells:NewField('pressureBoundary', L.double):
LoadConstant(0)
grid.cells:NewField('temperatureBoundary', L.double):
LoadConstant(0)
grid.cells:NewField('velocityGradientXBoundary', L.vec2d):
LoadConstant(L.NewVector(L.double, {0, 0}))
grid.cells:NewField('velocityGradientYBoundary', L.vec2d):
LoadConstant(L.NewVector(L.double, {0, 0}))

-- scratch (temporary) fields
-- intermediate value and copies
grid.cells:NewField('rho_old', L.double):
LoadConstant(0)
grid.cells:NewField('rhoVelocity_old', L.vec2d):
LoadConstant(L.NewVector(L.double, {0, 0}))
grid.cells:NewField('rhoEnergy_old', L.double):
LoadConstant(0)
grid.cells:NewField('rho_new', L.double):
LoadConstant(0)
grid.cells:NewField('rhoVelocity_new', L.vec2d):
LoadConstant(L.NewVector(L.double, {0, 0}))
grid.cells:NewField('rhoEnergy_new', L.double):
LoadConstant(0)
-- time derivatives
grid.cells:NewField('rho_t', L.double):
LoadConstant(0)
grid.cells:NewField('rhoVelocity_t', L.vec2d):
LoadConstant(L.NewVector(L.double, {0, 0}))
grid.cells:NewField('rhoEnergy_t', L.double):
LoadConstant(0)
-- fluxes
grid.cells:NewField('rhoFlux', L.double):
LoadConstant(0)
grid.cells:NewField('rhoVelocityFlux', L.vec2d):
LoadConstant(L.NewVector(L.double, {0, 0}))
grid.cells:NewField('rhoEnergyFlux', L.double):
LoadConstant(0)


-- Declare and initialize particle relation and fields over the particle

local particles = L.NewRelation(particles_options.num, 'particles')

particles:NewField('dual_cell', grid.dual_cells):
LoadConstant(0)
particles:NewField('position', L.vec2d):
Load(function(i)
    local pmax = particles_options.pos_max
    local p1 = cmath.fmod(cmath.rand_double(), pmax)
    local p2 = cmath.fmod(cmath.rand_double(), pmax)
    return {p1, p2}
end)
particles:NewField('velocity', L.vec2d):
LoadConstant(L.NewVector(L.double, {0, 0}))
particles:NewField('temperature', L.double):
LoadConstant(particles_options.initialTemperature)
particles:NewField('position_periodic', L.vec2d):
LoadConstant(L.NewVector(L.double, {0, 0}))

particles:NewField('diameter', L.double):
Load(function(i)
    return cmath.rand_unity() * particles_options.diameter_m +
        particles_options.diameter_a
end)
particles:NewField('density', L.double):
LoadConstant(particles_options.density)

particles:NewField('deltaVelocityOverRelaxationTime', L.vec2d):
LoadConstant(L.NewVector(L.double, {0, 0}))
particles:NewField('deltaTemperatureTerm', L.double):
LoadConstant(0)

-- scratch (temporary) fields
-- intermediate values and copies
particles:NewField('position_old', L.vec2d):
LoadConstant(L.NewVector(L.double, {0, 0}))
particles:NewField('velocity_old', L.vec2d):
LoadConstant(L.NewVector(L.double, {0, 0}))
particles:NewField('temperature_old', L.double):
LoadConstant(0)
particles:NewField('position_new', L.vec2d):
LoadConstant(L.NewVector(L.double, {0, 0}))
particles:NewField('velocity_new', L.vec2d):
LoadConstant(L.NewVector(L.double, {0, 0}))
particles:NewField('temperature_new', L.double):
LoadConstant(0)
-- derivatives
particles:NewField('position_t', L.vec2d):
LoadConstant(L.NewVector(L.double, {0, 0}))
particles:NewField('velocity_t', L.vec2d):
LoadConstant(L.NewVector(L.double, {0, 0}))
particles:NewField('temperature_t', L.double):
LoadConstant(0)

-----------------------------------------------------------------------------
--[[                       USER DEFINED FUNCTIONS                        ]]--
-----------------------------------------------------------------------------

-- Norm of a vector
local norm = liszt function(v)
    return cmath.sqrt(L.dot(v, v))
end

-- Compute fluid dynamic viscosity from fluid temperature
local GetDynamicViscosity = liszt function(temperature)
    return fluid_options.dynamic_viscosity_ref *
        cmath.pow(temperature/fluid_options.dynamic_viscosity_temp_ref, 0.75)
end

-- Compute fluid flow sound speed based on temperature
local GetSoundSpeed = liszt function(temperature)
    return cmath.sqrt(fluid_options.gamma * 
                      fluid_options.gasConstant *
                      temperature)
end

-- Function to retrieve particle area, volume and mass
-- These are Liszt user-defined function that behave like a field
particles:NewFieldFunction('area', liszt function(p)
    return pi * cmath.pow(p.diameter, 2)
end)
particles:NewFieldFunction('volume', liszt function(p)
    return pi * cmath.pow(p.diameter, 3) / 6.0
end)
particles:NewFieldFunction('mass', liszt function(p)
    return p.volume * p.density
end)

------------------------------------------------------------------------------
--[[                             LISZT MACROS                            ]]--
-----------------------------------------------------------------------------


-- Functions for calling inside liszt kernel

local Rho = L.NewMacro(function(r)
    return liszt `r.rho
end)

local Velocity = L.NewMacro(function(r)
    return liszt `r.velocity
end)

local Temperature = L.NewMacro(function(r)
    return liszt `r.temperature
end)

local InterpolateBilinear = L.NewMacro(function(dc, xy, Field)
    return liszt quote
        var cdl = dc.vertex.cell(-1,-1)
        var cul = dc.vertex.cell(-1,0)
        var cdr = dc.vertex.cell(0,-1)
        var cur = dc.vertex.cell(0,0)
        var delta_l = xy[1] - cdl.center[1]
        var delta_r = cur.center[1] - xy[1]
        var f1 = (delta_l*Field(cdl) + delta_r*Field(cul)) / (delta_l + delta_r)
        var f2 = (delta_l*Field(cdr) + delta_r*Field(cur)) / (delta_l + delta_r)
        var delta_d = xy[0] - cdl.center[0]
        var delta_u = cur.center[0] - xy[0]
    in
        (delta_d*f1 + delta_u*f2) / (delta_d + delta_u)
    end
end)

-----------------------------------------------------------------------------
--[[                            LISZT KERNELS                            ]]--
-----------------------------------------------------------------------------

-- Locate particles
Particles.Locate = liszt kernel(p : particles)
    p.dual_cell = grid.dual_locate(p.position)
end


-- Initialize flow variables
-- Cell center coordinates are stored in the grid field macro 'center'. 
-- Here, we use a field for convenience when outputting to file, but this is
-- to be removed after grid outputing is well defined from within the grid.t 
-- module
Flow.InitializeCenterCoordinates = liszt kernel(c : grid.cells)
    var xy = c.center
    c.centerCoordinates = L.vec2d({xy[0], xy[1]})
end
Flow.InitializePrimitives = liszt kernel(c : grid.cells)
    -- Define Taylor Green Vortex
    var taylorGreenPressure = 100.0
    -- Initialize
    var xy = c.center
    var coorZ = 0
    c.rho = 1.0
    c.velocity = 
        L.vec2d({cmath.sin(xy[0]) * 
                 cmath.cos(xy[1]) *
                 cmath.cos(coorZ),
               - cmath.cos(xy[0]) *
                 cmath.sin(xy[1]) *
                 cmath.cos(coorZ)})
    var factorA = cmath.cos(2.0*coorZ) + 2.0
    var factorB = cmath.cos(2.0*xy[0]) +
                  cmath.cos(2.0*xy[1])
    c.pressure = 
        taylorGreenPressure + (factorA*factorB - 2.0) / 16.0
end
Flow.UpdateConservedFromPrimitive = liszt kernel(c : grid.cells)
    -- Equation of state: T = p / ( R * rho )
    var tmpTemperature = c.pressure /(fluid_options.gasConstant * c.rho)
    var velocity = c.velocity

    c.rhoVelocity = c.rho * c.velocity

 
    -- rhoE = rhoe (= rho * cv * T) + kineticEnergy + sgsEnergy
    var cv = fluid_options.gasConstant / 
             (fluid_options.gamma - 1.0)
    c.rhoEnergy = 
      c.rho *
      ( cv * tmpTemperature 
        + 0.5 * L.dot(velocity,velocity) )
      + c.sgsEnergy

end

-- Initialize temporaries
Flow.InitializeTemporaries = liszt kernel(c : grid.cells)
    c.rho_old         = c.rho
    c.rhoVelocity_old = c.rhoVelocity
    c.rhoEnergy_old   = c.rhoEnergy
    c.rho_new         = c.rho
    c.rhoVelocity_new = c.rhoVelocity
    c.rhoEnergy_new   = c.rhoEnergy
end
Particles.InitializeTemporaries = liszt kernel(p : particles)
    p.position_old    = p.position
    p.velocity_old    = p.velocity
    p.temperature_old = p.temperature
    p.position_new    = p.position
    p.velocity_new    = p.velocity
    p.temperature_new = p.temperature
end


-- Initialize derivatives
Flow.InitializeTimeDerivatives = liszt kernel(c : grid.cells)
    c.rho_t = L.double(0)
    c.rhoVelocity_t = L.vec2d({0, 0})
    c.rhoEnergy_t = L.double(0)
end
Particles.InitializeTimeDerivatives = liszt kernel(p : particles)
    p.position_t = L.vec2d({0, 0})
    p.velocity_t = L.vec2d({0, 0})
    p.temperature_t = L.double(0)
end

-----------
-- Inviscid
-----------

-- Initialize enthalpy and derivatives
Flow.AddInviscidInitialize = liszt kernel(c : grid.cells)
    c.rhoEnthalpy = c.rhoEnergy + c.pressure
    --L.print(c.rho, c.rhoEnergy, c.pressure, c.rhoEnthalpy)
end

-- Compute inviscid fluxes in X direction
Flow.AddInviscidGetFluxX =  liszt kernel(c : grid.cells)
    if c.in_interior or c.xneg_depth > 0 then
        var directionIdx = 0
        var numInterpolateCoeffs  = spatial_stencil.numInterpolateCoeffs
        var interpolateCoeffs     = spatial_stencil.interpolateCoeffs
        var numFirstDerivativeCoeffs = spatial_stencil.numFirstDerivativeCoeffs
        var firstDerivativeCoeffs    = spatial_stencil.firstDerivativeCoeffs

        -- Diagonal terms
        var rhoFactorDiagonal = L.double(0)
        var rhoVelocityFactorDiagonal = L.vec2d({0.0, 0.0})
        var rhoEnergyFactorDiagonal   = L.double(0.0)
        var fpdiag = L.double(0.0)
        for ndx = 1, numInterpolateCoeffs do
            rhoFactorDiagonal += interpolateCoeffs[ndx] *
                          ( c(1-ndx,0).rho *
                            c(1-ndx,0).velocity[directionIdx] +
                            c(ndx,0).rho *
                            c(ndx,0).velocity[directionIdx] )
            rhoVelocityFactorDiagonal += interpolateCoeffs[ndx] *
                                   ( c(1-ndx,0).rhoVelocity *
                                     c(1-ndx,0).velocity[directionIdx] +
                                     c(ndx,0).rhoVelocity *
                                     c(ndx,0).velocity[directionIdx] )
            rhoEnergyFactorDiagonal += interpolateCoeffs[ndx] *
                                 ( c(1-ndx,0).rhoEnthalpy *
                                   c(1-ndx,0).velocity[directionIdx] +
                                   c(ndx,0).rhoEnthalpy *
                                   c(ndx,0).velocity[directionIdx] )
            fpdiag += interpolateCoeffs[ndx] *
                    ( c(1-ndx,0).pressure +
                      c(ndx,0).pressure )
        end

        -- Skewed terms
        var rhoFactorSkew         = L.double(0)
        var rhoVelocityFactorSkew = L.vec2d({0.0, 0.0})
        var rhoEnergyFactorSkew   = L.double(0.0)
        -- mdx = -N+1,...,0
        for mdx = 2-numFirstDerivativeCoeffs, 1 do
          var tmp = L.double(0)
          for ndx = 1, mdx+numFirstDerivativeCoeffs do
            tmp += firstDerivativeCoeffs[ndx-mdx] * 
                   c(ndx,0).velocity[directionIdx]
          end

          rhoFactorSkew += c(mdx,0).rho * tmp
          rhoVelocityFactorSkew += c(mdx,0).rhoVelocity * tmp
          rhoEnergyFactorSkew += c(mdx,0).rhoEnthalpy * tmp
        end
        --  mdx = 1,...,N
        for mdx = 1,numFirstDerivativeCoeffs do
          var tmp = L.double(0)
          for ndx = mdx-numFirstDerivativeCoeffs+1, 1 do
            tmp += firstDerivativeCoeffs[mdx-ndx] * 
                   c(ndx,0).velocity[directionIdx]
          end

          rhoFactorSkew += c(mdx,0).rho * tmp
          rhoVelocityFactorSkew += c(mdx,0).rhoVelocity * tmp
          rhoEnergyFactorSkew += c(mdx,0).rhoEnthalpy * tmp
        end

        var s = spatial_stencil.split
        c.rhoFlux          = s * rhoFactorDiagonal +
                             (1-s) * rhoFactorSkew
        c.rhoVelocityFlux  = s * rhoVelocityFactorDiagonal +
                             (1-s) * rhoVelocityFactorSkew
        c.rhoEnergyFlux    = s * rhoEnergyFactorDiagonal +
                             (1-s) * rhoEnergyFactorSkew
        c.rhoVelocityFlux[directionIdx] += fpdiag
    end
end

-- Compute inviscid fluxes in Y direction
Flow.AddInviscidGetFluxY =  liszt kernel(c : grid.cells)
    if c.in_interior or c.yneg_depth > 0 then
        var directionIdx = 1
        var numInterpolateCoeffs  = spatial_stencil.numInterpolateCoeffs
        var interpolateCoeffs     = spatial_stencil.interpolateCoeffs
        var numFirstDerivativeCoeffs = spatial_stencil.numFirstDerivativeCoeffs
        var firstDerivativeCoeffs    = spatial_stencil.firstDerivativeCoeffs
        var rhoFactorDiagonal = L.double(0)

        -- Diagonal terms
        var rhoVelocityFactorDiagonal = L.vec2d({0.0, 0.0})
        var rhoEnergyFactorDiagonal   = L.double(0.0)
        var fpdiag = L.double(0.0)
        for ndx = 1, numInterpolateCoeffs do
            rhoFactorDiagonal += interpolateCoeffs[ndx] *
                          ( c(0,1-ndx).rho *
                            c(0,1-ndx).velocity[directionIdx] +
                            c(0,ndx).rho *
                            c(0,ndx).velocity[directionIdx] )
            rhoVelocityFactorDiagonal += interpolateCoeffs[ndx] *
                                   ( c(0,1-ndx).rhoVelocity *
                                     c(0,1-ndx).velocity[directionIdx] +
                                     c(0,ndx).rhoVelocity *
                                     c(0,ndx).velocity[directionIdx] )
            rhoEnergyFactorDiagonal += interpolateCoeffs[ndx] *
                                 ( c(0,1-ndx).rhoEnthalpy *
                                   c(0,1-ndx).velocity[directionIdx] +
                                   c(0,ndx).rhoEnthalpy *
                                   c(0,ndx).velocity[directionIdx] )
            fpdiag += interpolateCoeffs[ndx] *
                    ( c(0,1-ndx).pressure +
                      c(0,ndx).pressure )
        end

        -- Skewed terms
        var rhoFactorSkew     = L.double(0)
        var rhoVelocityFactorSkew     = L.vec2d({0.0, 0.0})
        var rhoEnergyFactorSkew       = L.double(0.0)
        -- mdx = -N+1,...,0
        for mdx = 2-numFirstDerivativeCoeffs, 1 do
          var tmp = L.double(0)
          for ndx = 1, mdx+numFirstDerivativeCoeffs do
            tmp += firstDerivativeCoeffs[ndx-mdx] * 
                   c(0,ndx).velocity[directionIdx]
          end

          rhoFactorSkew += c(0,mdx).rho * tmp
          rhoVelocityFactorSkew += c(0,mdx).rhoVelocity * tmp
          rhoEnergyFactorSkew += c(0,mdx).rhoEnthalpy * tmp
        end
        --  mdx = 1,...,N
        for mdx = 1,numFirstDerivativeCoeffs do
          var tmp = L.double(0)
          for ndx = mdx-numFirstDerivativeCoeffs+1, 1 do
            tmp += firstDerivativeCoeffs[mdx-ndx] * 
                   c(0,ndx).velocity[directionIdx]
          end

          rhoFactorSkew += c(0,mdx).rho * tmp
          rhoVelocityFactorSkew += c(0,mdx).rhoVelocity * tmp
          rhoEnergyFactorSkew += c(0,mdx).rhoEnthalpy * tmp
        end

        var s = spatial_stencil.split
        c.rhoFlux          = s * rhoFactorDiagonal +
                             (1-s) * rhoFactorSkew
        c.rhoVelocityFlux  = s * rhoVelocityFactorDiagonal +
                             (1-s) * rhoVelocityFactorSkew
        c.rhoEnergyFlux    = s * rhoEnergyFactorDiagonal +
                             (1-s) * rhoEnergyFactorSkew
        c.rhoVelocityFlux[directionIdx]  += fpdiag
    end
end

-- Update conserved variables using flux values from previous part
-- write conserved variables, read flux variables
-- WARNING_START For non-uniform grids, the metrics used below are not 
-- appropriate and should be changed to reflect those expressed in the 
-- Python prototype code
local c_dx = L.NewGlobal(L.double, grid:cellWidth())
local c_dy = L.NewGlobal(L.double, grid:cellHeight())
-- WARNING_END
Flow.AddInviscidUpdateUsingFluxX = liszt kernel(c : grid.cells)
    if c.in_interior then
        c.rho_t -= (c(0,0).rhoFlux -
                    c(-1,0).rhoFlux)/c_dx
        c.rhoVelocity_t -= (c(0,0).rhoVelocityFlux -
                            c(-1,0).rhoVelocityFlux)/c_dx
        c.rhoEnergy_t -= (c(0,0).rhoEnergyFlux -
                          c(-1,0).rhoEnergyFlux)/c_dx
    end
end
Flow.AddInviscidUpdateUsingFluxY = liszt kernel(c : grid.cells)
    if c.in_interior then
        c.rho_t -= (c(0,0).rhoFlux -
                    c(0,-1).rhoFlux)/c_dx
        c.rhoVelocity_t -= (c(0,0).rhoVelocityFlux -
                            c(0,-1).rhoVelocityFlux)/c_dy
        c.rhoEnergy_t -= (c(0,0).rhoEnergyFlux -
                          c(0,-1).rhoEnergyFlux)/c_dy
    end
end

----------
-- Viscous
----------

-- Compute viscous fluxes in X direction
Flow.AddViscousGetFluxX =  liszt kernel(c : grid.cells)
    if c.in_interior or c.xneg_depth > 0 then
        var muFace = 0.5 * (GetDynamicViscosity(c(0,0).temperature) +
                            GetDynamicViscosity(c(1,0).temperature))
        var velocityFace    = L.vec2d({0.0, 0.0})
        var velocityX_YFace = L.double(0)
        var velocityX_ZFace = L.double(0)
        var velocityY_YFace = L.double(0)
        var velocityZ_ZFace = L.double(0)
        var numInterpolateCoeffs  = spatial_stencil.numInterpolateCoeffs
        var interpolateCoeffs     = spatial_stencil.interpolateCoeffs
        -- Interpolate velocity and derivatives to face
        for ndx = 1, numInterpolateCoeffs do
            velocityFace += interpolateCoeffs[ndx] *
                          ( c(1-ndx,0).velocity +
                            c(ndx,0).velocity )
            velocityX_YFace += interpolateCoeffs[ndx] *
                               ( c(1-ndx,0).velocityGradientY[0] +
                                 c(ndx,0).velocityGradientY[0] )
            velocityX_ZFace += 0.0 -- WARNING: to be updated for 3D (see Python code)
            velocityY_YFace += interpolateCoeffs[ndx] *
                               ( c(1-ndx,0).velocityGradientY[1] +
                                 c(ndx,0).velocityGradientY[1] )
            velocityZ_ZFace += 0.0 -- WARNING: to be updated for 3D (see Python code)
        end

        -- Differentiate at face
        var velocityX_XFace = L.double(0)
        var velocityY_XFace = L.double(0)
        var velocityZ_XFace = L.double(0)
        var temperature_XFace = L.double(0)
        var numFirstDerivativeCoeffs = spatial_stencil.numFirstDerivativeCoeffs
        var firstDerivativeCoeffs    = spatial_stencil.firstDerivativeCoeffs
        for ndx = 1, numFirstDerivativeCoeffs do
          velocityX_XFace += firstDerivativeCoeffs[ndx] *
            ( c(ndx,0).velocity[0] -
              c(1-ndx,0).velocity[0] )
          velocityY_XFace += firstDerivativeCoeffs[ndx] *
            ( c(ndx,0).velocity[1] -
              c(1-ndx,0).velocity[1] )
          velocityZ_XFace += 0.0
          temperature_XFace += firstDerivativeCoeffs[ndx] *
            ( c(ndx,0).temperature -
              c(1-ndx,0).temperature )
        end
       
        velocityX_XFace   /= c_dx
        velocityY_XFace   /= c_dx
        velocityZ_XFace   /= c_dx
        temperature_XFace /= c_dx

        -- Tensor components (at face)
        var sigmaXX = muFace * ( 4.0 * velocityX_XFace -
                                 2.0 * velocityY_YFace -
                                 2.0 * velocityZ_ZFace ) / 3.0
        var sigmaYX = muFace * ( velocityY_XFace + velocityX_YFace )
        var sigmaZX = muFace * ( velocityZ_XFace + velocityX_ZFace )
        var usigma = velocityFace[0] * sigmaXX +
                     velocityFace[1] * sigmaYX
        -- WARNING : Add term velocityFace[2] * sigmaZX to usigma for 3D
        var cp = fluid_options.gamma * fluid_options.gasConstant / 
                 (fluid_options.gamma - 1.0)
        var heatFlux = - cp / fluid_options.prandtl * 
                         muFace * temperature_XFace

        -- Fluxes
        c.rhoVelocityFlux[0] = sigmaXX
        c.rhoVelocityFlux[1] = sigmaYX
        -- WARNING: Uncomment for 3D rhoVelocityFlux[2] = sigmaZX
        c.rhoEnergyFlux = usigma - heatFlux
        -- WARNING: Add SGS terms for LES

    end
end

-- Compute viscous fluxes in Y direction
Flow.AddViscousGetFluxY =  liszt kernel(c : grid.cells)
    if c.in_interior or c.yneg_depth > 0 then
        var muFace = 0.5 * (GetDynamicViscosity(c(0,0).temperature) +
                            GetDynamicViscosity(c(0,1).temperature))
        var velocityFace    = L.vec2d({0.0, 0.0})
        var velocityY_XFace = L.double(0)
        var velocityY_ZFace = L.double(0)
        var velocityX_XFace = L.double(0)
        var velocityZ_ZFace = L.double(0)
        var numInterpolateCoeffs  = spatial_stencil.numInterpolateCoeffs
        var interpolateCoeffs     = spatial_stencil.interpolateCoeffs
        -- Interpolate velocity and derivatives to face
        for ndx = 1, numInterpolateCoeffs do
            velocityFace += interpolateCoeffs[ndx] *
                          ( c(0,1-ndx).velocity +
                            c(0,ndx).velocity )
            velocityY_XFace += interpolateCoeffs[ndx] *
                               ( c(0,1-ndx).velocityGradientX[1] +
                                 c(0,ndx).velocityGradientX[1] )
            velocityY_ZFace += 0.0 -- WARNING: to be updated for 3D (see Python code)
            velocityX_XFace += interpolateCoeffs[ndx] *
                               ( c(0,1-ndx).velocityGradientX[0] +
                                 c(0,ndx).velocityGradientX[0] )
            velocityZ_ZFace += 0.0 -- WARNING: to be updated for 3D (see Python code)
        end

        -- Differentiate at face
        var velocityX_YFace = L.double(0)
        var velocityY_YFace = L.double(0)
        var velocityZ_YFace = L.double(0)
        var temperature_YFace = L.double(0)
        var numFirstDerivativeCoeffs = spatial_stencil.numFirstDerivativeCoeffs
        var firstDerivativeCoeffs    = spatial_stencil.firstDerivativeCoeffs
        for ndx = 1, numFirstDerivativeCoeffs do
          velocityX_YFace += firstDerivativeCoeffs[ndx] *
            ( c(0,ndx).velocity[0] -
              c(0,1-ndx).velocity[0] )
          velocityY_YFace += firstDerivativeCoeffs[ndx] *
            ( c(0,ndx).velocity[1] -
              c(0,1-ndx).velocity[1] )
          velocityZ_YFace += 0.0
          temperature_YFace += firstDerivativeCoeffs[ndx] *
            ( c(0,ndx).temperature -
              c(0,1-ndx).temperature )
        end
       
        velocityX_YFace   /= c_dy
        velocityY_YFace   /= c_dy
        velocityZ_YFace   /= c_dy
        temperature_YFace /= c_dy

        -- Tensor components (at face)
        var sigmaXY = muFace * ( velocityX_YFace + velocityY_XFace )
        var sigmaYY = muFace * ( 4.0 * velocityY_YFace -
                                 2.0 * velocityX_XFace -
                                 2.0 * velocityZ_ZFace ) / 3.0
        var sigmaZY = muFace * ( velocityZ_YFace + velocityY_ZFace )
        var usigma = velocityFace[0] * sigmaXY +
                     velocityFace[1] * sigmaYY
        -- WARNING : Add term velocityFace[2] * sigmaZY to usigma for 3D
        var cp = fluid_options.gamma * fluid_options.gasConstant / 
                 (fluid_options.gamma - 1.0)
        var heatFlux = - cp / fluid_options.prandtl * 
                         muFace * temperature_YFace

        -- Fluxes
        c.rhoVelocityFlux[0] = sigmaXY
        c.rhoVelocityFlux[1] = sigmaYY
        -- WARNING: Uncomment for 3D rhoVelocityFlux[2] = sigmaZX
        c.rhoEnergyFlux = usigma - heatFlux
        -- WARNING: Add SGS terms for LES

    end
end

Flow.AddViscousUpdateUsingFluxX = liszt kernel(c : grid.cells)
    if c.in_interior then
        c.rhoVelocity_t += (c(0,0).rhoVelocityFlux -
                            c(-1,0).rhoVelocityFlux)/c_dx
        c.rhoEnergy_t   += (c(0,0).rhoEnergyFlux -
                            c(-1,0).rhoEnergyFlux)/c_dx
    end
end

Flow.AddViscousUpdateUsingFluxY = liszt kernel(c : grid.cells)
    if c.in_interior then
        c.rhoVelocity_t += (c(0,0).rhoVelocityFlux -
                            c(0,-1).rhoVelocityFlux)/c_dy
        c.rhoEnergy_t   += (c(0,0).rhoEnergyFlux -
                            c(0,-1).rhoEnergyFlux)/c_dy
    end
end


--------------------
-- Particle coupling
--------------------

Flow.AddParticleCoupling = liszt kernel(p : particles)
    -- WARNING: Assumes that deltaVelocityOverRelaxationTime and 
    -- deltaTemperatureTerm have been computed previously (for example, when
    -- adding the flow coupling to the particles, which should be called before 
    -- in the time stepper)

    -- Retrieve cell containing this particle
    var cell = p.dual_cell.vertex.cell(-1,-1)
    -- Add contribution to momentum and energy equations from the previously
    -- computed deltaVelocityOverRelaxationTime and deltaTemperatureTerm
    cell.rhoVelocity_t -= p.mass * p.deltaVelocityOverRelaxationTime
    cell.rhoEnergy_t   -= p.deltaTemperatureTerm
end

--------------
-- Body Forces
--------------

Flow.AddBodyForces = liszt kernel(c : grid.cells)
    if c.in_interior then
        -- Add body forces to momentum equation
        c.rhoVelocity_t += c.rho *
                           flow_options.bodyForce
    end
end

------------
-- Particles
------------

----------------
-- Flow Coupling
----------------

-- Update particle fields based on flow fields
Particles.AddFlowCouplingPartOne = liszt kernel(p: particles)
    var dc = p.dual_cell
    var flowDensity     = L.double(0)
    var flowVelocity    = L.vec2d({0, 0})
    var flowTemperature = L.double(0)
    var flowDynamicViscosity = L.double(0)
    var pos = p.position
    flowDensity     = InterpolateBilinear(dc, pos, Rho)
    flowVelocity    = InterpolateBilinear(dc, pos, Velocity)
    flowTemperature = InterpolateBilinear(dc, pos, Temperature)
    flowDynamicViscosity = GetDynamicViscosity(flowTemperature)
    p.position_t    += p.velocity
    -- Relaxation time for small particles (not necessarily Stokesian)
    var particleReynoldsNumber = 0
    --  (p.density * norm(flowVelocity - p.velocity) * p.diameter) / 
    --  flowDynamicViscosity
    var relaxationTime = 
      ( p.density * cmath.pow(p.diameter,2) / (18.0 * flowDynamicViscosity) ) /
      ( 1.0 + 0.15 * cmath.pow(particleReynoldsNumber,0.687) )
    p.deltaVelocityOverRelaxationTime = 
      (flowVelocity - p.velocity) / relaxationTime
    p.deltaTemperatureTerm = pi * cmath.pow(p.diameter, 2) *
        particles_options.convective_coefficient *
        (flowTemperature - p.temperature)
    p.velocity_t += p.deltaVelocityOverRelaxationTime
    p.temperature_t += p.deltaTemperatureTerm/
        (p.mass * particles_options.heat_capacity)
end

--------------
-- BODY FORCES
--------------

Particles.AddBodyForces= liszt kernel(p : particles)
    p.velocity_t += particles_options.bodyForce
end

------------
-- RADIATION
------------

Particles.AddRadiation = liszt kernel(p : particles)
    -- Calculate absorbed radiation intensity considering optically thin
    -- particles, for a collimated radiation source with negligible blackbody
    -- self radiation
    var absorbedRadiationIntensity =
      particles_options.absorptivity *
      radiation_options.radiationIntensity *
      p.area / 4

    -- Add contribution to particle temperature time evolution
    p.temperature_t += absorbedRadiationIntensity

end

-- Set particle velocities to flow for initialization
Particles.SetVelocitiesToFlow = liszt kernel(p: particles)
    var dc = p.dual_cell
    var flow_density     = L.double(0)
    var flow_velocity    = L.vec2d({0, 0})
    var flow_temperature = L.double(0)
    var flowDynamicViscosity = L.double(0)
    var pos = p.position
    flow_velocity    = InterpolateBilinear(dc, pos, Velocity)
    p.velocity = flow_velocity
end

------------------------------------------------------------------------------


-- Update flow variables using derivatives
Flow.UpdateKernels = {}
function Flow.GenerateUpdateKernels(relation, stage)
    -- Assumes 4th-order Runge-Kutta 
    local coeff_fun  = TimeIntegrator.coeff_function[stage]
    local coeff_time = TimeIntegrator.coeff_time[stage]
    local deltaTime  = TimeIntegrator.deltaTime
    if stage <= 3 then
        return liszt kernel(r : relation)
            r.rho_new  += coeff_fun * deltaTime * r.rho_t
            r.rho       = r.rho_old +
              coeff_time * deltaTime * r.rho_t
            r.rhoVelocity_new += 
              coeff_fun * deltaTime * r.rhoVelocity_t
            r.rhoVelocity      = r.rhoVelocity_old +
              coeff_time * deltaTime * r.rhoVelocity_t
            r.rhoEnergy_new  += 
              coeff_fun * deltaTime * r.rhoEnergy_t
            r.rhoEnergy       = r.rhoEnergy_old +
              coeff_time * deltaTime * r.rhoEnergy_t
        end
    elseif stage == 4 then
        return liszt kernel(r : relation)
            r.rho = r.rho_new +
               coeff_fun * deltaTime * r.rho_t
            r.rhoVelocity = r.rhoVelocity_new +
               coeff_fun * deltaTime * r.rhoVelocity_t
            r.rhoEnergy = r.rhoEnergy_new +
               coeff_fun * deltaTime * r.rhoEnergy_t
        end
    end
end
for sdx = 1, 4 do
    Flow.UpdateKernels[sdx] = Flow.GenerateUpdateKernels(grid.cells, sdx)
end


-- Update particle variables using derivatives
Particles.UpdateKernels = {}
function Particles.GenerateUpdateKernels(relation, stage)
    local coeff_fun  = TimeIntegrator.coeff_function[stage]
    local coeff_time = TimeIntegrator.coeff_time[stage]
    local deltaTime  = TimeIntegrator.deltaTime
    if stage <= 3 then
        return liszt kernel(r : relation)
            r.position_new += 
               coeff_fun * deltaTime * r.position_t
            r.position       = r.position_old +
               coeff_time * deltaTime * r.position_t
            r.velocity_new += 
               coeff_fun * deltaTime * r.velocity_t
            r.velocity       = r.velocity_old +
               coeff_time * deltaTime * r.velocity_t
            r.temperature_new += 
               coeff_fun * deltaTime * r.temperature_t
            r.temperature       = r.temperature_old +
               coeff_time * deltaTime * r.temperature_t
        end
    elseif stage == 4 then
        return liszt kernel(r : relation)
            r.position = r.position_new +
               coeff_fun * deltaTime * r.position_t
            r.velocity = r.velocity_new +
               coeff_fun * deltaTime * r.velocity_t
            r.temperature = r.temperature_new +
               coeff_fun * deltaTime * r.temperature_t
        end
    end
end
for i = 1, 4 do
    Particles.UpdateKernels[i] = Particles.GenerateUpdateKernels(particles, i)
end


Flow.UpdateAuxiliaryVelocity = liszt kernel(c : grid.cells)
    var velocity = c.rhoVelocity / c.rho
    c.velocity = velocity
    c.kineticEnergy = 0.5 *  L.dot(velocity,velocity)
end

Flow.UpdateGhostFieldsStep1 = liszt kernel(c : grid.cells)
    -- Note that this for now assumes the stencil uses only one point to each
    -- side of the boundary (for example, second order central difference), and
    -- is not able to handle higher-order schemes until a way to specify where
    -- in the (wider-than-one-point) boundary we are
    if c.xneg_depth > 0 then
        c.rhoBoundary            =   c(xoffsetLeft,0).rho
        c.rhoVelocityBoundary[0] =   c(xoffsetLeft,0).rhoVelocity[0] * signDouble
        c.rhoVelocityBoundary[1] =   c(xoffsetLeft,0).rhoVelocity[1]
        c.rhoEnergyBoundary      =   c(xoffsetLeft,0).rhoEnergy
        c.velocityBoundary[0]    =   c(xoffsetLeft,0).velocity[0]
        c.velocityBoundary[1]    =   c(xoffsetLeft,0).velocity[1]
        c.pressureBoundary       =   c(xoffsetLeft,0).pressure
        c.temperatureBoundary    =   c(xoffsetLeft,0).temperature
    end
    if c.xpos_depth > 0 then
        c.rhoBoundary            =   c(-xoffsetRight,0).rho
        c.rhoVelocityBoundary[0] =   c(-xoffsetRight,0).rhoVelocity[0] * signDouble
        c.rhoVelocityBoundary[1] =   c(-xoffsetRight,0).rhoVelocity[1]
        c.rhoEnergyBoundary      =   c(-xoffsetRight,0).rhoEnergy
        c.velocityBoundary[0]    =   c(-xoffsetRight,0).velocity[0]
        c.velocityBoundary[1]    =   c(-xoffsetRight,0).velocity[1]
        c.pressureBoundary       =   c(-xoffsetRight,0).pressure
        c.temperatureBoundary    =   c(-xoffsetRight,0).temperature
    end
    if c.yneg_depth > 0 then
        c.rhoBoundary            =   c(0,yoffsetDown).rho
        c.rhoVelocityBoundary[0] =   c(0,yoffsetDown).rhoVelocity[0]
        c.rhoVelocityBoundary[1] =   c(0,yoffsetDown).rhoVelocity[1] * signDouble
        c.rhoEnergyBoundary      =   c(0,yoffsetDown).rhoEnergy
        c.velocityBoundary[0]    =   c(0,yoffsetDown).velocity[0]
        c.velocityBoundary[1]    =   c(0,yoffsetDown).velocity[1]
        c.pressureBoundary       =   c(0,yoffsetDown).pressure
        c.temperatureBoundary    =   c(0,yoffsetDown).temperature
    end
    if c.ypos_depth > 0 then
        c.rhoBoundary            =   c(0,-yoffsetUp).rho
        c.rhoVelocityBoundary[0] =   c(0,-yoffsetUp).rhoVelocity[0]
        c.rhoVelocityBoundary[1] =   c(0,-yoffsetUp).rhoVelocity[1] * signDouble
        c.rhoEnergyBoundary      =   c(0,-yoffsetUp).rhoEnergy
        c.velocityBoundary[0]    =   c(0,-yoffsetUp).velocity[0]
        c.velocityBoundary[1]    =   c(0,-yoffsetUp).velocity[1]
        c.pressureBoundary       =   c(0,-yoffsetUp).pressure
        c.temperatureBoundary    =   c(0,-yoffsetUp).temperature
    end
    -- Corners:
    -- This step is not necessary with the current numerical schemes that only 
    -- use uni-directional stencils; but this simplifies postprocessing, 
    -- for example plotting field with ghost cells
    -- Currently sets the corner to the value next to it in the diagonal
    if c.xneg_depth > 0 and c.yneg_depth > 0 then
        c.rhoBoundary            =   c(1,1).rho
        c.rhoVelocityBoundary[0] =   c(1,1).rhoVelocity[0]
        c.rhoVelocityBoundary[1] =   c(1,1).rhoVelocity[1]
        c.rhoEnergyBoundary      =   c(1,1).rhoEnergy
        c.velocityBoundary[0]    =   c(1,1).velocity[0]
        c.velocityBoundary[1]    =   c(1,1).velocity[1]
        c.pressureBoundary       =   c(1,1).pressure
        c.temperatureBoundary    =   c(1,1).temperature
    end
    if c.xneg_depth > 0 and c.ypos_depth > 0 then
        c.rhoBoundary            =   c(1,-1).rho
        c.rhoVelocityBoundary[0] = - c(1,-1).rhoVelocity[0]
        c.rhoVelocityBoundary[1] = - c(1,-1).rhoVelocity[1]
        c.rhoEnergyBoundary      =   c(1,-1).rhoEnergy
        c.velocityBoundary[0]    = - c(1,-1).velocity[0]
        c.velocityBoundary[1]    = - c(1,-1).velocity[1]
        c.pressureBoundary       =   c(1,-1).pressure
        c.temperatureBoundary    =   c(1,-1).temperature
    end
    if c.xpos_depth > 0 and c.yneg_depth > 0 then
        c.rhoBoundary            =   c(-1,1).rho
        c.rhoVelocityBoundary[0] =   c(-1,1).rhoVelocity[0]
        c.rhoVelocityBoundary[1] =   c(-1,1).rhoVelocity[1]
        c.rhoEnergyBoundary      =   c(-1,1).rhoEnergy
        c.velocityBoundary[0]    =   c(-1,1).velocity[0]
        c.velocityBoundary[1]    =   c(-1,1).velocity[1]
        c.pressureBoundary       =   c(-1,1).pressure
        c.temperatureBoundary    =   c(-1,1).temperature
    end
    if c.xpos_depth > 0 and c.ypos_depth > 0 then
        c.rhoBoundary            =   c(-1,-1).rho
        c.rhoVelocityBoundary[0] =   c(-1,-1).rhoVelocity[0]
        c.rhoVelocityBoundary[1] =   c(-1,-1).rhoVelocity[1]
        c.rhoEnergyBoundary      =   c(-1,-1).rhoEnergy
        c.velocityBoundary[0]    =   c(-1,-1).velocity[0]
        c.velocityBoundary[1]    =   c(-1,-1).velocity[1]
        c.pressureBoundary       =   c(-1,-1).pressure
        c.temperatureBoundary    =   c(-1,-1).temperature
    end
end
Flow.UpdateGhostFieldsStep2 = liszt kernel(c : grid.cells)
    c.pressure    = c.pressureBoundary
    c.rho         = c.rhoBoundary
    c.rhoVelocity = c.rhoVelocityBoundary
    c.rhoEnergy   = c.rhoEnergyBoundary
    c.pressure    = c.pressureBoundary
    c.temperature = c.temperatureBoundary
end
function Flow.UpdateGhost()
    Flow.UpdateGhostFieldsStep1(grid.cells.boundary)
    Flow.UpdateGhostFieldsStep2(grid.cells.boundary)
end

Flow.UpdateGhostThermodynamicsStep1 = liszt kernel(c : grid.cells)
    -- Note that this for now assumes the stencil uses only one point to each
    -- side of the boundary (for example, second order central difference), and
    -- is not able to handle higher-order schemes until a way to specify where
    -- in the (wider-than-one-point) boundary we are
    if c.xneg_depth > 0 then
        c.pressureBoundary       =   c(xoffsetLeft,0).pressure
        c.temperatureBoundary    =   c(xoffsetLeft,0).temperature
    end
    if c.xpos_depth > 0 then
        c.pressureBoundary       =   c(-xoffsetRight,0).pressure
        c.temperatureBoundary    =   c(-xoffsetRight,0).temperature
    end
    if c.yneg_depth > 0 then
        c.pressureBoundary       =   c(0,yoffsetDown).pressure
        c.temperatureBoundary    =   c(0,yoffsetDown).temperature
    end
    if c.ypos_depth > 0 then
        c.pressureBoundary       =   c(0,-yoffsetUp).pressure
        c.temperatureBoundary    =   c(0,-yoffsetUp).temperature
    end
    -- Corners:
    -- This step is not necessary with the current numerical schemes that only 
    -- use uni-directional stencils; but this simplifies postprocessing, 
    -- for example plotting field with ghost cells
    -- Currently sets the corner to the value next to it in the diagonal
    if c.xneg_depth > 0 and c.yneg_depth > 0 then
        c.pressureBoundary       =   c(1,1).pressure
        c.temperatureBoundary    =   c(1,1).temperature
    end
    if c.xneg_depth > 0 and c.ypos_depth > 0 then
        c.pressureBoundary       =   c(1,-1).pressure
        c.temperatureBoundary    =   c(1,-1).temperature
    end
    if c.xpos_depth > 0 and c.yneg_depth > 0 then
        c.pressureBoundary       =   c(-1,1).pressure
        c.temperatureBoundary    =   c(-1,1).temperature
    end
    if c.xpos_depth > 0 and c.ypos_depth > 0 then
        c.pressureBoundary       =   c(-1,-1).pressure
        c.temperatureBoundary    =   c(-1,-1).temperature
    end
end
Flow.UpdateGhostThermodynamicsStep2 = liszt kernel(c : grid.cells)
    if c.in_boundary then
        c.pressure    = c.pressureBoundary
        c.temperature = c.temperatureBoundary
    end
end
function Flow.UpdateGhostThermodynamics()
    Flow.UpdateGhostThermodynamicsStep1(grid.cells.boundary)
    Flow.UpdateGhostThermodynamicsStep2(grid.cells.boundary)
end

Flow.UpdateGhostVelocityStep1 = liszt kernel(c : grid.cells)
    -- Note that this for now assumes the stencil uses only one point to each
    -- side of the boundary (for example, second order central difference), and
    -- is not able to handle higher-order schemes until a way to specify where
    -- in the (wider-than-one-point) boundary we are
    if c.xneg_depth > 0 then
        c.velocityBoundary[0] =   c(xoffsetLeft,0).velocity[0] * signDouble
        c.velocityBoundary[1] =   c(xoffsetLeft,0).velocity[1]
    end
    if c.xpos_depth > 0 then
        c.velocityBoundary[0] =   c(-xoffsetRight,0).velocity[0] * signDouble
        c.velocityBoundary[1] =   c(-xoffsetRight,0).velocity[1]
    end
    if c.yneg_depth > 0 then
        c.velocityBoundary[0] =   c(0,yoffsetDown).velocity[0]
        c.velocityBoundary[1] =   c(0,yoffsetDown).velocity[1] * signDouble
    end
    if c.ypos_depth > 0 then
        c.velocityBoundary[0] =   c(0,-yoffsetUp).velocity[0]
        c.velocityBoundary[1] =   c(0,-yoffsetUp).velocity[1] * signDouble
    end
    -- Corners:
    -- This step is not necessary with the current numerical schemes that only 
    -- use uni-directional stencils; but this simplifies postprocessing, 
    -- for example plotting field with ghost cells
    -- Currently sets the corner to the value next to it in the diagonal
    if c.xneg_depth > 0 and c.yneg_depth > 0 then
        c.velocityBoundary[0] = - c(1,1).velocity[0]
        c.velocityBoundary[1] = - c(1,1).velocity[1]
    end
    if c.xneg_depth > 0 and c.ypos_depth > 0 then
        c.velocityBoundary[0] = - c(1,-1).velocity[0]
        c.velocityBoundary[1] = - c(1,-1).velocity[1]
    end
    if c.xpos_depth > 0 and c.yneg_depth > 0 then
        c.velocityBoundary[0] = - c(-1,1).velocity[0]
        c.velocityBoundary[1] = - c(-1,1).velocity[1]
    end
    if c.xpos_depth > 0 and c.ypos_depth > 0 then
        c.velocityBoundary[0] = - c(-1,-1).velocity[0]
        c.velocityBoundary[1] = - c(-1,-1).velocity[1]
    end
end
Flow.UpdateGhostVelocityStep2 = liszt kernel(c : grid.cells)
    c.velocity = c.velocityBoundary
end
function Flow.UpdateGhostVelocity()
    Flow.UpdateGhostVelocityStep1(grid.cells.boundary)
    Flow.UpdateGhostVelocityStep2(grid.cells.boundary)
end


Flow.UpdateGhostConservedStep1 = liszt kernel(c : grid.cells)
    -- Note that this for now assumes the stencil uses only one point to each
    -- side of the boundary (for example, second order central difference), and
    -- is not able to handle higher-order schemes until a way to specify where
    -- in the (wider-than-one-point) boundary we are
    if c.xneg_depth > 0 then
        c.rhoBoundary            =   c(xoffsetLeft,0).rho
        c.rhoVelocityBoundary[0] =   c(xoffsetLeft,0).rhoVelocity[0] * signDouble
        c.rhoVelocityBoundary[1] =   c(xoffsetLeft,0).rhoVelocity[1]
        c.rhoEnergyBoundary      =   c(xoffsetLeft,0).rhoEnergy
    end
    if c.xpos_depth > 0 then
        c.rhoBoundary            =   c(-xoffsetRight,0).rho
        c.rhoVelocityBoundary[0] =   c(-xoffsetRight,0).rhoVelocity[0] * signDouble
        c.rhoVelocityBoundary[1] =   c(-xoffsetRight,0).rhoVelocity[1]
        c.rhoEnergyBoundary      =   c(-xoffsetRight,0).rhoEnergy
    end
    if c.yneg_depth > 0 then
        c.rhoBoundary            =   c(0,yoffsetDown).rho
        c.rhoVelocityBoundary[0] =   c(0,yoffsetDown).rhoVelocity[0]
        c.rhoVelocityBoundary[1] =   c(0,yoffsetDown).rhoVelocity[1] * signDouble
        c.rhoEnergyBoundary      =   c(0,yoffsetDown).rhoEnergy
    end
    if c.ypos_depth > 0 then
        c.rhoBoundary            =   c(0,-yoffsetUp).rho
        c.rhoVelocityBoundary[0] =   c(0,-yoffsetUp).rhoVelocity[0]
        c.rhoVelocityBoundary[1] =   c(0,-yoffsetUp).rhoVelocity[1] * signDouble
        c.rhoEnergyBoundary      =   c(0,-yoffsetUp).rhoEnergy
    end
    -- Corners:
    -- This step is not necessary with the current numerical schemes that only 
    -- use uni-directional stencils; but this simplifies postprocessing, 
    -- for example plotting field with ghost cells
    -- Currently sets the corner to the value next to it in the diagonal
    if c.xneg_depth > 0 and c.yneg_depth > 0 then
        c.rhoBoundary            =   c(1,1).rho
        c.rhoVelocityBoundary[0] = - c(1,1).rhoVelocity[0]
        c.rhoVelocityBoundary[1] = - c(1,1).rhoVelocity[1]
        c.rhoEnergyBoundary      =   c(1,1).rhoEnergy
    end
    if c.xneg_depth > 0 and c.ypos_depth > 0 then
        c.rhoBoundary            =   c(1,-1).rho
        c.rhoVelocityBoundary[0] = - c(1,-1).rhoVelocity[0]
        c.rhoVelocityBoundary[1] = - c(1,-1).rhoVelocity[1]
        c.rhoEnergyBoundary      =   c(1,-1).rhoEnergy
    end
    if c.xpos_depth > 0 and c.yneg_depth > 0 then
        c.rhoBoundary            =   c(-1,1).rho
        c.rhoVelocityBoundary[0] = - c(-1,1).rhoVelocity[0]
        c.rhoVelocityBoundary[1] = - c(-1,1).rhoVelocity[1]
        c.rhoEnergyBoundary      =   c(-1,1).rhoEnergy
    end
    if c.xpos_depth > 0 and c.ypos_depth > 0 then
        c.rhoBoundary            =   c(-1,-1).rho
        c.rhoVelocityBoundary[0] = - c(-1,-1).rhoVelocity[0]
        c.rhoVelocityBoundary[1] = - c(-1,-1).rhoVelocity[1]
        c.rhoEnergyBoundary      =   c(-1,-1).rhoEnergy
    end
end
Flow.UpdateGhostConservedStep2 = liszt kernel(c : grid.cells)
    c.pressure    = c.pressureBoundary
    c.rho         = c.rhoBoundary
    c.rhoVelocity = c.rhoVelocityBoundary
    c.rhoEnergy   = c.rhoEnergyBoundary
end
function Flow.UpdateGhostConserved()
    Flow.UpdateGhostConservedStep1(grid.cells.boundary)
    Flow.UpdateGhostConservedStep2(grid.cells.boundary)
end


Flow.UpdateAuxiliaryThermodynamics = liszt kernel(c : grid.cells)
    var kineticEnergy = 
      0.5 * c.rho * L.dot(c.velocity,c.velocity)
    -- Define temporary pressure variable to avoid error like this:
    -- Errors during typechecking liszt
    -- examples/soleil/soleil.t:557: access of 'cells.pressure' field in <Read> phase
    -- conflicts with earlier access in <Write> phase at examples/soleil/soleil.t:555
    -- when I try to reuse the c.pressure variable to calculate the temperature
    var pressure = (fluid_options.gamma - 1.0) * 
                   ( c.rhoEnergy - kineticEnergy )
    c.pressure = pressure 
    c.temperature =  pressure / ( fluid_options.gasConstant * c.rho)
end


Particles.UpdateAuxiliaryStep1 = liszt kernel(p : particles)
    p.position_periodic[0] = p.position[0]
    p.position_periodic[1] = p.position[1]
    if p.position[0] < gridOriginX then
        p.position_periodic[0] = p.position[0] + grid_options.width
    end
    if p.position[0] > gridOriginX + grid_options.width then
        p.position_periodic[0] = p.position[0] - grid_options.width
    end
    if p.position[1] < gridOriginY then
        p.position_periodic[1] = p.position[1] + grid_options.width
    end
    if p.position[1] > gridOriginY + grid_options.height then
        p.position_periodic[1] = p.position[1] - grid_options.height
    end
end

Particles.UpdateAuxiliaryStep2 = liszt kernel(p : particles)
    p.position = p.position_periodic
end

Flow.ComputeVelocityGradientX = liszt kernel(c : grid.cells)
    var numFirstDerivativeCoeffs = spatial_stencil.numFirstDerivativeCoeffs
    var firstDerivativeCoeffs    = spatial_stencil.firstDerivativeCoeffs
    var tmp = L.vec2d({0.0, 0.0})
    for ndx = 1, numFirstDerivativeCoeffs do
      tmp += firstDerivativeCoeffs[ndx] * 
              ( c(ndx,0).velocity -
                c(-ndx,0).velocity )
    end
    c.velocityGradientX = tmp / c_dx
end

Flow.ComputeVelocityGradientY = liszt kernel(c : grid.cells)
    var numFirstDerivativeCoeffs = spatial_stencil.numFirstDerivativeCoeffs
    var firstDerivativeCoeffs    = spatial_stencil.firstDerivativeCoeffs
    var tmp = L.vec2d({0.0, 0.0})
    for ndx = 1, numFirstDerivativeCoeffs do
      tmp += firstDerivativeCoeffs[ndx] * 
              ( c(0,ndx).velocity -
                c(0,-ndx).velocity )
    end
    c.velocityGradientY = tmp / c_dy
end

-- kernels to draw particles and velocity for debugging purpose

Particles.DrawKernel = liszt kernel (p : particles)
    var color = {1.0,1.0,0.0}
    vdb.color(color)
    var pmax = L.double(particles_options.pos_max)
    var tmax = 10.0
    var color : L.vec3d = 
      {p.temperature/particles_options.initialTemperature,
       1-p.temperature/particles_options.initialTemperature,
       0}
    var pos : L.vec3d = { p.position[0]/pmax,
                          p.position[1]/pmax,
                          0.0 }
    vdb.color(color)
    vdb.point(pos)
    --var vel = p.velocity
    --var v = L.vec3d({ vel[0], vel[1], 0.0 })
    --v = 200 * v + L.vec3d({2, 2, 0})
    --vdb.line(pos, pos+v)
end


-----------------------------------------------------------------------------
--[[                             MAIN LOOP                               ]]--
-----------------------------------------------------------------------------


function TimeIntegrator.InitializeTemporaries()
    Flow.InitializeTemporaries(grid.cells)
    Particles.InitializeTemporaries(particles)
end


function TimeIntegrator.InitializeTimeDerivatives()
    Flow.InitializeTimeDerivatives(grid.cells)
    Particles.InitializeTimeDerivatives(particles)
end


function Flow.AddInviscid()
    Flow.AddInviscidInitialize(grid.cells)
    Flow.AddInviscidGetFluxX(grid.cells)
    Flow.AddInviscidUpdateUsingFluxX(grid.cells)
    Flow.AddInviscidGetFluxY(grid.cells)
    Flow.AddInviscidUpdateUsingFluxY(grid.cells)
end

Flow.UpdateGhostVelocityGradientStep1 = liszt kernel(c : grid.cells)
    -- Note that this for now assumes the stencil uses only one point to each
    -- side of the boundary (for example, second order central difference), and
    -- is not able to handle higher-order schemes until a way to specify where
    -- in the (wider-than-one-point) boundary we are
    if c.xneg_depth > 0 then
        c.velocityGradientXBoundary[0] = - c(xoffsetLeft,0).velocityGradientX[0]
        c.velocityGradientXBoundary[1] =   c(xoffsetLeft,0).velocityGradientX[1]
        c.velocityGradientYBoundary[0] = - c(xoffsetLeft,0).velocityGradientY[0]
        c.velocityGradientYBoundary[1] =   c(xoffsetLeft,0).velocityGradientY[1]
    end
    if c.xpos_depth > 0 then
        c.velocityGradientXBoundary[0] = - c(-xoffsetRight,0).velocityGradientX[0]
        c.velocityGradientXBoundary[1] =   c(-xoffsetRight,0).velocityGradientX[1]
        c.velocityGradientYBoundary[0] = - c(-xoffsetRight,0).velocityGradientY[0]
        c.velocityGradientYBoundary[1] =   c(-xoffsetRight,0).velocityGradientY[1]
    end
    if c.yneg_depth > 0 then
        c.velocityGradientXBoundary[0] =   c(0,yoffsetDown).velocityGradientX[0]
        c.velocityGradientXBoundary[1] = - c(0,yoffsetDown).velocityGradientX[1]
        c.velocityGradientYBoundary[0] =   c(0,yoffsetDown).velocityGradientY[0]
        c.velocityGradientYBoundary[1] = - c(0,yoffsetDown).velocityGradientY[1]
    end
    if c.ypos_depth > 0 then
        c.velocityGradientXBoundary[0] =   c(0,-yoffsetUp).velocityGradientX[0]
        c.velocityGradientXBoundary[1] = - c(0,-yoffsetUp).velocityGradientX[1]
        c.velocityGradientYBoundary[0] =   c(0,-yoffsetUp).velocityGradientY[0]
        c.velocityGradientYBoundary[1] = - c(0,-yoffsetUp).velocityGradientY[1]
    end
end
Flow.UpdateGhostVelocityGradientStep2 = liszt kernel(c : grid.cells)
    if c.in_boundary then
        c.velocityGradientX = c.velocityGradientXBoundary
        c.velocityGradientY = c.velocityGradientYBoundary
    end
end
function Flow.UpdateGhostVelocityGradient()
    Flow.UpdateGhostVelocityGradientStep1(grid.cells)
    Flow.UpdateGhostVelocityGradientStep2(grid.cells)
end

function Flow.AddViscous()
    Flow.AddViscousGetFluxX(grid.cells)
    Flow.AddViscousUpdateUsingFluxX(grid.cells)
    Flow.AddViscousGetFluxY(grid.cells)
    Flow.AddViscousUpdateUsingFluxY(grid.cells)
end


function Particles.AddFlowCoupling()
    Particles.Locate(particles)
    Particles.AddFlowCouplingPartOne(particles)
end

function Flow.Update(stage)
    Flow.UpdateKernels[stage](grid.cells)
end

function Particles.Update(stage)
    Particles.UpdateKernels[stage](particles)
end

function Flow.ComputeVelocityGradients()
    Flow.ComputeVelocityGradientX(grid.cells.interior)
    Flow.ComputeVelocityGradientY(grid.cells.interior)
end
function Flow.UpdateAuxiliaryVelocityConservedAndGradients()
    Flow.UpdateAuxiliaryVelocity(grid.cells.interior)
    Flow.UpdateGhostConserved()
    Flow.UpdateGhostVelocity()
    Flow.ComputeVelocityGradients()
end

function Flow.UpdateAuxiliary()
    Flow.UpdateAuxiliaryVelocityConservedAndGradients()
    Flow.UpdateAuxiliaryThermodynamics(grid.cells.interior)
    Flow.UpdateGhostThermodynamics()
end

function Particles.UpdateAuxiliary()
    Particles.UpdateAuxiliaryStep1(particles)
    Particles.UpdateAuxiliaryStep2(particles)
end

function TimeIntegrator.UpdateAuxiliary()
    Flow.UpdateAuxiliary()
    Particles.UpdateAuxiliary()
end


-- Update time function
function TimeIntegrator.UpdateTime(timeOld, stage)
    TimeIntegrator.simTime = timeOld +
                             TimeIntegrator.coeff_time[stage] *
                             TimeIntegrator.deltaTime:get()
end

function TimeIntegrator.InitializeVariables()
    Flow.InitializeCenterCoordinates(grid.cells)
    Flow.InitializePrimitives(grid.cells.interior)
    Flow.UpdateConservedFromPrimitive(grid.cells.interior)
    Flow.UpdateGhost()
    Flow.UpdateAuxiliary()

    Particles.Locate(particles)
    --Particles.SetVelocitiesToFlow(particles)
end

function TimeIntegrator.ComputeDFunctionDt()
    Flow.AddInviscid()
    Flow.UpdateGhostVelocityGradient()
    Flow.AddViscous()
    Flow.AddBodyForces(grid.cells)
    Particles.AddFlowCoupling()
    Flow.AddParticleCoupling(particles)
    Particles.AddBodyForces(particles)
    Particles.AddRadiation(particles)
end

function TimeIntegrator.UpdateSolution(stage)
    Flow.Update(stage)
    Particles.Update(stage)
end

function TimeIntegrator.AdvanceTimeStep()

    TimeIntegrator.InitializeTemporaries()
    local timeOld = TimeIntegrator.simTime
    for stage = 1, 4 do
        TimeIntegrator.InitializeTimeDerivatives()
        TimeIntegrator.ComputeDFunctionDt()
        TimeIntegrator.UpdateSolution(stage)
        TimeIntegrator.UpdateAuxiliary()
        TimeIntegrator.UpdateTime(timeOld, stage)
    end

    TimeIntegrator.timeStep = TimeIntegrator.timeStep + 1

end

-- Integral quantities
-- Note: - numberOfInteriorCells and areaInterior could be defined as variables
-- from grid instead of Flow. Here Flow is used to avoid adding things to grid
-- externally
Flow.numberOfInteriorCells = L.NewGlobal(L.int, 0)
Flow.areaInterior = L.NewGlobal(L.double, 0)
Flow.averagePressure = L.NewGlobal(L.double, 0.0)
Flow.averageTemperature = L.NewGlobal(L.double, 0.0)
Flow.averageKineticEnergy = L.NewGlobal(L.double, 0.0)
Particles.averageTemperature= L.NewGlobal(L.double, 0.0)

Flow.IntegrateQuantities = liszt kernel(c : grid.cells)
    if c.in_interior then
        -- WARNING: update cellArea computation for non-uniform grids
        --var cellArea = c.cellWidth() * c.cellHeight()
        var cellArea = c_dx * c_dy
        Flow.numberOfInteriorCells += 1
        Flow.areaInterior += cellArea
        Flow.averagePressure += c.pressure * cellArea
        Flow.averageTemperature += c.temperature * cellArea
        Flow.averageKineticEnergy += c.kineticEnergy * cellArea
    end
end

Particles.IntegrateQuantities = liszt kernel(p : particles)
    Particles.averageTemperature += p.temperature
end

function Statistics.ResetSpatialAverages()
    Flow.numberOfInteriorCells:set(0)
    Flow.areaInterior:set(0)
    Flow.averagePressure:set(0.0)
    Flow.averageTemperature:set(0.0)
    Flow.averageKineticEnergy:set(0.0)
    Particles.averageTemperature:set(0.0)
end

function Statistics.UpdateSpatialAverages(grid, particles)
    -- Flow
    Flow.averagePressure:set(
      Flow.averagePressure:get()/
      Flow.areaInterior:get())
    Flow.averageTemperature:set(
      Flow.averageTemperature:get()/
      Flow.areaInterior:get())
    Flow.averageKineticEnergy:set(
      Flow.averageKineticEnergy:get()/
      Flow.areaInterior:get())

    -- Particles
    Particles.averageTemperature:set(
      Particles.averageTemperature:get()/
      particles:Size())

end

function Statistics.ComputeSpatialAverages()
    Statistics.ResetSpatialAverages()
    Flow.IntegrateQuantities(grid.cells)
    Particles.IntegrateQuantities(particles)
    Statistics.UpdateSpatialAverages(grid, particles)
end

Flow.CalculateSpectralRadii = liszt kernel(c : grid.cells)
    var dXYInverseSquare = 1.0/c_dx * 1.0/c_dx +
                           1.0/c_dy * 1.0/c_dy
    -- Convective spectral radii
    c.convectiveSpectralRadius = 
       (cmath.fabs(c.velocity[0])/c_dx  +
        cmath.fabs(c.velocity[1])/c_dy  +
        GetSoundSpeed(c.temperature) * cmath.sqrt(dXYInverseSquare)) *
       spatial_stencil.firstDerivativeModifiedWaveNumber
    
    -- Viscous spectral radii (including sgs model component)
    var dynamicViscosity = GetDynamicViscosity(c.temperature)
    var eddyViscosity = c.sgsEddyViscosity
    c.viscousSpectralRadius = 
       (2.0 * ( dynamicViscosity + eddyViscosity ) /
        c.rho * dXYInverseSquare) *
       spatial_stencil.secondDerivativeModifiedWaveNumber
    
    -- Heat conduction spectral radii (including sgs model 
    -- component)
    var cv = fluid_options.gasConstant / 
             (fluid_options.gamma - 1.0)
    var cp = fluid_options.gamma * cv
    var kappa = cp / fluid_options.prandtl *  dynamicViscosity
    
    c.heatConductionSpectralRadius = 
       ((kappa + c.sgsEddyKappa) / (cv * c.rho) * dXYInverseSquare) *
       spatial_stencil.secondDerivativeModifiedWaveNumber

end

-- Get maximum of field
-- Note: This should likely be built-in within grid.t. Here it is placed under
-- the Flow namespace to avoid interference with grid (where it seems more
-- appropriate)
Flow.GetMaximumOfField = function (field)
    local maxval = - math.huge

    local function test_row_larger(id, val)
        if val > maxval then maxval = val end
    end

    field:DumpFunction(test_row_larger)
    return maxval
end

function TimeIntegrator.CalculateDeltaTime()

    Flow.CalculateSpectralRadii(grid.cells)

    -- Get maximum spectral radii in the domain
    local convectiveSpectralRadius = 
            Flow.GetMaximumOfField(grid.cells.convectiveSpectralRadius)
    local viscousSpectralRadius =
            Flow.GetMaximumOfField(grid.cells.viscousSpectralRadius)
    local heatConductionSpectralRadius =
            Flow.GetMaximumOfField(grid.cells.heatConductionSpectralRadius)

    -- Calculate diffusive spectral radius as the maximum between
    -- heat conduction and convective spectral radii
    local diffusiveSpectralRadius = viscousSpectralRadius
    if heatConductionSpectralRadius > viscousSpectralRadius then
      diffusiveSpectralRadius = heatConductionSpectralRadius
    end

    -- Calculate global spectral radius as the maximum between the convective 
    -- and diffusive spectral radii
    local spectralRadius = convectiveSpectralRadius
    if diffusiveSpectralRadius > convectiveSpectralRadius then
      spectralRadius = diffusiveSpectralRadius
    end
    TimeIntegrator.deltaTime:set(TimeIntegrator.cfl / spectralRadius)
    --TimeIntegrator.deltaTime:set(0.005)

end

-----------------------------------------------------------------------------
--[[                            OUTPUT                                   ]]--
-----------------------------------------------------------------------------

-- Write cells field to output file
Flow.WriteField = function (outputFileNamePrefix,xSize,ySize,field)
    -- Make up complete file name based on name of field
    local outputFileName = outputFileNamePrefix .. "_" ..
                           field:Name() .. ".txt"
    -- Open file
    local outputFile = io.output(outputFileName)
    -- Write data
    local values = field:DumpToList()
    local N      = field:Size()

    if field:Type():isVector() then
        local veclen = field:Type().N
        io.write("# ", xSize, " ", ySize, " ", N, " ", veclen, "\n")
        for i=1,N do
            local s = ''
            for j=1,veclen do
                local t = tostring(values[i][j]):gsub('ULL',' ')
                s = s .. ' ' .. t .. ''
            end
            -- i-1 to return to 0 indexing
            io.write("", i-1, s, "\n")
        end
    else
        io.write("# ", xSize, " ", ySize, " ", N, " ", 1, "\n")
        for i=1,N do
            local t = tostring(values[i]):gsub('ULL', ' ')
            -- i-1 to return to 0 indexing
            io.write("", i-1, ' ', t,"\n")
        end
    end
    io.close()
end

-- Write particles field to output file
Particles.WriteField = function (outputFileNamePrefix,field)
    -- Make up complete file name based on name of field
    local outputFileName = outputFileNamePrefix .. "_" ..
                           field:Name() .. ".txt"
    -- Open file
    local outputFile = io.output(outputFileName)
    -- Write data
    local values = field:DumpToList()
    local N      = field:Size()

    if field:Type():isVector() then
        local veclen = field:Type().N
        io.write("# ", N, " ", veclen, "\n")
        for i=1,N do
            local s = ''
            for j=1,veclen do
                local t = tostring(values[i][j]):gsub('ULL',' ')
                s = s .. ' ' .. t .. ''
            end
            -- i-1 to return to 0 indexing
            io.write("", i-1, s, "\n")
        end
    else
        io.write("# ", N, " ", 1, "\n")
        for i=1,N do
            local t = tostring(values[i]):gsub('ULL', ' ')
            -- i-1 to return to 0 indexing
            io.write("", i-1, ' ', t,"\n")
        end
    end
    io.close()
end

function IO.WriteOutput(timeStep)

    -- Output log message
    print("Time step ", string.format("%d",TimeIntegrator.timeStep), 
          ", dt", string.format("%10.8f",TimeIntegrator.deltaTime:get()),
          ", t", string.format("%10.8f",TimeIntegrator.simTime),
          "flowP", string.format("%10.8f",Flow.averagePressure:get()),
          "flowT", string.format("%10.8f",Flow.averageTemperature:get()),
          "kineticEnergy", string.format("%10.8f",Flow.averageKineticEnergy:get()),
          "particlesT", string.format("%10.8f",Particles.averageTemperature:get())
          )

    -- Check if it is time to output to file
    if timeStep  % TimeIntegrator.outputEveryTimeSteps == 0 then
        --print("Time to output")
        local outputFileName = IO.outputFileNamePrefix .. "_" ..
          tostring(timeStep)
        Flow.WriteField(outputFileName .. "_flow",
          grid:xSize(),grid:ySize(),grid.cells.temperature)
        --Flow.WriteField(outputFileName .. "_flow",
        --  grid:xSize(),grid:ySize(),grid.cells.rho)
        Flow.WriteField(outputFileName .. "_flow",
          grid:xSize(),grid:ySize(),grid.cells.pressure)
        Flow.WriteField(outputFileName .. "_flow",
          grid:xSize(),grid:ySize(),grid.cells.kineticEnergy)
        Particles.WriteField(outputFileName .. "_particles",
          particles.position)
        Particles.WriteField(outputFileName .. "_particles",
          particles.velocity)
        Particles.WriteField(outputFileName .. "_particles",
          particles.temperature)
    end
end

function Particles.Draw()
    vdb.vbegin()
    vdb.frame()
    Particles.DrawKernel(particles)
    vdb.vend()
end

-----------------------------------------------------------------------------
--[[                            MAIN EXECUTION                           ]]--
-----------------------------------------------------------------------------

TimeIntegrator.InitializeVariables()

Flow.WriteField(IO.outputFileNamePrefix,
                grid:xSize(),grid:ySize(),
                grid.cells.centerCoordinates)
Particles.WriteField(IO.outputFileNamePrefix .. "_particles",
                     particles.diameter)

-- Time loop
while (TimeIntegrator.simTime < TimeIntegrator.final_time) do

    TimeIntegrator.CalculateDeltaTime()
    TimeIntegrator.AdvanceTimeStep()
    Statistics.ComputeSpatialAverages()
    IO.WriteOutput(TimeIntegrator.timeStep)

    Particles.Draw()

end
