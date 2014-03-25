local S = {}
package.loaded["soleil_liszt.time_integrator"] = S

import "compiler.liszt"
local Grid = terralib.require "compiler.grid"
local cmath = terralib.includecstring "#include <math.h>"

--------------------------------------------------------------------------------
--[[ TimeIntegrator type                                                    ]]--
--------------------------------------------------------------------------------

local TimeIntegrator = {}
TimeIntegrator.__index = TimeIntegrator

function S.NewTimeIntegrator(input, flow, particles)
    local t = setmetatable({}, TimeIntegrator)

    -- member variables
    -- TODO: fill these based on configuration file
    t.timeIntegratorElement = {}
    t.type = {}
    t.maxTimeStep = {}
    t.finalTime = {}
    t.cfl = {}

    t.flow = flow
    t.particles = particles
    t.simulationTime = 0.0
    t.timeStep = 0
    t.deltaTime = -1
    t.stage = -1

    -- TODO: fill this based on the math library being used
    assert(t.type == "RungeKutta4")
    t.coeffTime = {}
    t.coeffFunction = {}
    
    -- member methods
    t.CalculateDeltaTime = DefineCalculateDeltaTimeHelper(t)

    return t
end

local function DefineCalculateDeltaTimeHelper(t)
    local convectiveSpectralRadius     = L.NewScalar(L.float, 0)
    local viscousSpectralRadius        = L.NewScalar(L.float, 0)
    local heatConductionSpectralRadius = L.NewScalar(L.float, 0)
    -- define kernel the first time
    local fluid = t.flow.fluid
    local grid = t.flow.gridData.grid
    local spatialStencils = t.flow.gridData.spatialStencils
    local CalculateDeltaTimeKernel = liszt kernel(c : grid.cells)
        -- Calculate equivalent cell diagonal from
        var dXYZInverse = {1.0/c.dXYZ[0], 1.0/c.dXYZ[1], 1.0/c.dXYZ[2]}
        var dXYZInverseSquare  = dot(dXYZInverse, dXYZInverse)
        var dXYZInverse = cmath.sqrt(dXYZInverseSquare)
        -- Convective spectral radii
        var local_cond_spectral_radius = cmath.fabs(c.velocity[0]) * dXInverse +
                                         cmath.fabs(c.velocity[1]) * dYInverse +
                                         cmath.fabs(c.velocity[2]) * dZInverse +
                                         fluid.GetSoundSpeed(c.temperature) *
                                         dXYZInverse
        -- TODO: max reduction required. fill in appropriate code.
        convectiveSpectralRadius = convectiveSpectralRadius
                                   max local_cond_spectral_radius
        -- Viscous spectral radii (including sgs model component)
        var dynamicViscosity = fluid.
                               GetDynamicViscosity(c.temperature * dXYZInverse)
        var eddyViscosity = c.sgsEddyViscosity
        local_viscous_spectral_radius = 2.0 *
                                        (dynamicViscosity + eddyViscosity)/
                                        c.rho * dXYZInverseSquare
        -- TODO: max reduction required. fill in appropriate code.
        viscousSpectralRadius = viscousSpectralRadius
                                max local_viscous_spectral_radius
        -- Heat conduction spectral radii (including sgs model component)
        var kappa = fluid.cp * dynamicViscosity / fluid.prandtl
        var eddyKappa = c.sgsEddyKappa
        var local_heatcond_spectral_radius = (kappa + eddyKappa) /
                                             (fluid.cv * c.rho) *
                                             dXYZInverseSquare
        -- TODO: max reduction required. fill in appropriate code.
        heatConductionSpectralRadius = heatConductionSpectralRadius
                                       max local_heatcond_spectral_radius
    end
    return function()
        data.convectiveSpectralRadius:setTo(0)
        data.viscousSpectralRadius:setTo(0)
        data.heatConductionSpectralRadius:setTo(0)
        CalculateDeltaTimeKernel(grid.cells)
        local csr = convectiveSpectralRadius:value() *
                    spatialStencils.firstDerivativeModifiedWaveNumber
        local vsr = viscousSpectralRadius:value() *
                    spatialStencils.secondDerivativeModifiedWaveNumber
        local hsr = heatConductionSpectralRadius:value() *
                    spatialStencils.secondDerivativeModifiedWaveNumber
        local diffusiveSpectralRadius = cmath.fmax(vsr, hsr)
        local spectralRadius = cmath.fmax(csr, diffusiveSpectralRadius)
        t.deltaTime = t.cfl / spectralRadius
    end
end

function TimeIntegrator:ComputeDFunctionDt(rho_t,
                                           rhoVel_t,
                                           rhoEnergy_t,
                                           particlesPosition_t,
                                           particlesVelocity_t)
end

function TimeIntegrator:UpdateArrays(oldArray,
                                     newArray,
                                     dFunctionDtArray,
                                     solutionArray)
end

function TimeIntegrator:UpdateAuxiliary()
end

function TimeIntegrator:AdvanceTimeStep()
end
