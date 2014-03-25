local S = {}
package.loaded["soleil_liszt.particle_distribution"] = S

import "compiler.liszt"
local Grid = terralib.require "compiler.grid"
local cmath = terralib.includecstring "#include <math.h>"

--------------------------------------------------------------------------------
--[[ ParticleDistribution type                                              ]]--
--------------------------------------------------------------------------------

local ParticleDistribution = {}
ParticleDistribution.__index = ParticleDistribution

function S.NewParticleDistribution(gridData,
                        number, type,
                        position, velocity,
                        temperature, diameter, density,
                        bodyForceType, bodyForce,
                        heatCapacity, convectionCoefficient)
    local pd = setmetatable({}, ParticleDistribution)
    local grid = gridData.grid

    -- particles relation
    local particles = L.NewRelation(number, 'particles')
    grid.particles = particles

    -- member variables
    pd.grid = grid
    pd.particlesType = type
    pd.bodyForceType = bodyForceType
    pd.bodyForce = bodyForce
    pd.heatCapacity = heatCapacity
    pd.convectionCoefficient = convectionCoefficient

    -- fields for particles relation
    particles:NewField('position', L.vector(L.double, 3)):LoadTableArray(position)
    particles:NewField('velocity', L.vector(L.double, 3)):LoadTableArray(velocity)
    particles:NewField('temperature', L.double):LoadTableArray(temperature)
    particles:NewField('diameter', L.double):LoadTableArray(diameter)
    particles:NewField('density', L.double):LoadTableArray(density)
    particles:NewField('deltaVelocityOverRelaxationTime',
                       L.vector(L.double, 3)):
                       LoadConstant(L.NewVector(L.double, {0.0, 0.0, 0.0}))
    particles:NewField('deltaTemperatureTerm', L.double):LoadConstant(0.0)
    -- particles have a field cell
    particles:NewField('cell', grid.cells)
    particles:NewField('dual_cell', grid.dual_cells)

    -- member methods
    pd.SetVelocitiesToFlow = SetVelocitiesToFlowHelper(pd)
    pd.AddFlowCoupling = AddFlowCouplingHelper(pd)

    return pd
end

local function SetVelocitiesToFlowHelper(pd)
    local SetVelocitiesToFlowKernel = liszt kernel(p : pd)
        -- TODO: define trilinear interpolation macro
        p.velocity = InterpolateTrilinear(p.dual_cell, 'velocity')
    end
    return function()
        SetVelocitiesToFlowKernel(pd)
    end
end

local function AddFlowCouplingHelper(pd)
    local AddFlowCouplingKernel = liszt kernel(p : pd)
        p.velocity = InterpolateTrilinear(p.dual_cell, 'velocity')
    end
end
