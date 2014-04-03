import "compiler.liszt"
local Grid  = terralib.require 'compiler.grid'
local cmath = terralib.includecstring [[ #include <math.h> ]]


-----------------------------------------------------------------------------
--[[                             OPTIONS                                 ]]--
-----------------------------------------------------------------------------


local grid_options = {
    xnum = 16,
    ynum = 16,
    pos = {0.0, 0.0},
    width = 6.28,
    height = 6.28
}


local particle_options = {
    num = 50
}


-----------------------------------------------------------------------------
--[[                       FLOW/ PARTICLE RELATIONS                      ]]--
-----------------------------------------------------------------------------


-- Declare and initialize grid  and related fields

local grid = Grid.New2dUniformGrid(grid_options.xnum, grid_options.ynum,
                                   grid_options.pos,
                                   grid_options.width, grid_options.height)

-- conserved variables
grid.cells:NewField('rho', L.float)
grid.cells:NewField('rho_velocity', L.vec2f)
grid.cells:NewField('rho_energy', L.float)

-- primitive variables
grid.cells:NewField('velocity', L.vec2f)
grid.cells:NewField('temperature', L.float)
grid.cells:NewField('pressure', L.float)

-- scratch (temporary) fields
-- intermediate value
grid.cells:NewField('rho_temp', L.float)
grid.cells:NewField('rho_velocity_temp', L.vec2f)
grid.cells:NewField('rho_energy_temp', L.float)
-- derivatives
grid.cells:NewField('rho', L.float)
grid.cells:NewField('rho_velocity_t', L.vec2f)
grid.cells:NewGield('rho_energy_t', L.float)
-- flux
-- TODO: Define face and related macros
grid.faces:NewField('rho_flux', L.float)
grid.faces:NewField('rho_velocity_flux', L.vec2f)
grid.faces:NewField('rho_energy_flux', L.float)
grid.faces:NewField('rho_enthalpy', L.float)


-- Declare and initialize particle relation and fields over the particle

local particles = L.NewRelation(particle_options.num, 'particles')

particles:NewField('dual_cell', grid.dual_cells)
particles:NewField('position', L.vec2f)
particles:NewField('velocity', L.vec2f)
particles:NewField('temperature', L.float)

particles:NewField('diameter', L.float)
particles:NewField('density', L.float)

particles:NewField('delta_velocity_over_relaxaion_time', L.vec2f)
particles:NewField('delta_temperature_term', L.float)

-- scratch (temporary) fields
-- derivatives
particles:NewField('position_t', L.vec2f)
particles:NewField('velocity_t', L.vec2f)
particles:NewField('temperature_t', L.float)


-----------------------------------------------------------------------------
--[[                            LISZT GLOBALS                            ]]--
-----------------------------------------------------------------------------


-- If you want to use global variables like scalar values or vectors within
-- a Liszt kernel (variables that are not constants), declare them here.
-- If you use lua variables, Liszt will treat them as constant values.


-----------------------------------------------------------------------------
--[[                             LISZT MACROS                            ]]--
-----------------------------------------------------------------------------


local Rho = L.NewMacro(function(c)
    return liszt `c.rho
end)

local Velocity = L.NewMacro(function(c)
    return liszt `c.velocity
end)

local Temperature = L.NewMacro(function(c)
    return liszt `c.temperature
end)

local InterpolateBilinear = L.NewMacro(function(dc, Field)
    return liszt quote
        var c00 = dc.cell(-1, -1)
        var c01 = dc.cell(-1, 1)
        var c10 = dc.cell(1, -1)
        var c11 = dc.cell(1, 1)
        var delta1 = cmath.abs(c00.position[1] - dc.y)
        var delta2 = cmath.abs(c11.position[1] - dc.y)
        var f1 = (delta1*Field(c00) + delta2*Field(c01)) / (delta1 + delta2)
        var f2 = (delta1*Field(c10) + delta2*Field(c11)) / (delta1 + delta2)
        delta1 = cmath.abs(c00.position[0] - dc.x)
        delta2 = cmath.abs(c11.position[0] - dc.x)
    in
        (delta1*f1 + delta2*f2) / (delta1 + delta2)
    end
end)

local GetDynamicViscosity = L.NewMacro(function(temperature)
    return liszt `fluid_options.dynamic_viscosity_ref *
        cmath.pow(temperature/fluid_options.dynamic_viscosity_temp_ref, 0.75)
end)

-----------------------------------------------------------------------------
--[[                            LISZT KERNELS                            ]]--
-----------------------------------------------------------------------------


-- Locate particles
local LocateParticles = liszt kernel(p : particles)
    p.dual_cell = grid.dual(p.position)
end

-- Initialize enthalpy
local AddInviscidInitialize = liszt kernel(c : grid.cells)
    c.rho_enthalpy = c.rho_energy + c.pressure
end


-- Interpolation for flux values
-- read conserved variables, write flux variables
local AddInviscidGetFlux = liszt kernel(f : grid.faces)
    var num_coeffs = spatial_stencil.num_interpolate_coeffs
    var coeffs     = spatial_stencil.interpolate_coeffs
    var rho_diagonal = 0.0
    var rho_skew     = 0.0
    var rho_velocity_diagonal = {0.0, 0.0}
    var rho_velocity_skew     = {0.0, 0.0}
    var rho_energy_diagonal   = 0.0
    var rho_energy_skew       = 0.0
    var fpdiag = 0.0
    var axis = f.axis
    for i = 0, num_coeffs do
        rho_diagonal += coeffs[i] *
                      ( f.cell(i+1).rho *
                        f.cell(i+1).velocity[axis] +
                        f.cell(i-1).rho *
                        f.cell(i-1).velocity[axis] )
        rho_velocity_diagonal += coeffs[i] *
                               ( f.cell(i+1).rho_velocity *
                                 f.cell(i+1).velocity[axis] +
                                 f.cell(i-1).rho_velocity *
                                 f.cell(i-1).velocity[axis] )
        rho_energy_diagonal += coeffs[i] *
                             ( f.cell(i+1).rho_enthalpy *
                               f.cell(i+1).velocity[axis] +
                               f.cell(i-1).rho_enthalpy *
                               f.cell(i-1).velocity[axis] )
        fpdiag += coeffs[i] *
                ( f.cell(i+1).pressure +
                  f.cell(i-1).pressure )
    end
    -- TODO: I don't understand what skew is. Ask Ivan.
    var s = spatial_stencil.split
    f.rho_flux          = s * rho_diagonal +
                          (1-s) * rho_skew
    f.rho_velocity_flux = s * rho_velocity_diagonal +
                          (1-s) * rho_velocity_skew
    f.rho_energy_flux   = s * rho_energy_diagonal +
                          (1-s) * rho_energy_skew
end


-- Update conserved variables using flux values from part 1
-- write conserved variables, read flux variables
local AddInviscidUpdateUsingFlux = liszt kernel(c : grid.cells)
    c.rho_t -= (c.face(1,0,0).rho_flux -
                c.face(-1,0,0).rho_flux)/c.dx
    c.rho_t -= (c.face(0,1,0).rho_flux -
                c.face(1,-1,0.rho_flux))/c.dy
    c.rho_t -= (c.face(0,0,1).rho_flux -
                c.face(0,0,-1).rho_flux)/c.dz
    c.rho_velocity_t -= (c.face(1,0,0).rho_velocity_flux -
                         c.face(-1,0,0).rho_velocity_flux)/c.dx
    c.rho_velocity_t -= (c.face(0,1,0).rho_velocity_flux -
                         c.face(1,-1,0).rho_velocity_flux)/c.dy
    c.rho_velocity_t -= (c.face(0,0,1).rho_velocity_flux -
                         c.face(0,0,-1).rho_velocity_flux)/c.dz
    c.rho_energy_t -= (c.face(1,0,0).rho_energy_flux -
                       c.face(-1,0,0).rho_energy_flux)/c.dx
    c.rho_energy_t -= (c.face(0,1,0).rho_energy_flux -
                       c.face(1,-1,0).rho_energy_flux)/c.dy
    c.rho_energy_t -= (c.face(0,0,1).rho_energy_flux -
                       c.face(0,0,-1).rho_energy_flux)/c.dz
end


-- Update particle fields based on flow fields
local AddFlowCouplingPartOne = liszt kernel(p: particles)
    var dc = p.dual_cell
    var flow_density     = InterpolateBilinear(dc, Rho)
    var flow_velocity    = InterpolateBilinear(dc, Velocity)
    var flow_temperature = InterpolateBilineat(dc, Temperature)
    var flow_dyn_viscosity = GetDynamicViscosity(flow_temperature)
    p.position_t    += p.velocity
    var relaxation_time = p.density * cmath.pow(p.diameter, 2) /
        (18.0 * flow_dyn_viscosity)
    p.delta_velocity_over_relaxation_time = (flow_velocity - p.velocity)/
        relaxation_time
    var particle_mass = 3.14 * p.density * cmath.pow(p.diameter, 3) / 6.0
    p.delta_temperature_term = 3.14 * cmath.pow(p.diameter, 2) *
        particle_options.convective_coefficient *
        (flow_temperature - p.temperature)
end


local AddFlowCouplingPartTwo = liszt kernel(p : particles)
    p.velocity_t += p.delta_velocity_over_relaxation_time
    p.temperature_t += p.delta_temperature_term/
        (particle_mass * particle_options.heat_capacity)
end


-- Update flow variables using derivatives
-- (intermediate steps in a single time step)
local UpdateFlowFieldsIntermediate = liszt kernel(c : grid.cells)
end


-- Update flow variables using derivatives
-- (last step in a single time step)
local UpdateFlowFieldsFinal = liszt kernel(c : grid.cells)
end


-- Update particle variables using derivatives
-- (intermediate steps in a single time step)
local UpdateParticleFieldsIntermediate = liszt kernel(p : particles)
end


-- Update flow variables using derivatives
-- (last step in a single time step)
local UpdateParticleFieldsFinal = liszt kernel(p : particles)
end


local UpdateAuxiliaryGridVelocity = liszt kernel(c : grid.cells)
    c.velocity = c.rho_velocity / c.rho
end


local UpdateAuxiliaryGridThermodynamics = liszt kernel(c : grid.cells)
end


local UpdateAuxiliaryParticles = liszt kernel(p : particles)
end


-----------------------------------------------------------------------------
--[[                             MAIN LOOP                               ]]--
-----------------------------------------------------------------------------


