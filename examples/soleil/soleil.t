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
-- Grid offers two relations, cells and dual cells
-- Fields are declared over cells here

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

-- gradients of primitive variables
grid.cells:NewField('velocity_gradient_x', L.vec2f)
grid.cells:NewField('velocity_gradient_y', L.vec2f)

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
-- TODO: Define face and vertex macros
grid.faces:NewField('rho_flux', L.float)
grid.faces:NewField('rho_velocity_flux', L.vec2f)
grid.faces:NewField('rho_energy_flux', L.float)
grid.faces:NewField('rho_enthalpy', L.float)


-- Declare and initialize particle relation and fields over the particle

local particles = L.NewRelation(particle_options.num, 'particles')

particles:NewField('position', L.vec2f)
particles:NewField('velocity', L.vec2f)
particles:NewField('temperature', L.float)

particles:NewField('diameter', L.float)
particles:NewField('density', L.float)

particles:NewField('deltaVelocity_over_relaxaion_time', L.vec2f)
particles:NewField('delta_temperature_term', L.float)


-----------------------------------------------------------------------------
--[[                            LISZT GLOBALS                            ]]--
-----------------------------------------------------------------------------


-- If you want to use global variables like scalar values or vectors within
-- a Liszt kernel (variables that are not constants), declare them here.
-- If you use lua variables, Liszt will treat them as constant values.


-----------------------------------------------------------------------------
--[[                             LISZT MACROS                            ]]--
-----------------------------------------------------------------------------


-----------------------------------------------------------------------------
--[[                            LISZT KERNELS                            ]]--
-----------------------------------------------------------------------------


-- Initialize enthalpy
local AddInviscidInitialize = liszt kernel(c : grid.cells)
    c.rho_enthalpy = c.rho_energy + c.pressure
end


-- Interpolation for flux values
-- read conserved variables, write flux variables
-- TODO:
-- 1. discuss if to allow lookups based on variables that are
-- known at compile time?
-- 2. discuss indexing lua tables
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
    for i = 0, num_coeffs do
        rho_diagonal += coeffs[i] *
                      ( f.cell(i+1).rho *
                        f.cell(i+1).velocity[0] +
                        f.cell(i-1).rho *
                        f.cell(i-1).velocity[0] )
        rho_velocity_diagonal += coeffs[i] *
                               ( f.cell(i+1).rho_velocity *
                                 f.cell(i+1).velocity[0] +
                                 f.cell(i-1).rho_velocity *
                                 f.cell(i-1).velocity[0] )
        rho_energy_diagonal += coeffs[i] *
                             ( f.cell(i+1).rho_enthalpy *
                               f.cell(i+1).velocity[0] +
                               f.cell(i-1).rho_enthalpy *
                               f.cell(i-1).velocity[0] )
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
local AddFlowCoupling = liszt kernel(p: particles)
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
end


local UpdateAuxiliaryGridThermodynamics = liszt kernel(c : grid.cells)
end


local UpdateAuxiliaryParticles = liszt kernel(p : particles)
end


-----------------------------------------------------------------------------
--[[                             MAIN LOOP                               ]]--
-----------------------------------------------------------------------------


