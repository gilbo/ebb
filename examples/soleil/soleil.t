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


local spatial_stencil = {
}


local time_integrator = {
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
-- intermediate value and copies
grid.cells:NewField('rho_copy', L.float)
grid.cells:NewField('rho_velocity_copy', L.vec2f)
grid.cells:NewField('rho_energy_copy', L.float)
grid.cells:NewField('rho_temp', L.float)
grid.cells:NewField('rho_velocity_temp', L.vec2f)
grid.cells:NewField('rho_energy_temp', L.float)
-- derivatives
grid.cells:NewField('rho_t', L.float)
grid.cells:NewField('rho_velocity_t', L.vec2f)
grid.cells:NewGield('rho_energy_t', L.float)
-- flux
-- TODO: Define edge and related macros
grid.x_edges:NewField('rho_flux', L.float)
grid.x_edges:NewField('rho_velocity_flux', L.vec2f)
grid.x_edges:NewField('rho_energy_flux', L.float)
grid.x_edges:NewField('rho_enthalpy', L.float)
grid.y_edges:NewField('rho_flux', L.float)
grid.y_edges:NewField('rho_velocity_flux', L.vec2f)
grid.y_edges:NewField('rho_energy_flux', L.float)
grid.y_edges:NewField('rho_enthalpy', L.float)


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
-- intermediate values and copies
particles:NewField('position_copy', L.vec2f)
particles:NewField('velocity_copy', L.vec2f)
particles:NewField('temperature_copy', L.float)
particles:NewField('position_temp', L.vec2f)
particles:NewField('velocity_temp', L.vec2f)
particles:NewField('temperature_temp', L.float)
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

local delta_time = L.NewGlobal(L.float, 0.05)


-----------------------------------------------------------------------------
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

local InterpolateBilinear = L.NewMacro(function(dc, Field)
    return liszt quote
        var cdl = dc.downleft
        var cul = dc.upleft
        var cdr = dc.downright
        var cur = dc.upright
        var xy = dc.center
        var delta_l = cmath.abs(cdl.position[1] - xy.y)
        var delta_r = cmath.abs(cur.position[1] - xy.y)
        var f1 = (delta_l*Field(cdl) + delta_r*Field(cul)) / (delta_l + delta_r)
        var f2 = (delta_l*Field(cdr) + delta_r*Field(cur)) / (delta_l + delta_r)
        delta_d = cmath.abs(cdl.position[0] - xy.x)
        delta_u = cmath.abs(cur.position[0] - xy.x)
    in
        (delta_d*f1 + delta_u*f2) / (delta_d + delta_u)
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

-- Initialize enthalpy and derivatives
local AddInviscidInitialize = liszt kernel(c : grid.cells)
    c.rho_enthalpy = c.rho_energy + c.pressure
end


-- Interpolation for flux values
-- read conserved variables, write flux variables
local AddInviscidGetFlux = {}
local function GenerateAddInviscidGetFlux(edges)
    return liszt kernel(e : edges)
        var num_coeffs = spatial_stencil.num_interpolate_coeffs
        var coeffs     = spatial_stencil.interpolate_coeffs
        var rho_diagonal = 0.0
        var rho_skew     = 0.0
        var rho_velocity_diagonal = {0.0, 0.0}
        var rho_velocity_skew     = {0.0, 0.0}
        var rho_energy_diagonal   = 0.0
        var rho_energy_skew       = 0.0
        var epdiag = 0.0
        var axis = f.axis
        for i = 0, num_coeffs do
            rho_diagonal += coeffs[i] *
                          ( e.cell_next.rho *
                            e.cell_next.velocity[axis] +
                            e.cell_previous.rho *
                            e.cell_previous.velocity[axis] )
            rho_velocity_diagonal += coeffs[i] *
                                   ( e.cell_next.rho_velocity *
                                     e.cell_next.velocity[axis] +
                                     e.cell_previous.rho_velocity *
                                     e.cell_previous.velocity[axis] )
            rho_energy_diagonal += coeffs[i] *
                                 ( e.cell_next.rho_enthalpy *
                                   e.cell_next.velocity[axis] +
                                   e.cell_previous.rho_enthalpy *
                                   e.cell_previous.velocity[axis] )
            epdiag += coeffs[i] *
                    ( e.cell_next.pressure +
                      e.cell_previous.pressure )
        end
        -- TODO: I don't understand what skew is. Ask Ivan.
        var s = spatial_stencil.split
        e.rho_flux          = s * rho_diagonal +
                              (1-s) * rho_skew
        e.rho_velocity_flux = s * rho_velocity_diagonal +
                              (1-s) * rho_velocity_skew
        e.rho_energy_flux   = s * rho_energy_diagonal +
                              (1-s) * rho_energy_skew
    end
end
AddInviscidFetFlux.X = GenerateAddInviscidGetFlux(grid.x_edges)
AddInviscidFetFlux.Y = GenerateAddInviscidGetFlux(grid.y_edges)


-- Update conserved variables using flux values from previous part
-- write conserved variables, read flux variables
local c_dx = grid:cellWidth()
local c_dy = grid:cellHeight()
local AddInviscidUpdateUsingFlux = liszt kernel(c : grid.cells)
    c.rho_t -= (c.edge_right.rho_flux -
                c.edge_left.rho_flux)/c.dx
    c.rho_t -= (c.edge_up.rho_flux -
                c.edge_downrho_flux)/c.dy
    c.rho_velocity_t -= (c.edge_right.rho_velocity_flux -
                         c.edge_left.rho_velocity_flux)/c_dx
    c.rho_velocity_t -= (c.edge_up.rho_velocity_flux -
                         c.edge_down.rho_velocity_flux)/c_dy
    c.rho_energy_t -= (c.edge_right.rho_energy_flux -
                       c.edge_left.rho_energy_flux)/c_dx
    c.rho_energy_t -= (c.edge_up.rho_energy_flux -
                       c.edge_down.rho_energy_flux)/c_dy
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
local UpdateFlowKernels = {}
local function GenerateUpdateFlowKernels(relation, stage)
    local coeff_fun  = time_integrator.coeff_function[stage]
    local coeff_time = time_integrator.coeff_time[stage]
    if stage <= 3 then
        return liszt kernel(r : relation)
            r.rho_temp += coeff_fun * delta_time * r.rho_t
            r.rho       = r.rho_copy +
                          coeff_time * delta_time * r.rho_t
            r.rho_velocity_temp += coeff_fun * delta_time * r.rho_velocity_t
            r.rho_velocity       = r.rho_velocity_copy +
                                   coeff_time * delta_time * r.rho_velocity_t
            r.rho_energy_temp += coeff_fun * delta_time * r.rho_energy_t
            r.rho_energy       = r.rho_energy_copy +
                                 coeff_time * delta_time * r.rho_energy_t
        end
    elseif
        return liszt kernel(r : relation)
            r.rho = r.rho_temp +
                    coeff_fun * delta_time * r.rho_t
            r.rho_velocity = r.rho_velocity_temp +
                             coeff_fun * delta_time * r.rho_velocity_t
            r.rho_energy = r.rho_energy_temp +
                           coeff_fun * delta_time * r.rho_energy_t
        end
    end
end
for i = 1, 4 do
    UpdateFlowKernels[i] = GenerateUpdateFlowKernels(grid.cells, i)
end


-- Update particle variables using derivatives
local UpdateParticleKernels = {}
local function GenerateUpdateParticleKernels(relation, stage)
    local coeff_fun  = time_integrator.coeff_function[stage]
    local coeff_time = time_integrator.coeff_time[stage]
    if stage <= 3 then
        return liszt kernel(r : relation)
            r.position_temp += coeff_fun * delta_time * r.position_t
            r.position       = r.position_copy +
                               coeff_time * delta_time * r.position_t
            r.velocity_temp += coeff_fun * delta_time * r.velocity_t
            r.velocity       = r.velocity_copy +
                               coeff_time * delta_time * r.velocity_t
            r.temperature_temp += coeff_fun * delta_time * r.temperature_t
            r.temperature       = r.temperature_copy +
                                  coeff_time * delta_time * r.temperature_t
        end
    elseif
        return liszt kernel(r : relation)
            r.position = r.position_temp +
                         coeff_fun * delta_time * r.position_t
            r.velocity = r.velocity_temp +
                         coeff_fun * delta_time * r.velocity_t
            r.temperature = r.rho_temperature_temp +
                            coeff_fun * delta_time * r.temperature_t
        end
    end
end
for i = 1, 4 do
    UpdateParticleKernels[i] = GenerateUpdateParticleKernels(particles, i)
end


-- Update time function
local function UpdateTime(stage)
    time_integrator.sim_time = time_integrator.sim_time +
                               time_integrator.coeff_time[stage] *
                               delta_time:value()
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


local function InitializeTemporaries()
end


local function InitializeDerivatives()
end


local function AddInviscid()
    AddInviscidInitialize(grid.cells)
    AddInviscidGetFlux.X(grid.x_edges)
    AddInviscidGetFlux.Y(grid.y_edges)
    AddInviscidUpdateUsingFlux(grid.cells)
end


local function AddFlowCoupling()
    AddFlowCouplingPartOne(particles)
    AddFlowCouplingPartTwo(particles)
end

local function UpdateFlow(i)
    UpdateFlowKernels[i](grid.cells)
end


local function UpdateParticles(i)
    UpdateParticleKernels[i](particles)
end


while (time_integrator.sim_time < time_integrator.final_time) do
    InitializeTemporaries()
    for stage = 1, 4 do
        AddInviscid()
        AddFlowCoupling()
        UpdateFlow(i)
        UpdateParticles(i)
    end
end
