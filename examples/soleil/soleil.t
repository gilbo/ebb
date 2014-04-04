import "compiler.liszt"
local Grid  = terralib.require 'compiler.grid'
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
local vdb   = terralib.require 'compiler.vdb'


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
    num = 50,
    convective_coefficient = L.NewGlobal(L.double, 0.7),
    heat_capacity = L.NewGlobal(L.double, 0.7),
    pos_max = 6.2,
    temperature = 20,
    density = 1000,
    diameter_m = 0.01,
    diameter_a = 0.001
}


local spatial_stencil = {
    order = 6,
    size = 6,
    num_interpolate_coeffs = 4,
    interpolate_coeffs = L.NewVector(L.double, {0, 37/60, -8/60, 1/60}),
    split = 0.5
}


local time_integrator = {
    coeff_function = {1/6, 1/3, 1/3, 1/6},
    coeff_time     = {0.5, 0.5, 1, 1},
    sim_time = 0,
    final_time = 0.1,
    time_step = 0
}


local fluid_options = {
    dynamic_viscosity_ref = L.NewGlobal(L.double, 0.00044),
    dynamic_viscosity_temp_ref = L.NewGlobal(L.double, 1.0)
}


local flow_options = {
    rho = 1.0,
}


-----------------------------------------------------------------------------
--[[                       FLOW/ PARTICLE RELATIONS                      ]]--
-----------------------------------------------------------------------------


-- Declare and initialize grid  and related fields

local grid = Grid.New2dUniformGrid(grid_options.xnum, grid_options.ynum,
                                   grid_options.pos,
                                   grid_options.width, grid_options.height,
                                   spatial_stencil.order/2)

-- conserved variables
grid.cells:NewField('rho', L.double):
LoadConstant(flow_options.rho)
grid.cells:NewField('rho_velocity', L.vec2d):
LoadConstant(L.NewVector(L.double, {0, 0}))
grid.cells:NewField('rho_energy', L.double):
LoadConstant(0)

-- primitive variables
grid.cells:NewField('velocity', L.vec2d):
LoadConstant(L.NewVector(L.double, {0, 0}))
grid.cells:NewField('temperature', L.double):
LoadConstant(0)
grid.cells:NewField('pressure', L.double):
LoadConstant(0)
grid.cells:NewField('rho_enthalpy', L.double):
LoadConstant(0)

-- scratch (temporary) fields
-- intermediate value and copies
grid.cells:NewField('rho_copy', L.double):
LoadConstant(0)
grid.cells:NewField('rho_velocity_copy', L.vec2d):
LoadConstant(L.NewVector(L.double, {0, 0}))
grid.cells:NewField('rho_energy_copy', L.double):
LoadConstant(0)
grid.cells:NewField('rho_temp', L.double):
LoadConstant(0)
grid.cells:NewField('rho_velocity_temp', L.vec2d):
LoadConstant(L.NewVector(L.double, {0, 0}))
grid.cells:NewField('rho_energy_temp', L.double):
LoadConstant(0)
-- derivatives
grid.cells:NewField('rho_t', L.double):
LoadConstant(0)
grid.cells:NewField('rho_velocity_t', L.vec2d):
LoadConstant(L.NewVector(L.double, {0, 0}))
grid.cells:NewField('rho_energy_t', L.double):
LoadConstant(0)
-- flux
-- TODO: Define edge and related macros
grid.x_edges:NewField('rho_flux', L.double):
LoadConstant(0)
grid.x_edges:NewField('rho_velocity_flux', L.vec2d):
LoadConstant(L.NewVector(L.double, {0, 0}))
grid.x_edges:NewField('rho_energy_flux', L.double):
LoadConstant(0)
grid.x_edges:NewField('rho_enthalpy', L.double):
LoadConstant(0)
grid.y_edges:NewField('rho_flux', L.double):
LoadConstant(0)
grid.y_edges:NewField('rho_velocity_flux', L.vec2d):
LoadConstant(L.NewVector(L.double, {0, 0}))
grid.y_edges:NewField('rho_energy_flux', L.double):
LoadConstant(0)
grid.y_edges:NewField('rho_enthalpy', L.double):
LoadConstant(0)


-- Declare and initialize particle relation and fields over the particle

local particles = L.NewRelation(particle_options.num, 'particles')

particles:NewField('dual_cell', grid.dual_cells):
LoadConstant(0)
particles:NewField('position', L.vec2d):
Load(function(i)
    local pmax = particle_options.pos_max
    local p1 = cmath.fmod(cmath.rand_double(), pmax)
    local p2 = cmath.fmod(cmath.rand_double(), pmax)
    return {p1, p2}
end)
particles:NewField('velocity', L.vec2d):
LoadConstant(L.NewVector(L.double, {0, 0}))
particles:NewField('temperature', L.double):
LoadConstant(particle_options.temperature)

particles:NewField('diameter', L.double):
Load(function(i)
    return cmath.rand_unity() * particle_options.diameter_m +
        particle_options.diameter_a
end)
particles:NewField('density', L.double):
LoadConstant(particle_options.density)

particles:NewField('delta_velocity_over_relaxation_time', L.vec2d):
LoadConstant(L.NewVector(L.double, {0, 0}))
particles:NewField('delta_temperature_term', L.double):
LoadConstant(0)

-- scratch (temporary) fields
-- intermediate values and copies
particles:NewField('position_copy', L.vec2d):
LoadConstant(L.NewVector(L.double, {0, 0}))
particles:NewField('velocity_copy', L.vec2d):
LoadConstant(L.NewVector(L.double, {0, 0}))
particles:NewField('temperature_copy', L.double):
LoadConstant(0)
particles:NewField('position_temp', L.vec2d):
LoadConstant(L.NewVector(L.double, {0, 0}))
particles:NewField('velocity_temp', L.vec2d):
LoadConstant(L.NewVector(L.double, {0, 0}))
particles:NewField('temperature_temp', L.double):
LoadConstant(0)
-- derivatives
particles:NewField('position_t', L.vec2d):
LoadConstant(L.NewVector(L.double, {0, 0}))
particles:NewField('velocity_t', L.vec2d):
LoadConstant(L.NewVector(L.double, {0, 0}))
particles:NewField('temperature_t', L.double):
LoadConstant(0)


-----------------------------------------------------------------------------
--[[                            LISZT GLOBALS                            ]]--
-----------------------------------------------------------------------------


-- If you want to use global variables like scalar values or vectors within
-- a Liszt kernel (variables that are not constants), declare them here.
-- If you use lua variables, Liszt will treat them as constant values.

local delta_time = L.NewGlobal(L.double, 0.02)
local pi = L.NewGlobal(L.double, 3.14)


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

local GetDynamicViscosity = L.NewMacro(function(temperature)
    return liszt `fluid_options.dynamic_viscosity_ref *
        cmath.pow(temperature/fluid_options.dynamic_viscosity_temp_ref, 0.75)
end)


-----------------------------------------------------------------------------
--[[                            LISZT KERNELS                            ]]--
-----------------------------------------------------------------------------


-- Locate particles
local LocateParticles = liszt kernel(p : particles)
    p.dual_cell = grid.dual_locate(p.position)
end

-- Initialize flow variables
local InitializeFlowPrimitivesDummy = liszt kernel(c : grid.cells)
    if  c.in_interior then
        var xy = c.center
        var x = xy[0]
        var y = xy[1]
        c.rho_energy = cmath.sin(x)
        c.rho_velocity[0] = cmath.sin(y)
        c.rho_velocity[1] = cmath.sin(x)
    end
end

-- Initialize temporaries
local InitializeFlowTemporaries = liszt kernel(c : grid.cells)
    c.rho_copy = c.rho
    c.rho_velocity_copy = c.rho_velocity
    c.rho_energy_copy   = c.rho_energy
    c.rho_temp = c.rho
    c.rho_velocity_temp = c.rho_velocity
    c.rho_energy_temp   = c.rho_energy
end
local InitializeParticleTemporaries = liszt kernel(p : particles)
    p.position_copy = p.position
    p.velocity_copy = p.velocity
    p.temperature_copy = p.temperature
    p.position_temp = p.position
    p.velocity_temp = p.velocity
    p.temperature_temp = p.temperature
end

-- Initialize derivatives
local InitializeFlowDerivatives = liszt kernel(c : grid.cells)
    c.rho_t = L.double(0)
    c.rho_velocity_t = L.vec2d({0, 0})
    c.rho_energy_t = L.double(0)
end
local InitializeParticleDerivatives = liszt kernel(p : particles)
    p.position_t = L.vec2d({0, 0})
    p.velocity_t = L.vec2d({0, 0})
    p.temperature_t = L.double(0)
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
        if e.in_interior then
            var num_coeffs = spatial_stencil.num_interpolate_coeffs
            var coeffs     = spatial_stencil.interpolate_coeffs
            var rho_diagonal = L.double(0)
            var rho_skew     = L.double(0)
            var rho_velocity_diagonal = L.vec2d({0.0, 0.0})
            var rho_velocity_skew     = L.vec2d({0.0, 0.0})
            var rho_energy_diagonal   = L.double(0.0)
            var rho_energy_skew       = L.double(0.0)
            var epdiag = L.double(0.0)
            var axis = e.axis
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
            -- TODO: I couldn't understand how skew is implemented. Setting s =
            -- 1 for now, this should be changed.
            var s = spatial_stencil.split
            s = 1
            e.rho_flux          = s * rho_diagonal +
                                  (1-s) * rho_skew
            e.rho_velocity_flux = s * rho_velocity_diagonal +
                                  (1-s) * rho_velocity_skew
            e.rho_energy_flux   = s * rho_energy_diagonal +
                                  (1-s) * rho_energy_skew
        end
    end
end
AddInviscidGetFlux.X = GenerateAddInviscidGetFlux(grid.x_edges)
AddInviscidGetFlux.Y = GenerateAddInviscidGetFlux(grid.y_edges)


-- Update conserved variables using flux values from previous part
-- write conserved variables, read flux variables
local c_dx = L.NewGlobal(L.double, grid:cellWidth())
local c_dy = L.NewGlobal(L.double, grid:cellHeight())
local AddInviscidUpdateUsingFlux = liszt kernel(c : grid.cells)
    if c.in_interior then
        c.rho_t -= (c.edge_right.rho_flux -
                    c.edge_left.rho_flux)/c_dx
        c.rho_t -= (c.edge_up.rho_flux -
                    c.edge_down.rho_flux)/c_dy
        c.rho_velocity_t -= (c.edge_right.rho_velocity_flux -
                             c.edge_left.rho_velocity_flux)/c_dx
        c.rho_velocity_t -= (c.edge_up.rho_velocity_flux -
                             c.edge_down.rho_velocity_flux)/c_dy
        c.rho_energy_t -= (c.edge_right.rho_energy_flux -
                           c.edge_left.rho_energy_flux)/c_dx
        c.rho_energy_t -= (c.edge_up.rho_energy_flux -
                           c.edge_down.rho_energy_flux)/c_dy
    end
end


-- Update particle fields based on flow fields
local AddFlowCouplingPartOne = liszt kernel(p: particles)
    var dc = p.dual_cell
    var flow_density     = L.double(0)
    var flow_velocity    = L.vec2d({0, 0})
    var flow_temperature = L.double(0)
    var flow_dyn_viscosity = L.double(0)
    flow_density     = InterpolateBilinear(dc, Rho)
    flow_velocity    = InterpolateBilinear(dc, Velocity)
    flow_temperature = InterpolateBilinear(dc, Temperature)
    flow_dyn_viscosity = GetDynamicViscosity(flow_temperature)
    p.position_t    += p.velocity
    var relaxation_time = p.density * cmath.pow(p.diameter, 2) /
        (18.0 * flow_dyn_viscosity)
    p.delta_velocity_over_relaxation_time = (flow_velocity - p.velocity)/
        relaxation_time
    p.delta_temperature_term = pi * cmath.pow(p.diameter, 2) *
        particle_options.convective_coefficient *
        (flow_temperature - p.temperature)
end


local AddFlowCouplingPartTwo = liszt kernel(p : particles)
    p.velocity_t += p.delta_velocity_over_relaxation_time
    var particle_mass = pi * p.density * cmath.pow(p.diameter, 3) / 6.0
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
    elseif stage == 4 then
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
    elseif stage == 4 then
        return liszt kernel(r : relation)
            r.position = r.position_temp +
                         coeff_fun * delta_time * r.position_t
            r.velocity = r.velocity_temp +
                         coeff_fun * delta_time * r.velocity_t
            r.temperature = r.temperature_temp +
                            coeff_fun * delta_time * r.temperature_t
        end
    end
end
for i = 1, 4 do
    UpdateParticleKernels[i] = GenerateUpdateParticleKernels(particles, i)
end


local UpdateAuxiliaryGridVelocity = liszt kernel(c : grid.cells)
    c.velocity = c.rho_velocity / c.rho
end


local UpdateAuxiliaryGridThermodynamics = liszt kernel(c : grid.cells)
end


local UpdateAuxiliaryParticles = liszt kernel(p : particles)
end


-- kernels to draw particles and velocity for debugging purpose

local DrawParticlesKernel = liszt kernel (p : particles)
    var color = {1.0,1.0,0.0}
    vdb.color(color)
    var pmax = L.double(particle_options.pos_max)
    var pos : L.vec3d = { p.position[0]/pmax,
                          p.position[1]/pmax,
                          0.0 }
    vdb.point(pos)
    var vel = p.velocity
    --var v = L.vec3d({ vel[0], vel[1], 0.0 })
    --v = 20 * v + {200, 200, 0}
    --vdb.line(pos, pos+v)
end


-----------------------------------------------------------------------------
--[[                             MAIN LOOP                               ]]--
-----------------------------------------------------------------------------


local function InitializeVariables()
    InitializeFlowPrimitivesDummy(grid.cells)
    LocateParticles(particles)
end


local function InitializeTemporaries()
    InitializeFlowTemporaries(grid.cells)
    InitializeParticleTemporaries(particles)
end


local function InitializeDerivatives()
    InitializeFlowDerivatives(grid.cells)
    InitializeParticleDerivatives(particles)
end


local function AddInviscid()
    AddInviscidInitialize(grid.cells)
    AddInviscidGetFlux.X(grid.x_edges)
    AddInviscidGetFlux.Y(grid.y_edges)
    AddInviscidUpdateUsingFlux(grid.cells)
end


local function AddFlowCoupling()
    LocateParticles(particles)
    AddFlowCouplingPartOne(particles)
    AddFlowCouplingPartTwo(particles)
end

local function UpdateFlow(i)
    UpdateFlowKernels[i](grid.cells)
end


local function UpdateParticles(i)
    UpdateParticleKernels[i](particles)
end


-- Update time function
local function UpdateTime(stage)
    time_integrator.sim_time = time_integrator.sim_time +
                               time_integrator.coeff_time[stage] *
                               delta_time:value()
    if stage == 4 then
        time_integrator.time_step = time_integrator.time_step + 1
    end
end


local function DrawParticles()
    vdb.vbegin()
    vdb.frame()
    DrawParticlesKernel(particles)
    vdb.vend()
end


InitializeVariables()
--particles.position:print()
--particles.dual_cell:print()

while (time_integrator.sim_time < time_integrator.final_time) do
    print("Running time step ", time_integrator.time_step) 
    InitializeTemporaries()
    for stage = 1, 4 do
        InitializeDerivatives()
        AddInviscid()
        AddFlowCoupling()
        UpdateFlow(stage)
        UpdateParticles(stage)
        UpdateTime(stage)
    end
    DrawParticles()
    particles.position:print()
    particles.velocity:print()
end

--particles.position:print()
--particles.diameter:print()
