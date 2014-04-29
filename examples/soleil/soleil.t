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
--local vdb   = L.require 'lib.vdb'


-----------------------------------------------------------------------------
--[[                             OPTIONS                                 ]]--
-----------------------------------------------------------------------------


local grid_options = {
    xnum = 32,
    ynum = 32,
    pos = {0.0, 0.0},
    width = 6.28,
    height = 6.28
}


local particle_options = {
    num = 100,
    convective_coefficient = L.NewGlobal(L.double, 0.7),
    heat_capacity = L.NewGlobal(L.double, 0.7),
    pos_max = 6.2,
    temperature = 20,
    density = 1000,
    diameter_m = 0.01,
    diameter_a = 0.001
}


local spatial_stencil = {
--    order = 6,
--    size = 6,
--    num_interpolate_coeffs = 4,
--    interpolate_coeffs = L.NewVector(L.double, {0, 37/60, -8/60, 1/60}),
    order = 2,
    size = 2,
    num_interpolate_coeffs = 2,
    interpolate_coeffs = L.NewVector(L.double, {0, 0.5}),
    split = 0.5
}


local time_integrator = {
    coeff_function = {1/6, 1/3, 1/3, 1/6},
    coeff_time     = {0.5, 0.5, 1, 1},
    sim_time = 0,
    final_time = 1.0,
    time_step = 0
}


local fluid_options = {
    -- Question: When should I use L.NewGlobal instead?
    gasConstant = 20.4128,
    gamma = 1.4,
    cv = 51.32,
    gammaMinus1 = 0.4,
    dynamic_viscosity_ref = L.NewGlobal(L.double, 0.00044),
    dynamic_viscosity_temp_ref = L.NewGlobal(L.double, 1.0),
    prandtl = 0.7
}


local flow_options = {
    rho = 1.0,
}


-----------------------------------------------------------------------------
--[[                       FLOW/ PARTICLE RELATIONS                      ]]--
-----------------------------------------------------------------------------


-- Declare and initialize grid and related fields

local bnum = spatial_stencil.order/2
local bw   = grid_options.width/grid_options.xnum * bnum
local newPos = grid_options.pos
newPos[1] = newPos[1] - bnum * grid_options.width/grid_options.xnum
newPos[2] = newPos[2] - bnum * grid_options.height/grid_options.xnum
print(newPos[1],newPos[2])
local grid = Grid.New2dUniformGrid(grid_options.xnum + 2*bnum,
                                   grid_options.ynum + 2*bnum,
                                   newPos,
                                   grid_options.width + 2*bw,
                                   grid_options.height + 2*bw,
                                   bnum)
-- conserved variables
grid.cells:NewField('rho', L.double):
LoadConstant(flow_options.rho)
grid.cells:NewField('rhoVelocity', L.vec2d):
LoadConstant(L.NewVector(L.double, {0, 0}))
grid.cells:NewField('rhoEnergy', L.double):
LoadConstant(0)

-- primitive variables
grid.cells:NewField('centerCoordinates', L.vec2d):
LoadConstant(L.NewVector(L.double, {0, 0}))
grid.cells:NewField('velocity', L.vec2d):
LoadConstant(L.NewVector(L.double, {0, 0}))
grid.cells:NewField('temperature', L.double):
LoadConstant(0)
grid.cells:NewField('pressure', L.double):
LoadConstant(0)
grid.cells:NewField('rhoEnthalpy', L.double):
LoadConstant(0)
grid.cells:NewField('sgsEnergy', L.double):
LoadConstant(0)
grid.cells:NewField('typeFlag', L.double):
LoadConstant(0)

-- fields for boundary treatment (should be removed once once I figure out how
-- to READ/WRITE the same field within the same kernel safely
grid.cells:NewField('rhoBoundary', L.double):
LoadConstant(flow_options.rho)
grid.cells:NewField('rhoVelocityBoundary', L.vec2d):
LoadConstant(L.NewVector(L.double, {0, 0}))
grid.cells:NewField('rhoEnergyBoundary', L.double):
LoadConstant(0)
grid.cells:NewField('pressureBoundary', L.double):
LoadConstant(0)
grid.cells:NewField('temperatureBoundary', L.double):
LoadConstant(0)

-- scratch (temporary) fields
-- intermediate value and copies
grid.cells:NewField('rho_copy', L.double):
LoadConstant(0)
grid.cells:NewField('rhoVelocity_copy', L.vec2d):
LoadConstant(L.NewVector(L.double, {0, 0}))
grid.cells:NewField('rhoEnergy_copy', L.double):
LoadConstant(0)
grid.cells:NewField('rho_temp', L.double):
LoadConstant(0)
grid.cells:NewField('rhoVelocity_temp', L.vec2d):
LoadConstant(L.NewVector(L.double, {0, 0}))
grid.cells:NewField('rhoEnergy_temp', L.double):
LoadConstant(0)
-- time derivatives
grid.cells:NewField('rho_t', L.double):
LoadConstant(0)
grid.cells:NewField('rhoVelocity_t', L.vec2d):
LoadConstant(L.NewVector(L.double, {0, 0}))
grid.cells:NewField('rhoEnergy_t', L.double):
LoadConstant(0)
-- flux
-- TODO: Define edge and related macros
grid.x_edges:NewField('rhoFlux', L.double):
LoadConstant(0)
grid.x_edges:NewField('rhoVelocityFlux', L.vec2d):
LoadConstant(L.NewVector(L.double, {0, 0}))
grid.x_edges:NewField('rhoEnergyFlux', L.double):
LoadConstant(0)
grid.x_edges:NewField('rhoEnthalpy', L.double):
LoadConstant(0)
grid.y_edges:NewField('rhoFlux', L.double):
LoadConstant(0)
grid.y_edges:NewField('rhoVelocityFlux', L.vec2d):
LoadConstant(L.NewVector(L.double, {0, 0}))
grid.y_edges:NewField('rhoEnergyFlux', L.double):
LoadConstant(0)
grid.y_edges:NewField('rhoEnthalpy', L.double):
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

local InterpolateBilinear = L.NewMacro(function(dc, xy, Field)
    return liszt quote
        var cdl = dc.downleft
        var cul = dc.upleft
        var cdr = dc.downright
        var cur = dc.upright
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
local InitializeCenterCoordinates= liszt kernel(c : grid.cells)
    var xy = c.center
    c.centerCoordinates = L.vec2d({xy[0], xy[1]})
end
local InitializeFlowPrimitives = liszt kernel(c : grid.cells)
    if  c.in_interior then
        -- Define Taylor Green Vortex
        var taylorGreenPressure = 100.0
        -- Initialize
        var xy = c.center
        var sx = cmath.sin(xy[0])
        var sy = cmath.sin(xy[1])
        var cx = cmath.cos(xy[0])
        var cy = cmath.cos(xy[1])
        var coorZ = 0
        c.velocity = L.vec2d({sy, cx})
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
        c.typeFlag = 0
    end
    if c.is_left_bnd then
        c.typeFlag = 1
    end
    if c.is_right_bnd then
        c.typeFlag = 2
    end
    if c.is_down_bnd then
        c.typeFlag = 3
    end
    if c.is_up_bnd then
        c.typeFlag = 4
    end
end
local UpdateConservedFromPrimitive = liszt kernel(c : grid.cells)
    if c.in_interior then
        -- Equation of state: T = p / ( R * rho )
        var tmpTemperature = c.pressure /(fluid_options.gasConstant * c.rho)
        var velocity = c.velocity

        c.rhoVelocity = c.rho * c.velocity

 
        ---- rhoE = rhoe (= rho * cv * T) + kineticEnergy + sgsEnergy
        c.rhoEnergy = 
          c.rho *
          ( fluid_options.cv * tmpTemperature 
            + 0.5 * L.dot(velocity,velocity) )
          + c.sgsEnergy

    end
end

-- Write cells field to output file
local WriteCellsField = function (outputFileNamePrefix,xSize,ySize,field)
   -- Make up complete file name based on name of field
   local outputFileName = outputFileNamePrefix .. "_" ..
                          field['name'] .. ".txt"
   -- Open file
   local outputFile = io.output(outputFileName)
   -- Write data
   local N = field.owner._size
   if (field.type:isVector()) then
       io.write("# ", xSize, " ", ySize, " ", N, " ", field.type.N, "\n")
       for i = 0, N-1 do
           local s = ''
           for j = 0, field.type.N-1 do
               local t = tostring(field.data[i].d[j]):gsub('ULL',' ')
               s = s .. ' ' .. t .. ''
           end
           io.write("", i, s,"\n")
       end
   else
       io.write("# ", xSize, " ", ySize, " ", N, " ", 1, "\n")
       for i = 0, N-1 do
           local t = tostring(field.data[i]):gsub('ULL', ' ')
           io.write("", i, ' ', t,"\n")
       end
   end
end

-- Write particles field to output file
local WriteParticlesArray = function (outputFileNamePrefix,field)
   -- Make up complete file name based on name of field
   local outputFileName = outputFileNamePrefix .. "_" ..
                          field['name'] .. ".txt"
   -- Open file
   local outputFile = io.output(outputFileName)
   -- Write data
   local N = field.owner._size
   if (field.type:isVector()) then
       io.write("# ", N, " ", field.type.N, "\n")
       for i = 0, N-1 do
           local s = ''
           for j = 0, field.type.N-1 do
               local t = tostring(field.data[i].d[j]):gsub('ULL',' ')
               s = s .. ' ' .. t .. ''
           end
           io.write("", i, s,"\n")
       end
   else
       io.write("# ", N, " ", 1, "\n")
       for i = 0, N-1 do
           local t = tostring(field.data[i]):gsub('ULL', ' ')
           io.write("", i, ' ', t,"\n")
       end
   end
end


-- Initialize temporaries
local InitializeFlowTemporaries = liszt kernel(c : grid.cells)
    c.rho_copy = c.rho
    c.rhoVelocity_copy = c.rhoVelocity
    c.rhoEnergy_copy   = c.rhoEnergy
    c.rho_temp = c.rho
    c.rhoVelocity_temp = c.rhoVelocity
    c.rhoEnergy_temp   = c.rhoEnergy
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
    c.rhoVelocity_t = L.vec2d({0, 0})
    c.rhoEnergy_t = L.double(0)
end
local InitializeParticleDerivatives = liszt kernel(p : particles)
    p.position_t = L.vec2d({0, 0})
    p.velocity_t = L.vec2d({0, 0})
    p.temperature_t = L.double(0)
end

-----------
-- Inviscid
-----------

-- Initialize enthalpy and derivatives
local AddInviscidInitialize = liszt kernel(c : grid.cells)
    c.rhoEnthalpy = c.rhoEnergy + c.pressure
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
            var rhoVelocity_diagonal = L.vec2d({0.0, 0.0})
            var rhoVelocity_skew     = L.vec2d({0.0, 0.0})
            var rhoEnergy_diagonal   = L.double(0.0)
            var rhoEnergy_skew       = L.double(0.0)
            var epdiag = L.double(0.0)
            var axis = e.axis
            for i = 0, num_coeffs do
                rho_diagonal += coeffs[i] *
                              ( e.cell_next.rho *
                                e.cell_next.velocity[axis] +
                                e.cell_previous.rho *
                                e.cell_previous.velocity[axis] )
                rhoVelocity_diagonal += coeffs[i] *
                                       ( e.cell_next.rhoVelocity *
                                         e.cell_next.velocity[axis] +
                                         e.cell_previous.rhoVelocity *
                                         e.cell_previous.velocity[axis] )
                rhoEnergy_diagonal += coeffs[i] *
                                     ( e.cell_next.rhoEnthalpy *
                                       e.cell_next.velocity[axis] +
                                       e.cell_previous.rhoEnthalpy *
                                       e.cell_previous.velocity[axis] )
                epdiag += coeffs[i] *
                        ( e.cell_next.pressure +
                          e.cell_previous.pressure )
            end
            -- TODO: I couldnt understand how skew is implemented. Setting s =
            -- 1 for now, this should be changed.
            var s = spatial_stencil.split
            s = 1
            e.rhoFlux          = s * rho_diagonal +
                                  (1-s) * rho_skew
            e.rhoVelocityFlux = s * rhoVelocity_diagonal +
                                  (1-s) * rhoVelocity_skew
            e.rhoEnergyFlux   = s * rhoEnergy_diagonal +
                                  (1-s) * rhoEnergy_skew
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
        c.rho_t -= (c.edge_right.rhoFlux -
                    c.edge_left.rhoFlux)/c_dx
        c.rho_t -= (c.edge_up.rhoFlux -
                    c.edge_down.rhoFlux)/c_dy
        c.rhoVelocity_t -= (c.edge_right.rhoVelocityFlux -
                             c.edge_left.rhoVelocityFlux)/c_dx
        c.rhoVelocity_t -= (c.edge_up.rhoVelocityFlux -
                             c.edge_down.rhoVelocityFlux)/c_dy
        c.rhoEnergy_t -= (c.edge_right.rhoEnergyFlux -
                           c.edge_left.rhoEnergyFlux)/c_dx
        c.rhoEnergy_t -= (c.edge_up.rhoEnergyFlux -
                           c.edge_down.rhoEnergyFlux)/c_dy
    end
end



-------------
---- Viscous
-------------
--
---- Initialize viscous
--local AddViscousInitialize = liszt kernel(c : grid.cells)
--end
--
--
---- Interpolation for flux values
---- read conserved variables, write flux variables
--local AddViscousGetFlux = {}
--local function GenerateAddViscousGetFlux(edges)
--    return liszt kernel(e : edges)
--        if e.in_interior then
--            var num_coeffs = spatial_stencil.num_interpolate_coeffs
--            var coeffs     = spatial_stencil.interpolate_coeffs
--            var rho_diagonal = L.double(0)
--            var rho_skew     = L.double(0)
--            var rhoVelocity_diagonal = L.vec2d({0.0, 0.0})
--            var rhoVelocity_skew     = L.vec2d({0.0, 0.0})
--            var rhoEnergy_diagonal   = L.double(0.0)
--            var rhoEnergy_skew       = L.double(0.0)
--            var epdiag = L.double(0.0)
--            var axis = e.axis
--            for i = 0, num_coeffs do
--                rho_diagonal += coeffs[i] *
--                              ( e.cell_next.rho *
--                                e.cell_next.velocity[axis] +
--                                e.cell_previous.rho *
--                                e.cell_previous.velocity[axis] )
--                rhoVelocity_diagonal += coeffs[i] *
--                                       ( e.cell_next.rhoVelocity *
--                                         e.cell_next.velocity[axis] +
--                                         e.cell_previous.rhoVelocity *
--                                         e.cell_previous.velocity[axis] )
--                rhoEnergy_diagonal += coeffs[i] *
--                                     ( e.cell_next.rhoEnthalpy *
--                                       e.cell_next.velocity[axis] +
--                                       e.cell_previous.rhoEnthalpy *
--                                       e.cell_previous.velocity[axis] )
--                epdiag += coeffs[i] *
--                        ( e.cell_next.pressure +
--                          e.cell_previous.pressure )
--            end
--            -- TODO: I couldnt understand how skew is implemented. Setting s =
--            -- 1 for now, this should be changed.
--            var s = spatial_stencil.split
--            s = 1
--            e.rhoFlux          = s * rho_diagonal +
--                                  (1-s) * rho_skew
--            e.rhoVelocityFlux = s * rhoVelocity_diagonal +
--                                  (1-s) * rhoVelocity_skew
--            e.rhoEnergyFlux   = s * rhoEnergy_diagonal +
--                                  (1-s) * rhoEnergy_skew
--        end
--    end
--end
--AddViscousGetFlux.X = GenerateAddViscousGetFlux(grid.x_edges)
--AddViscousGetFlux.Y = GenerateAddViscousGetFlux(grid.y_edges)
--
---- Update conserved variables using flux values from previous part
---- write conserved variables, read flux variables
--local c_dx = L.NewGlobal(L.double, grid:cellWidth())
--local c_dy = L.NewGlobal(L.double, grid:cellHeight())
--local AddViscousUpdateUsingFlux = liszt kernel(c : grid.cells)
--    if c.in_interior then
--        c.rho_t -= (c.edge_right.rhoFlux -
--                    c.edge_left.rhoFlux)/c_dx
--        c.rho_t -= (c.edge_up.rhoFlux -
--                    c.edge_down.rhoFlux)/c_dy
--        c.rhoVelocity_t -= (c.edge_right.rhoVelocityFlux -
--                             c.edge_left.rhoVelocityFlux)/c_dx
--        c.rhoVelocity_t -= (c.edge_up.rhoVelocityFlux -
--                             c.edge_down.rhoVelocityFlux)/c_dy
--        c.rhoEnergy_t -= (c.edge_right.rhoEnergyFlux -
--                           c.edge_left.rhoEnergyFlux)/c_dx
--        c.rhoEnergy_t -= (c.edge_up.rhoEnergyFlux -
--                           c.edge_down.rhoEnergyFlux)/c_dy
--    end
--end


-- Update particle fields based on flow fields
local AddFlowCouplingPartOne = liszt kernel(p: particles)
    var dc = p.dual_cell
    var flow_density     = L.double(0)
    var flow_velocity    = L.vec2d({0, 0})
    var flow_temperature = L.double(0)
    var flow_dyn_viscosity = L.double(0)
    var pos = p.position
    flow_density     = InterpolateBilinear(dc, pos, Rho)
    flow_velocity    = InterpolateBilinear(dc, pos, Velocity)
    flow_temperature = InterpolateBilinear(dc, pos, Temperature)
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

-- Set particle velocities to flow for initialization
local SetParticleVelocitiesToFlow = liszt kernel(p: particles)
    var dc = p.dual_cell
    var flow_density     = L.double(0)
    var flow_velocity    = L.vec2d({0, 0})
    var flow_temperature = L.double(0)
    var flow_dyn_viscosity = L.double(0)
    var pos = p.position
    flow_velocity    = InterpolateBilinear(dc, pos, Velocity)
    p.velocity = flow_velocity
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
            r.rhoVelocity_temp += coeff_fun * delta_time * r.rhoVelocity_t
            r.rhoVelocity       = r.rhoVelocity_copy +
                                   coeff_time * delta_time * r.rhoVelocity_t
            r.rhoEnergy_temp += coeff_fun * delta_time * r.rhoEnergy_t
            r.rhoEnergy       = r.rhoEnergy_copy +
                                 coeff_time * delta_time * r.rhoEnergy_t
        end
    elseif stage == 4 then
        return liszt kernel(r : relation)
            r.rho = r.rho_temp +
                    coeff_fun * delta_time * r.rho_t
            r.rhoVelocity = r.rhoVelocity_temp +
                             coeff_fun * delta_time * r.rhoVelocity_t
            r.rhoEnergy = r.rhoEnergy_temp +
                           coeff_fun * delta_time * r.rhoEnergy_t
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


local UpdateAuxiliaryFlowVelocity = liszt kernel(c : grid.cells)
    if c.in_interior then
        c.velocity = c.rhoVelocity / c.rho
    end
end

local UpdateGhostFlowFieldsStep1 = liszt kernel(c : grid.cells)
    if c.is_left_bnd then
        c.rhoBoundary            =   c(1,0).rho
        c.rhoVelocityBoundary[0] = - c(1,0).rhoVelocity[0]
        c.rhoVelocityBoundary[1] =   c(1,0).rhoVelocity[1]
        c.rhoEnergyBoundary      =   c(1,0).rhoEnergy
        c.pressureBoundary       =   c(1,0).pressure
        c.temperatureBoundary    =   c(1,0).temperature
    end
    if c.is_right_bnd then
        c.rhoBoundary            =   c(-1,0).rho
        c.rhoVelocityBoundary[0] = - c(-1,0).rhoVelocity[0]
        c.rhoVelocityBoundary[1] =   c(-1,0).rhoVelocity[1]
        c.rhoEnergyBoundary      =   c(-1,0).rhoEnergy
        c.pressureBoundary       =   c(-1,0).pressure
        c.temperatureBoundary    =   c(-1,0).temperature
    end
    if c.is_down_bnd then
        c.rhoBoundary            =   c(0,1).rho
        c.rhoVelocityBoundary[0] =   c(0,1).rhoVelocity[0]
        c.rhoVelocityBoundary[1] = - c(0,1).rhoVelocity[1]
        c.rhoEnergyBoundary      =   c(0,1).rhoEnergy
        c.pressureBoundary       =   c(0,1).pressure
        c.temperatureBoundary    =   c(0,1).temperature
    end
    if c.is_up_bnd then
        c.rhoBoundary            =   c(0,-1).rho
        c.rhoVelocityBoundary[0] =   c(0,-1).rhoVelocity[0]
        c.rhoVelocityBoundary[1] = - c(0,-1).rhoVelocity[1]
        c.rhoEnergyBoundary      =   c(0,-1).rhoEnergy
        c.pressureBoundary       =   c(0,-1).pressure
        c.temperatureBoundary    =   c(0,-1).temperature
    end
    L.print(L.id(c))
    -- Corners
    if c.is_left_bnd and c.is_down_bnd then
        c.rhoBoundary            =   c(1,1).rho
        c.rhoVelocityBoundary[0] = - c(1,1).rhoVelocity[0]
        c.rhoVelocityBoundary[1] = - c(1,1).rhoVelocity[1]
        c.rhoEnergyBoundary      =   c(1,1).rhoEnergy
        c.pressureBoundary       =   c(1,1).pressure
        c.temperatureBoundary    =   c(1,1).temperature
    end
    if c.is_left_bnd and c.is_up_bnd then
        c.rhoBoundary            =   c(1,-1).rho
        c.rhoVelocityBoundary[0] = - c(1,-1).rhoVelocity[0]
        c.rhoVelocityBoundary[1] = - c(1,-1).rhoVelocity[1]
        c.rhoEnergyBoundary      =   c(1,-1).rhoEnergy
        c.pressureBoundary       =   c(1,-1).pressure
        c.temperatureBoundary    =   c(1,-1).temperature
    end
    if c.is_right_bnd and c.is_down_bnd then
        c.rhoBoundary            =   c(-1,1).rho
        c.rhoVelocityBoundary[0] = - c(-1,1).rhoVelocity[0]
        c.rhoVelocityBoundary[1] = - c(-1,1).rhoVelocity[1]
        c.rhoEnergyBoundary      =   c(-1,1).rhoEnergy
        c.pressureBoundary       =   c(-1,1).pressure
        c.temperatureBoundary    =   c(-1,1).temperature
    end
    if c.is_right_bnd and c.is_up_bnd then
        c.rhoBoundary            =   c(-1,-1).rho
        c.rhoVelocityBoundary[0] = - c(-1,-1).rhoVelocity[0]
        c.rhoVelocityBoundary[1] = - c(-1,-1).rhoVelocity[1]
        c.rhoEnergyBoundary      =   c(-1,-1).rhoEnergy
        c.pressureBoundary       =   c(-1,-1).pressure
        c.temperatureBoundary    =   c(-1,-1).temperature
    end
end

local UpdateGhostFlowFieldsStep2 = liszt kernel(c : grid.cells)
    if c.is_bnd then
        c.pressure    = c.pressureBoundary
        c.rho         = c.rhoBoundary
        c.rhoVelocity = c.rhoVelocityBoundary
        c.rhoEnergy   = c.rhoEnergyBoundary
        c.pressure    = c.pressureBoundary
        c.temperature = c.temperatureBoundary
    end
end


local UpdateAuxiliaryFlowThermodynamics = liszt kernel(c : grid.cells)
    if c.in_interior then
        var kineticEnergy = 
          0.5 * c.rho * L.dot(c.velocity,c.velocity)
        -- Define temporary pressure variable to avoid error like this:
        -- Errors during typechecking liszt
        -- examples/soleil/soleil.t:557: access of 'cells.pressure' field in <Read> phase
        -- conflicts with earlier access in <Write> phase at examples/soleil/soleil.t:555
        -- when I try to reuse the c.pressure variable to calculate the temperature
        var pressure = fluid_options.gammaMinus1 * ( c.rhoEnergy - kineticEnergy )
        c.pressure = pressure 
        c.temperature =  pressure / ( fluid_options.gasConstant * c.rho)
    end
end


local UpdateAuxiliaryParticles = liszt kernel(p : particles)
end


-- kernels to draw particles and velocity for debugging purpose

--local DrawParticlesKernel = liszt kernel (p : particles)
--    var color = {1.0,1.0,0.0}
--    vdb.color(color)
--    var pmax = L.double(particle_options.pos_max)
--    var pos : L.vec3d = { p.position[0]/pmax,
--                          p.position[1]/pmax,
--                          0.0 }
--    vdb.point(pos)
--    --var vel = p.velocity
--    --var v = L.vec3d({ vel[0], vel[1], 0.0 })
--    --v = 200 * v + L.vec3d({2, 2, 0})
--    --vdb.line(pos, pos+v)
--end


-----------------------------------------------------------------------------
--[[                             MAIN LOOP                               ]]--
-----------------------------------------------------------------------------



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

--local function AddViscous()
--    --AddViscousInitialize(grid.cells)
--    AddViscousGetFlux.X(grid.x_edges)
--    AddViscousGetFlux.Y(grid.y_edges)
--    AddViscousUpdateUsingFlux(grid.cells)
--end


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


local function UpdateAuxiliary()
    UpdateAuxiliaryFlowVelocity(grid.cells)
    UpdateAuxiliaryFlowThermodynamics(grid.cells)
end

local function UpdateGhost()
    UpdateGhostFlowFieldsStep1(grid.cells)
    UpdateGhostFlowFieldsStep2(grid.cells)
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


local function WriteOutput()
    outputFileName = "soleilOutput/output_" ..
      tostring(time_integrator.time_step)
    WriteCellsField(outputFileName .. "_flow",
      grid:xSize(),grid:ySize(),grid.cells.typeFlag)
    WriteCellsField(outputFileName .. "_flow",
      grid:xSize(),grid:ySize(),grid.cells.temperature)
    WriteCellsField(outputFileName .. "_flow",
      grid:xSize(),grid:ySize(),grid.cells.rho)
    WriteCellsField(outputFileName .. "_flow",
      grid:xSize(),grid:ySize(),grid.cells.pressure)
    WriteParticlesArray(outputFileName .. "_particles",
      particles.position)
    WriteParticlesArray(outputFileName .. "_particles",
      particles.velocity)
    WriteParticlesArray(outputFileName .. "_particles",
      particles.temperature)
end
--local function DrawParticles()
--    vdb.vbegin()
--    vdb.frame()
--    DrawParticlesKernel(particles)
--    vdb.vend()
--end

local function InitializeVariables()
    InitializeCenterCoordinates(grid.cells)
    InitializeFlowPrimitives(grid.cells)
    UpdateConservedFromPrimitive(grid.cells)
    UpdateAuxiliary()
    UpdateGhost()
    LocateParticles(particles)
    SetParticleVelocitiesToFlow(particles)
end


InitializeVariables()

outputFileName = "soleilOutput/output"
WriteCellsField(outputFileName,grid:xSize(),grid:ySize(),grid.cells.centerCoordinates)
WriteParticlesArray(outputFileName .. "_particles",
  particles.diameter)
while (time_integrator.sim_time < time_integrator.final_time) do
    print("Running time step ", time_integrator.time_step, ", time",
          time_integrator.sim_time) 
    WriteOutput(time_integrator.time_step)
    InitializeTemporaries()
    for stage = 1, 4 do
        InitializeDerivatives()
        AddInviscid()
        --AddViscous()
        AddFlowCoupling()
        UpdateFlow(stage)
        UpdateParticles(stage)
        UpdateTime(stage)
        UpdateAuxiliary()
    end
--    DrawParticles()
end

--particles.position:print()
--particles.velocity:print()
--grid.cells.velocity:print()
--grid.cells.temperature:print()
--particles.position:print()
----particles.velocity:print()
