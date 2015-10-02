import "ebb"

-- dld for terra callbacks (Tecplot output)
local dld  = require 'ebb.src.dld'

local Grid  = require 'ebb.domains.grid' 
local C = terralib.includecstring [[
#include <math.h> 
#include <stdlib.h> 
#include <time.h>
#include <stdio.h>

double rand_double() {
      double r = (double)rand();
      return r;
}

double rand_unity() {
    double r = (double)rand()/(double)RAND_MAX;
    return r;
}

]]

C.srand(C.time(nil));
local vdb = require 'ebb.lib.vdb'

-- Load the pathname library, which just provides a couple of 
-- convenience functions for manipulating filesystem paths.
local PN = require 'ebb.lib.pathname'

-----------------------------------------------------------------------------
--[[                       LOAD THE CONFIG FILE                          ]]--
-----------------------------------------------------------------------------

-- This location is hard-coded at the moment. The idea is that you will
-- have multiple config files available in other locations, and copy them
-- to this location with the name params.lua before running.

local filename = './devapps/soleil/params.lua'
local config = loadfile(filename)()

--- Immediately check that the output directory exists. Throw an error if not.

local Pathname  = PN.Pathname
local outputdir = Pathname.new(config.outputDirectory)

if not outputdir:exists() then
  outputdir:mkdir()
  print("\nWARNING: The requested output directory does not exist: "..
        config.outputDirectory .. ", \n"..
        "so it has been created. Please check your intended behavior.\n")
end


-----------------------------------------------------------------------------
--[[                            CONSTANT VARIABLES                       ]]--
-----------------------------------------------------------------------------

local pi = 2.0*L.acos(0)
local twoPi = 2.0*pi

-----------------------------------------------------------------------------
--[[                            NAMESPACES                               ]]--
-----------------------------------------------------------------------------

local Flow = {};
local Viscosity = {};
local Particles = {};
local TimeIntegrator = {};
local Statistics = {};
local IO = {};
local Visualization = {};

-----------------------------------------------------------------------------
--[[      Global variables used for specialization within functions      ]]--
-----------------------------------------------------------------------------

-- Flow type
Flow.Uniform             = L.Global(L.int, 0)
Flow.TaylorGreen2DVortex = L.Global(L.int, 1)
Flow.TaylorGreen3DVortex = L.Global(L.int, 2)
Flow.Restart             = L.Global(L.int, 3)
Flow.Perturbed           = L.Global(L.int, 4)

-- Viscosity Model
Viscosity.Constant   = L.Global(L.int, 0)
Viscosity.PowerLaw   = L.Global(L.int, 1)
Viscosity.Sutherland = L.Global(L.int, 2)

-- Particles feeder
Particles.FeederAtStartTimeInRandomBox = L.Global(L.int, 0)
Particles.FeederOverTimeInRandomBox    = L.Global(L.int, 1)
Particles.FeederUQCase                 = L.Global(L.int, 2)
Particles.Random                       = L.Global(L.int, 3)
Particles.Restart                      = L.Global(L.int, 4)

-- Particles collector
Particles.CollectorNone     = L.Global(L.int, 0)
Particles.CollectorOutOfBox = L.Global(L.int, 1)

-- Particle Type (Fixed or Free)
Particles.Fixed = L.Global(L.int, 0)
Particles.Free  = L.Global(L.int, 1)

-- Particle Boundary
Particles.Permeable = L.Global(L.int, 0)
Particles.Solid     = L.Global(L.int, 1)

-- Output formats
IO.Tecplot = L.Global(L.int, 0)

-- General ON/OFF Flags
OFF = L.Global(L.bool, false)
ON  = L.Global(L.bool, true)

-----------------------------------------------------------------------------
--[[                       COLORS FOR VISUALIZATION                      ]]--
-----------------------------------------------------------------------------

local unity = L.Constant(L.vec3f,{1.0,1.0,1.0})
local red   = L.Constant(L.vec3f,{1.0,0.0,0.0})
local green = L.Constant(L.vec3f,{0.0,1.0,0.0})
local blue  = L.Constant(L.vec3f,{0.0,0.0,1.0})
local white = L.Constant(L.vec3f,{1.0,1.0,1.0})

-----------------------------------------------------------------------------
--[[                   INITIALIZE OPTIONS FROM CONFIG                    ]]--
-----------------------------------------------------------------------------

local grid_options = {
  -- Number of cells in the x, y, & z directions
  xnum        = config.xnum,
  ynum        = config.ynum,
  znum        = config.znum,
  -- Origin of the computational domain (meters)
  origin      = config.origin,
  -- Width of the computational domain in the x, y, & z directions (meters)
  xWidth      = config.xWidth,
  yWidth      = config.yWidth,
  zWidth      = config.zWidth,
  -- Boundary condition type for each face of the block and possible
  -- wall velocity, if no-slip.
  xBCLeft      = config.xBCLeft,
  xBCLeftVel   = config.xBCLeftVel,
  xBCLeftTemp  = config.xBCLeftTemp,
  xBCRight     = config.xBCRight,
  xBCRightVel  = config.xBCRightVel,
  xBCRightTemp = config.xBCRightTemp,
  yBCLeft      = config.yBCLeft,
  yBCLeftVel   = config.yBCLeftVel,
  yBCLeftTemp  = config.yBCLeftTemp,
  yBCRight     = config.yBCRight,
  yBCRightVel  = config.yBCRightVel,
  yBCRightTemp = config.yBCRightTemp,
  zBCLeft      = config.zBCLeft,
  zBCLeftVel   = config.zBCLeftVel,
  zBCLeftTemp  = config.zBCLeftTemp,
  zBCRight     = config.zBCRight,
  zBCRightVel  = config.zBCRightVel,
  zBCRightTemp = config.zBCRightTemp,
}

-- Define offsets for boundary conditions in flow solver
-- The sign variables define the necessary reflections for the
-- different types of BCs. The wall velocity is specified above,
-- and then the velocity adjustment is calculated here and applied
-- to the boundaries below.

-- Define offsets, signs, and velocities for the x BCs

x_sign = L.Global(L.vec3d,{1.0,1.0,1.0})
xpos_velocity    = L.Global(L.vec3d,{0.0,0.0,0.0})
xneg_velocity    = L.Global(L.vec3d,{0.0,0.0,0.0})
xpos_temperature = L.Global(L.double,-1.0)
xneg_temperature = L.Global(L.double,-1.0)

if grid_options.xBCLeft  == "periodic" and 
   grid_options.xBCRight == "periodic" then
  x_sign:set({1.0,1.0,1.0})
  xpos_velocity:set({0.0,0.0,0.0})
  xneg_velocity:set({0.0,0.0,0.0})
  xpos_temperature:set(-1.0)
  xneg_temperature:set(-1.0)
  grid_options.xBCLeftParticles  = Particles.Permeable
  grid_options.xBCRightParticles = Particles.Permeable
elseif grid_options.xBCLeft == "symmetry" and
       grid_options.xBCRight == "symmetry" then
  x_sign:set({-1.0,1.0,1.0})
  xpos_velocity:set({0.0,0.0,0.0})
  xneg_velocity:set({0.0,0.0,0.0})
  xpos_temperature:set(-1.0)
  xneg_temperature:set(-1.0)
  grid_options.xBCLeftParticles  = Particles.Solid
  grid_options.xBCRightParticles = Particles.Solid
elseif grid_options.xBCLeft  == "adiabatic_wall" and
       grid_options.xBCRight == "adiabatic_wall" then
  x_sign:set({-1.0,-1.0,-1.0})
  xpos_velocity:set({2.0*grid_options.xBCRightVel[1],
                    2.0*grid_options.xBCRightVel[2],
                    2.0*grid_options.xBCRightVel[3]})
  xneg_velocity:set({2.0*grid_options.xBCLeftVel[1],
                    2.0*grid_options.xBCLeftVel[2],
                    2.0*grid_options.xBCLeftVel[3]})
  xpos_temperature:set(-1.0)
  xneg_temperature:set(-1.0)
  grid_options.xBCLeftParticles  = Particles.Solid
  grid_options.xBCRightParticles = Particles.Solid
elseif grid_options.xBCLeft  == "isothermal_wall" and
       grid_options.xBCRight == "isothermal_wall" then
  x_sign:set({-1.0,-1.0,-1.0})
  xpos_velocity:set({2.0*grid_options.xBCRightVel[1],
                    2.0*grid_options.xBCRightVel[2],
                    2.0*grid_options.xBCRightVel[3]})
  xneg_velocity:set({2.0*grid_options.xBCLeftVel[1],
                     2.0*grid_options.xBCLeftVel[2],
                     2.0*grid_options.xBCLeftVel[3]})
  xpos_temperature:set(grid_options.xBCRightTemp)
  xneg_temperature:set(grid_options.xBCLeftTemp)
  grid_options.xBCLeftParticles  = Particles.Solid
  grid_options.xBCRightParticles = Particles.Solid
else
  error("Boundary conditions in x not implemented")
end

-- Define offsets, signs, and velocities for the y BCs

y_sign = L.Global(L.vec3d,{1.0,1.0,1.0})
ypos_velocity    = L.Global(L.vec3d,{0.0,0.0,0.0})
yneg_velocity    = L.Global(L.vec3d,{0.0,0.0,0.0})
ypos_temperature = L.Global(L.double,-1.0)
yneg_temperature = L.Global(L.double,-1.0)

if grid_options.yBCLeft  == "periodic" and 
   grid_options.yBCRight == "periodic" then
  y_sign:set({1.0,1.0,1.0})
  ypos_velocity:set({0.0,0.0,0.0})
  yneg_velocity:set({0.0,0.0,0.0})
  ypos_temperature:set(-1.0)
  yneg_temperature:set(-1.0)
  grid_options.yBCLeftParticles  = Particles.Permeable
  grid_options.yBCRightParticles = Particles.Permeable
elseif grid_options.yBCLeft  == "symmetry" and
       grid_options.yBCRight == "symmetry" then
  y_sign:set({1.0,-1.0,1.0})
  ypos_velocity:set({0.0,0.0,0.0})
  yneg_velocity:set({0.0,0.0,0.0})
  ypos_temperature:set(-1.0)
  yneg_temperature:set(-1.0)
  grid_options.yBCLeftParticles  = Particles.Solid
  grid_options.yBCRightParticles = Particles.Solid
elseif grid_options.yBCLeft  == "adiabatic_wall" and
       grid_options.yBCRight == "adiabatic_wall" then
  y_sign:set({-1.0,-1.0,-1.0})
  ypos_velocity:set({2.0*grid_options.yBCRightVel[1],
                    2.0*grid_options.yBCRightVel[2],
                    2.0*grid_options.yBCRightVel[3]})
  yneg_velocity:set({2.0*grid_options.yBCLeftVel[1],
                     2.0*grid_options.yBCLeftVel[2],
                     2.0*grid_options.yBCLeftVel[3]})
  ypos_temperature:set(-1.0)
  yneg_temperature:set(-1.0)
  grid_options.yBCLeftParticles  = Particles.Solid
  grid_options.yBCRightParticles = Particles.Solid
elseif grid_options.yBCLeft  == "isothermal_wall" and
       grid_options.yBCRight == "isothermal_wall" then
  y_sign:set({-1.0,-1.0,-1.0})
  ypos_velocity:set({2.0*grid_options.yBCRightVel[1],
                    2.0*grid_options.yBCRightVel[2],
                    2.0*grid_options.yBCRightVel[3]})
  yneg_velocity:set({2.0*grid_options.yBCLeftVel[1],
                     2.0*grid_options.yBCLeftVel[2],
                     2.0*grid_options.yBCLeftVel[3]})
  ypos_temperature:set(grid_options.yBCRightTemp)
  yneg_temperature:set(grid_options.yBCLeftTemp)
  grid_options.yBCLeftParticles  = Particles.Solid
  grid_options.yBCRightParticles = Particles.Solid
else
  error("Boundary conditions in y not implemented")
end

-- Define offsets, signs, and velocities for the z BCs

z_sign = L.Global(L.vec3d,{1.0,1.0,1.0})
zpos_velocity    = L.Global(L.vec3d,{0.0,0.0,0.0})
zneg_velocity    = L.Global(L.vec3d,{0.0,0.0,0.0})
zpos_temperature = L.Global(L.double,-1.0)
zneg_temperature = L.Global(L.double,-1.0)

if grid_options.zBCLeft  == "periodic" and 
   grid_options.zBCRight == "periodic" then
  z_sign:set({1.0,1.0,1.0})
  zpos_velocity:set({0.0,0.0,0.0})
  zneg_velocity:set({0.0,0.0,0.0})
  zpos_temperature:set(-1.0)
  zneg_temperature:set(-1.0)
  grid_options.zBCLeftParticles  = Particles.Permeable
  grid_options.zBCRightParticles = Particles.Permeable
elseif grid_options.zBCLeft == "symmetry" and
       grid_options.zBCRight == "symmetry" then
  z_sign:set({1.0,1.0,-1.0})
  zpos_velocity:set({0.0,0.0,0.0})
  zneg_velocity:set({0.0,0.0,0.0})
  zpos_temperature:set(-1.0)
  zneg_temperature:set(-1.0)
  grid_options.zBCLeftParticles  = Particles.Solid
  grid_options.zBCRightParticles = Particles.Solid
elseif grid_options.zBCLeft  == "adiabatic_wall" and
       grid_options.zBCRight == "adiabatic_wall" then
  z_sign:set({-1.0,-1.0,-1.0})
  zpos_velocity:set({2.0*grid_options.zBCRightVel[1],
                    2.0*grid_options.zBCRightVel[2],
                    2.0*grid_options.zBCRightVel[3]})
  zneg_velocity:set({2.0*grid_options.zBCLeftVel[1],
                     2.0*grid_options.zBCLeftVel[2],
                     2.0*grid_options.zBCLeftVel[3]})
  zpos_temperature:set(-1.0)
  zneg_temperature:set(-1.0)
  grid_options.zBCLeftParticles  = Particles.Solid
  grid_options.zBCRightParticles = Particles.Solid
elseif grid_options.zBCLeft  == "isothermal_wall" and
       grid_options.zBCRight == "isothermal_wall" then
  z_sign:set({-1.0,-1.0,-1.0})
  zpos_velocity:set({2.0*grid_options.zBCRightVel[1],
                    2.0*grid_options.zBCRightVel[2],
                    2.0*grid_options.zBCRightVel[3]})
  zneg_velocity:set({2.0*grid_options.zBCLeftVel[1],
                    2.0*grid_options.zBCLeftVel[2],
                    2.0*grid_options.zBCLeftVel[3]})
  zpos_temperature:set(grid_options.zBCRightTemp)
  zneg_temperature:set(grid_options.zBCLeftTemp)
  grid_options.zBCLeftParticles  = Particles.Solid
  grid_options.zBCRightParticles = Particles.Solid
else
  error("Boundary conditions in z not implemented")
end

-- Spatial integration options
local spatial_stencil = {}
spatial_stencil = {
  --  Splitting parameter
  split = 0.5
}

-- Time integrator options
TimeIntegrator.coeff_function        = {1/6, 1/3, 1/3, 1/6}
TimeIntegrator.coeff_time            = {0.5, 0.5, 1, 1}
TimeIntegrator.simTime               = L.Global(L.double,0)
TimeIntegrator.final_time            = config.final_time
TimeIntegrator.max_iter              = config.max_iter
TimeIntegrator.timeStep              = L.Global(L.int,0)
TimeIntegrator.cfl                   = config.cfl
TimeIntegrator.delta_time            = config.delta_time
TimeIntegrator.outputEveryTimeSteps  = config.outputEveryTimeSteps
TimeIntegrator.restartEveryTimeSteps = config.restartEveryTimeSteps
TimeIntegrator.headerFrequency       = config.headerFrequency
TimeIntegrator.deltaTime             = L.Global(L.double, 0.01)

local fluid_options = {}
if config.viscosity_model == 'Constant' then
  fluid_options.viscosity_model = Viscosity.Constant
elseif config.viscosity_model  == 'PowerLaw' then
  fluid_options.viscosity_model = Viscosity.PowerLaw
elseif config.viscosity_model  == 'Sutherland' then
  fluid_options.viscosity_model = Viscosity.Sutherland
else
  error("Viscosity model not defined")
end
fluid_options.gasConstant        = config.gasConstant
fluid_options.gamma              = config.gamma
fluid_options.prandtl            = config.prandtl
fluid_options.constant_visc      = config.constant_visc
fluid_options.powerlaw_visc_ref  = config.powerlaw_visc_ref
fluid_options.powerlaw_temp_ref  = config.powerlaw_temp_ref
fluid_options.suth_visc_ref      = config.suth_visc_ref
fluid_options.suth_temp_ref      = config.suth_temp_ref
fluid_options.suth_s_ref         = config.suth_s_ref

local flow_options = {}
if config.initCase == 'Uniform' then
  flow_options.initCase = Flow.Uniform
elseif config.initCase == 'Restart' then
  flow_options.initCase = Flow.Restart
elseif config.initCase == 'Perturbed' then
  flow_options.initCase = Flow.Perturbed
elseif config.initCase == 'TaylorGreen2DVortex' then
  flow_options.initCase = Flow.TaylorGreen2DVortex
elseif config.initCase == 'TaylorGreen3DVortex' then
  flow_options.initCase = Flow.TaylorGreen3DVortex
else
  error("Flow initialization type not defined")
end
flow_options.initParams     = L.Global(L.vector(L.double,5), config.initParams)
flow_options.bodyForce      = L.Global(L.vec3d, config.bodyForce)
flow_options.turbForceCoeff = L.Global(L.double, config.turbForceCoeff)

local particles_options = {
    -- Feeder is defined by a type and a set of parameters
    -- feedeerParams is a vector of double values whose meaning
    -- differs for each feederType. Please refer to the 
    -- Particles.Feed function where it is specialized
    --
    -- Feed all particles at start randomly
    -- distributed on a box defined by its center and sides
    feederType = Particles.FeederAtStartTimeInRandomBox,
    --feederParams = L.Global(L.vector(L.double,6),
    --                           {pi,pi,pi,2*pi,2*pi,2*pi}), -- TGV problem
    feederParams = L.Global(L.vector(L.double,6),  -- Default is entire box
                               {(grid_options.origin[1]+grid_options.xWidth)/2.,
                               (grid_options.origin[2]+grid_options.yWidth)/2.,
                               (grid_options.origin[3]+grid_options.zWidth)/2.,
                               (grid_options.origin[1]+grid_options.xWidth),
                               (grid_options.origin[2]+grid_options.yWidth),
                               (grid_options.origin[3]+grid_options.zWidth)}),

    -- Feeding a given number of particles every timestep randomly
    -- distributed on a box defined by its center and sides
    --feederType = Particles.FeederOverTimeInRandomBox,
    --feederParams = L.Global(L.vector(L.double,10),
    --                           {pi,pi,pi, -- centerCoor
    --                            2*pi,2*pi,2*pi, -- widthCoor
    --                            2.0,2.0,0, -- velocity
    --                            0.5, -- particlesPerTimestep
    --                           }), 
    --
    ---- UQCase
    --feederType = Particles.FeederUQCase,
    --feederParams = L.Global(L.vector(L.double,20),
    --                   {pi/4,  pi/2,pi,0.1*pi,0.1*pi,pi/2,4, 4,0,0.3,
    --                    pi/4,3*pi/2,pi,0.1*pi,0.1*pi,pi/2,4,-4,0,0.8}),
    
    -- Collector is defined by a type and a set of parameters
    -- collectorParams is a vector of double values whose meaning
    -- differs for each collectorType. Please refer to the 
    -- Particles.Collect function where it is specialized
    --
    -- Do not collect particles (freely move within the domain)
    collectorType   = Particles.CollectorNone,
    collectorParams = L.Global(L.vector(L.double,1),{0}),
    
    -- Collect all particles that exit a box defined by its Cartesian 
    -- min/max coordinates
    --collectorType = Particles.CollectorOutOfBox,
    --collectorParams = L.Global(L.vector(L.double,6),{0.5,0.5,0.5,12,6,6}),

    num = config.num,
    restitution_coefficient = L.Global(L.double,
                                          config.restitutionCoefficient),
    convective_coefficient = L.Global(L.double,
                                         config.convectiveCoefficient),
    heat_capacity = L.Global(L.double, config.heatCapacity),
    initialTemperature = config.initialTemperature,
    density = config.density,
    diameter_mean = config.diameter_mean,
    diameter_maxDeviation = config.diameter_maxDeviation,
    bodyForce = L.Global(L.vec3d, config.bodyForceParticles),
    absorptivity = config.absorptivity,
    restartParticleIter = config.restartParticleIter,
}
if config.initParticles == 'Random' then
  particles_options.initParticles = Particles.Random
elseif config.initParticles == 'Restart' then
  particles_options.initParticles = Particles.Restart
else
  error("Particle initialization type not defined")
end
-- Lastly, check whether the particles are fixed or free
if config.particleType == 'Fixed' then
  particles_options.particleType = Particles.Fixed
elseif config.particleType == 'Free' then
  particles_options.particleType = Particles.Free
else
  error("Particle motion type not defined (Fixed or Free)")
end
if config.twoWayCoupling == 'ON' then
  particles_options.twoWayCoupling = ON
elseif config.twoWayCoupling == 'OFF' then
  particles_options.twoWayCoupling = OFF
else
  error("Particle two-way couplding not defined (ON or OFF)")
end

local radiation_options = {}
if config.radiationType == 'ON' then
  radiation_options.radiationType = ON
elseif config.radiationType == 'OFF' then
  radiation_options.radiationType = OFF
else
  error("Radiation type not defined (ON or OFF)")
end
radiation_options.radiationIntensity = config.radiationIntensity
if config.zeroAvgHeatSource == 'ON' then
  radiation_options.zeroAvgHeatSource = ON
  elseif config.zeroAvgHeatSource == 'OFF' then
  radiation_options.zeroAvgHeatSource = OFF
  else
  error("Fixing average flow temp (fixAvgFlowTemp) not defined (ON or OFF)")
end

-- IO options
-- Choose an output format (Only Tecplot at the moment)
if config.outputFormat == 'Tecplot' then
  IO.outputFormat = IO.Tecplot
else
  error("Output format not implemented")
end
if config.wrtRestart == 'ON' then
  IO.wrtRestart = ON
  elseif config.wrtRestart == 'OFF' then
  IO.wrtRestart = OFF
  else
  error("Restart writing not defined (wrtRestart ON or OFF)")
end
if config.wrtVolumeSolution == 'ON' then
  IO.wrtVolumeSolution = ON
  elseif config.wrtVolumeSolution == 'OFF' then
  IO.wrtVolumeSolution = OFF
  else
  error("Volume solution writing not defined (wrtVolumeSolution ON or OFF)")
end
if config.wrt1DSlice == 'ON' then
  IO.wrt1DSlice = ON
  elseif config.wrt1DSlice == 'OFF' then
  IO.wrt1DSlice = OFF
  else
  error("1D slice writing not defined (wrt1DSlice ON or OFF)")
end
if config.wrtParticleEvolution == 'ON' then
  IO.wrtParticleEvolution = ON
  elseif config.wrtParticleEvolution == 'OFF' then
  IO.wrtParticleEvolution = OFF
  else
  error("Particle evolution writing not defined (wrtParticleEvolution ON or OFF)")
end
-- Store the index of the particle that we would like to track
IO.particleEvolutionIndex = config.particleEvolutionIndex

-- Store the directory for all output files from the config
IO.outputFileNamePrefix = config.outputDirectory

-- VDB options. For now, disable VDB (can add back in config.visualize later)
local vdb_options = {}
vdb_options.visualize = OFF


-----------------------------------------------------------------------------
--[[                       Load Data for a Restart                       ]]--
-----------------------------------------------------------------------------

-- Create empty arrays for storing the restart info

local restartNX, restartNY, restartNZ, restartIter, restartTime

if flow_options.initCase == Flow.Restart then

  -- here's the path object for our soleil restart info file. note that
  -- this file only contains auxiliary info we need, and that the fields
  -- are contained in CSVs to be read in below
  local restart_filename = IO.outputFileNamePrefix .. 'restart_' ..
                           config.restartIter .. '.dat'

  -- Restart info files have the following format
  --[[
   Soleil Flow Restart
   #cells currentTimeStep currentPhysicalTime
  ]]--

  -- In Lua, we can open files just like in C
  local soleil_in = io.open(tostring(restart_filename), "r")
  if not soleil_in then
    error('Error: failed to open '..tostring(restart_filename))
  end

  -- we can read a line like so
  local SOLEIL_SIG = soleil_in:read('*line')

  if SOLEIL_SIG ~= 'Soleil Flow Restart' then
    error('Restart file must begin with the first line "Soleil Flow Restart"')
  end

  -- read the counts of cells, iterations, and the time
  restartNX   = soleil_in:read('*number')
  restartNY   = soleil_in:read('*number')
  restartNZ   = soleil_in:read('*number')
  restartIter = soleil_in:read('*number')
  restartTime = soleil_in:read('*number')

  -- don't forget to close the file when done
  soleil_in:close()

  -- Before exiting, increment the time step and physical time so
  -- the simulation doesn't repeat from 0. Also, increased the max number
  -- of iterations so the solver doesn't immediately exit.

  TimeIntegrator.timeStep:set(restartIter)
  TimeIntegrator.simTime:set(restartTime)
  TimeIntegrator.max_iter = TimeIntegrator.max_iter + restartIter
  
  
end


-----------------------------------------------------------------------------
--[[                       GRID/PARTICLES RELATIONS                      ]]--
-----------------------------------------------------------------------------

-- Check boundary type consistency for the periodic BCs

if ( grid_options.xBCLeft  == 'periodic' and 
     grid_options.xBCRight ~= 'periodic' ) or 
   ( grid_options.xBCLeft  ~= 'periodic' and 
     grid_options.xBCRight == 'periodic' ) then
    error("Boundary conditions in x should match for periodicity")
end
if ( grid_options.yBCLeft  == 'periodic' and 
     grid_options.yBCRight ~= 'periodic' ) or 
   ( grid_options.yBCLeft  ~= 'periodic' and 
     grid_options.yBCRight == 'periodic' ) then
    error("Boundary conditions in y should match for periodicity")
end
if ( grid_options.zBCLeft  == 'periodic' and 
     grid_options.zBCRight ~= 'periodic' ) or 
   ( grid_options.zBCLeft  ~= 'periodic' and 
     grid_options.zBCRight == 'periodic' ) then
    error("Boundary conditions in z should match for periodicity")
end
if ( grid_options.xBCLeft  == 'periodic' and 
     grid_options.xBCRight == 'periodic' ) then
  xBCPeriodic = true
else
  xBCPeriodic = false
end
if ( grid_options.yBCLeft  == 'periodic' and 
     grid_options.yBCRight == 'periodic' ) then
  yBCPeriodic = true
else
  yBCPeriodic = false
end
if ( grid_options.zBCLeft  == 'periodic' and 
     grid_options.zBCRight == 'periodic' ) then
  zBCPeriodic = true
else
  zBCPeriodic = false
end

-- Declare and initialize grid and related fields
-- As we are second-order, we will initialize the grid
-- with a single layer of halo cells (unless running a
-- periodic case, which is natively handled w/out halos).
local bnum = 1
if xBCPeriodic then xBnum = 0 else xBnum = bnum end
if yBCPeriodic then yBnum = 0 else yBnum = bnum end
if zBCPeriodic then zBnum = 0 else zBnum = bnum end
local xBw = grid_options.xWidth/grid_options.xnum * xBnum
local yBw = grid_options.yWidth/grid_options.ynum * yBnum
local zBw = grid_options.zWidth/grid_options.znum * zBnum
local gridOriginInteriorX = grid_options.origin[1]
local gridOriginInteriorY = grid_options.origin[2]
local gridOriginInteriorZ = grid_options.origin[3]
local gridWidthX = grid_options.xWidth
local gridWidthY = grid_options.yWidth
local gridWidthZ = grid_options.zWidth

local grid = Grid.NewGrid3d{
              size           = {grid_options.xnum + 2*xBnum,
                                grid_options.ynum + 2*yBnum,
                                grid_options.znum + 2*zBnum},
              origin         = {grid_options.origin[1] - 
                                xBnum * grid_options.xWidth/grid_options.xnum,
                                grid_options.origin[2] - 
                                yBnum * grid_options.yWidth/grid_options.ynum,
                                grid_options.origin[3] - 
                                zBnum * grid_options.zWidth/grid_options.znum},
              width          = {grid_options.xWidth + 2*xBw,
                                grid_options.yWidth + 2*yBw,
                                grid_options.zWidth + 2*zBw},
              boundary_depth = {xBnum, yBnum, zBnum},
              periodic_boundary = {xBCPeriodic, yBCPeriodic, zBCPeriodic} }

-- Define uniform grid spacing
-- WARNING: These are used for uniform grids and should be replaced by different
-- metrics for non-uniform ones (see other WARNINGS throughout the code)
local grid_originX = L.Constant(L.double, grid:xOrigin())
local grid_originY = L.Constant(L.double, grid:yOrigin())
local grid_originZ = L.Constant(L.double, grid:zOrigin())
local grid_dx      = L.Constant(L.double, grid:xCellWidth())
local grid_dy      = L.Constant(L.double, grid:yCellWidth())
local grid_dz      = L.Constant(L.double, grid:zCellWidth())

-- Create a field for the center coords of the dual cells (i.e., vertices)
grid.vertices:NewField('centerCoordinates', L.vec3d)          :Load({0, 0, 0})

-- Create a field to mark the rind layer so it is not written in the output
-- We need this for both the dual cells (coords) and cells (cell-center data)
grid.vertices:NewField('vertexRindLayer', L.int)              :Load(1)
grid.cells:NewField('cellRindLayer', L.int)                   :Load(1)

-- Primitive variables (note that these may be initialized from a restart)
grid.cells:NewField('rho', L.double)
grid.cells:NewField('pressure', L.double)
grid.cells:NewField('velocity', L.vec3d)
if flow_options.initCase == Flow.Restart then
  grid.cells.rho:LoadFromCSV(IO.outputFileNamePrefix .. 'restart_rho_' ..
                             config.restartIter .. '.csv')
  grid.cells.pressure:LoadFromCSV(IO.outputFileNamePrefix .. 'restart_pressure_'
                                  .. config.restartIter .. '.csv')
  grid.cells.velocity:LoadFromCSV(IO.outputFileNamePrefix .. 'restart_velocity_'
                                  .. config.restartIter .. '.csv')
else
  grid.cells.rho        :Load(0)
  grid.cells.pressure   :Load(0)
  grid.cells.velocity   :Load({0, 0, 0})
end

-- Remaining primitive variables
grid.cells:NewField('centerCoordinates', L.vec3d)             :Load({0, 0, 0})
grid.cells:NewField('velocityGradientX', L.vec3d)             :Load({0, 0, 0})
grid.cells:NewField('velocityGradientY', L.vec3d)             :Load({0, 0, 0})
grid.cells:NewField('velocityGradientZ', L.vec3d)             :Load({0, 0, 0})
grid.cells:NewField('temperature', L.double)                  :Load(0)
grid.cells:NewField('rhoEnthalpy', L.double)                  :Load(0)
grid.cells:NewField('kineticEnergy', L.double)                :Load(0)
grid.cells:NewField('sgsEnergy', L.double)                    :Load(0)
grid.cells:NewField('sgsEddyViscosity', L.double)             :Load(0)
grid.cells:NewField('sgsEddyKappa', L.double)                 :Load(0)
grid.cells:NewField('convectiveSpectralRadius', L.double)     :Load(0)
grid.cells:NewField('viscousSpectralRadius', L.double)        :Load(0)
grid.cells:NewField('heatConductionSpectralRadius', L.double) :Load(0)

-- Conserved variables
grid.cells:NewField('rhoVelocity', L.vec3d)                   :Load({0, 0, 0})
grid.cells:NewField('rhoEnergy', L.double)                    :Load(0)

-- Fields for boundary treatment
grid.cells:NewField('rhoBoundary', L.double)                  :Load(0)
grid.cells:NewField('rhoVelocityBoundary', L.vec3d)           :Load({0, 0, 0})
grid.cells:NewField('rhoEnergyBoundary', L.double)            :Load(0)
grid.cells:NewField('velocityBoundary', L.vec3d)              :Load({0, 0, 0})
grid.cells:NewField('pressureBoundary', L.double)             :Load(0)
grid.cells:NewField('temperatureBoundary', L.double)          :Load(0)
grid.cells:NewField('velocityGradientXBoundary', L.vec3d)     :Load({0, 0, 0})
grid.cells:NewField('velocityGradientYBoundary', L.vec3d)     :Load({0, 0, 0})
grid.cells:NewField('velocityGradientZBoundary', L.vec3d)     :Load({0, 0, 0})

-- scratch (temporary) fields
-- intermediate value and copies
grid.cells:NewField('rho_old', L.double)                      :Load(0)
grid.cells:NewField('rhoVelocity_old', L.vec3d)               :Load({0, 0, 0})
grid.cells:NewField('rhoEnergy_old', L.double)                :Load(0)
grid.cells:NewField('rho_new', L.double)                      :Load(0)
grid.cells:NewField('rhoVelocity_new', L.vec3d)               :Load({0, 0, 0})
grid.cells:NewField('rhoEnergy_new', L.double)                :Load(0)
-- time derivatives
grid.cells:NewField('rho_t', L.double)                        :Load(0)
grid.cells:NewField('rhoVelocity_t', L.vec3d)                 :Load({0, 0, 0})
grid.cells:NewField('rhoEnergy_t', L.double)                  :Load(0)
-- fluxes
grid.cells:NewField('rhoFlux', L.double)                      :Load(0)
grid.cells:NewField('rhoVelocityFlux', L.vec3d)               :Load({0, 0, 0})
grid.cells:NewField('rhoEnergyFlux', L.double)                :Load(0)

-- Declare and initialize particle relation and fields over the particle

local particles = L.NewRelation {
  size = particles_options.num,
  name = 'particles'
}

particles:NewField('dual_cell', grid.dual_cells)              :Load({0,0,0})
particles:NewField('cell', grid.cells)                        :Load({0,0,0})

particles:NewField('position', L.vec3d)
particles:NewField('velocity', L.vec3d)
particles:NewField('temperature', L.double)
particles:NewField('diameter', L.double)
-- state of a particle:
--   - a particle not yet fed has a state = 0
--   - a particle fed into the domain and active has a state = 1
--   - a particle already collected has a state = 2
particles:NewField('state', L.int)
if particles_options.initParticles == Particles.Restart then
  particles.position:LoadFromCSV(IO.outputFileNamePrefix ..
                                 'restart_particle_position_' ..
                                 config.restartParticleIter .. '.csv')
  particles.velocity:LoadFromCSV(IO.outputFileNamePrefix ..
                                 'restart_particle_velocity_' ..
                                 config.restartParticleIter .. '.csv')
  particles.temperature:LoadFromCSV(IO.outputFileNamePrefix ..
                                    'restart_particle_temperature_' ..
                                    config.restartParticleIter .. '.csv')
  particles.diameter:LoadFromCSV(IO.outputFileNamePrefix ..
                                 'restart_particle_diameter_' ..
                                 config.restartParticleIter .. '.csv')
  particles.state         :Load(1)
else
particles.position      :Load({0, 0, 0})
particles.velocity      :Load({0, 0, 0})
particles.temperature   :Load(particles_options.initialTemperature)
-- Initialize to random distribution with given mean value and maximum
-- deviation from it
particles.diameter      :Load(function(i)
                              return C.rand_unity() *
                                particles_options.diameter_maxDeviation +
                                particles_options.diameter_mean
                              end)
particles.state         :Load(0)
end

particles:NewField('position_ghost', L.vec3d)                 :Load({0, 0, 0})
particles:NewField('velocity_ghost', L.vec3d)                 :Load({0, 0, 0})
particles:NewField('velocity_t_ghost', L.vec3d)               :Load({0, 0, 0})

particles:NewField('density', L.double)       :Load(particles_options.density)
particles:NewField('deltaVelocityOverRelaxationTime', L.vec3d):Load({0, 0, 0})
particles:NewField('deltaTemperatureTerm', L.double)          :Load(0)
-- ID field
particles:NewField('id', L.int):
-- Initialize to random distribution with given mean value and maximum 
-- deviation from it
Load(function(i)
    return i -- TODO: THIS IS NOT RANDOM
end)
-- groupID: differentiates particles within a given distribution
-- For example, when multiple injectors are used
particles:NewField('groupID', L.int)                          :Load(0)

-- scratch (temporary) fields
-- intermediate values and copies
particles:NewField('position_old', L.vec3d)                   :Load({0, 0, 0})
particles:NewField('velocity_old', L.vec3d)                   :Load({0, 0, 0})
particles:NewField('temperature_old', L.double)               :Load(0)
particles:NewField('position_new', L.vec3d)                   :Load({0, 0, 0})
particles:NewField('velocity_new', L.vec3d)                   :Load({0, 0, 0})
particles:NewField('temperature_new', L.double)               :Load(0)
-- derivatives
particles:NewField('position_t', L.vec3d)                     :Load({0, 0, 0})
particles:NewField('velocity_t', L.vec3d)                     :Load({0, 0, 0})
particles:NewField('temperature_t', L.double)                 :Load(0)


-- Statistics quantities

-- Note: - numberOfInteriorCells and areaInterior could be defined as variables
-- from grid instead of Flow. Here Flow is used to avoid adding things to grid
-- externally
Flow.numberOfInteriorCells   = L.Global(L.int, 0)
Flow.areaInterior            = L.Global(L.double, 0.0)
Flow.averagePressure         = L.Global(L.double, 0.0)
Flow.averageTemperature      = L.Global(L.double, 0.0)
Flow.averageHeatSource       = L.Global(L.double, 0.0)
Flow.averageKineticEnergy    = L.Global(L.double, 0.0)
Flow.minTemperature          = L.Global(L.double, 0.0)
Flow.maxTemperature          = L.Global(L.double, 0.0)
Particles.averageTemperature = L.Global(L.double, 0.0)

-----------------------------------------------------------------------------
--[[                 CONSOLE OUTPUT AFTER PREPROCESSING                  ]]--
-----------------------------------------------------------------------------

print("\n")
print("---------------------------------------------------------------------")
print("|    _____    ___     _      ____   _____   _          __    __     |")
print("|   / ____|  / __ \\  | |    | ___| |_   _| | |        \\  \\  /  /    |")
print("|  | (___   | |  | | | |    | |_     | |   | |   ___   \\  \\/  /     |")
print("|   \\___ \\  | |  | | | |    |  _|    | |   | |  |___|   |    |      |")
print("|   ____) | | |__| | | |__  | |__   _| |_  | |__       /  /\\  \\     |")
print("|  |_____/   \\____/  |____| |____| |_____| |____|     /__/  \\__\\    |")
print("|                                                                   |")
print("| Soleil-X is a turbulence/particle/radiation solver written in     |")
print("| the Liszt-Ebb DSL for execution with the Legion runtime.          |")
print("|                                                                   |")
print("---------------------------------------------------------------------")
print("|                                                                   |")
print("| Soleil-X Version 0.0.1                                            |")
print("| Copyright (C) 2013-2015, Dr. Thomas D. Economon,                  |")
print("|                          Dr. Ivan Bermejo-Moreno                  |")
print("|                                                                   |")
print("| This program is free software; you can redistribute it and/or     |")
print("| modify it under the terms of the GNU General Public               |")
print("| License as published by the Free Software Foundation; either      |")
print("| version 3 of the License, or (at your option) any later version.  |")
print("|                                                                   |")
print("| This program is distributed in the hope that it will be useful,   |")
print("| but WITHOUT ANY WARRANTY; without even the implied warranty of    |")
print("| MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU  |")
print("| General Public License for more details.                          |")
print("|                                                                   |")
print("| You should have received a copy of the GNU General Public         |")
print("| License along with this program. If not, see                      |")
print("| <http://www.gnu.org/licenses/>.                                   |")
print("|                                                                   |")
print("---------------------------------------------------------------------")
print("")
print("------------------------- Grid Definition ---------------------------")
io.stdout:write(" Grid cells: ",
                string.format(" %d",grid_options.xnum), " x",
                string.format(" %d",grid_options.ynum), " x",
                string.format(" %d",grid_options.znum), "\n")
io.stdout:write(" Grid boundary depth in cells: ",
                string.format(" %d",grid:xBoundaryDepth()), " x",
                string.format(" %d",grid:yBoundaryDepth()), " x",
                string.format(" %d",grid:zBoundaryDepth()), "\n")
io.stdout:write(" Grid origin (w/ halo): (",
                string.format(" %6f",grid:xOrigin()), ",",
                string.format(" %6f",grid:yOrigin()), ",",
                string.format(" %6f",grid:zOrigin()), " ) meters\n")
io.stdout:write(" Domain size (meters w/ halo): ",
                string.format(" %6f",grid:xWidth()), " x",
                string.format(" %6f",grid:yWidth()), " x",
                string.format(" %6f",grid:zWidth()), "\n")
io.stdout:write(" Cell size (meters): ",
                string.format(" %6f",grid:xCellWidth()), " x",
                string.format(" %6f",grid:yCellWidth()), " x",
                string.format(" %6f",grid:zCellWidth()), "\n")
io.stdout:write(" Total grid cells (w/ halo): ",
                string.format(" %d",(grid_options.xnum+2*grid:xBoundaryDepth())
                *(grid_options.ynum+2*grid:yBoundaryDepth())
                *(grid_options.znum+2*grid:zBoundaryDepth())), "\n")
print("")
print("----------------------- Boundary Conditions -------------------------")
io.stdout:write(" X- : ", grid_options.xBCLeft, ", V = (",
                string.format("%4f",grid_options.xBCLeftVel[1]), ",",
                string.format("%4f",grid_options.xBCLeftVel[2]), ",",
                string.format("%4f",grid_options.xBCLeftVel[3]), "), T = ",
                string.format("%4f",grid_options.xBCLeftTemp), "\n")
io.stdout:write(" X+ : ", grid_options.xBCRight, ", V = (",
                string.format("%4f",grid_options.xBCRightVel[1]), ",",
                string.format("%4f",grid_options.xBCRightVel[2]), ",",
                string.format("%4f",grid_options.xBCRightVel[3]), "), T = ",
                string.format("%4f",grid_options.xBCRightTemp), "\n")
io.stdout:write(" Y- : ", grid_options.yBCLeft, ", V = (",
                string.format("%4f",grid_options.yBCLeftVel[1]), ",",
                string.format("%4f",grid_options.yBCLeftVel[2]), ",",
                string.format("%4f",grid_options.yBCLeftVel[3]), "), T = ",
                string.format("%4f",grid_options.yBCLeftTemp), "\n")
io.stdout:write(" Y+ : ", grid_options.yBCRight, ", V = (",
                string.format("%4f",grid_options.yBCRightVel[1]), ",",
                string.format("%4f",grid_options.yBCRightVel[2]), ",",
                string.format("%4f",grid_options.yBCRightVel[3]), "), T = ",
                string.format("%4f",grid_options.yBCRightTemp), "\n")
io.stdout:write(" Z- : ", grid_options.zBCLeft, ", V = (",
                string.format("%4f",grid_options.zBCLeftVel[1]), ",",
                string.format("%4f",grid_options.zBCLeftVel[2]), ",",
                string.format("%4f",grid_options.zBCLeftVel[3]), "), T = ",
                string.format("%4f",grid_options.zBCLeftTemp), "\n")
io.stdout:write(" Z+ : ", grid_options.zBCRight, ", V = (",
                string.format("%4f",grid_options.zBCRightVel[1]), ",",
                string.format("%4f",grid_options.zBCRightVel[2]), ",",
                string.format("%4f",grid_options.zBCRightVel[3]), "), T = ",
                string.format("%4f",grid_options.zBCRightTemp), "\n")
print("")
print("-------------------------- Fluid Options ----------------------------")
io.stdout:write(" Gas constant: ",
                string.format(" %f",fluid_options.gasConstant), "\n")
io.stdout:write(" Ratio of spcific heats: ",
                string.format(" %f",fluid_options.gamma), "\n")
io.stdout:write(" Viscosity model: ", config.viscosity_model, "\n")
if fluid_options.viscosity_model == Viscosity.Constant then
  io.stdout:write(" Constant viscosity value: ",
                  string.format(" %f",fluid_options.constant_visc), "\n")
elseif fluid_options.viscosity_model == Viscosity.PowerLaw then
  io.stdout:write(" Power law reference viscosity value: ",
                string.format(" %f",fluid_options.powerlaw_visc_ref), "\n")
  io.stdout:write(" Power law reference temperature value: ",
                string.format(" %f",fluid_options.powerlaw_temp_ref), "\n")
elseif fluid_options.viscosity_model == Viscosity.Sutherland then
  io.stdout:write(" Sutherland's law reference viscosity value: ",
                string.format(" %f",fluid_options.suth_visc_ref), "\n")
  io.stdout:write(" Sutherland's law reference temperature value: ",
                string.format(" %f",fluid_options.suth_temp_ref), "\n")
  io.stdout:write(" Sutherland's law reference S value: ",
                string.format(" %f",fluid_options.suth_s_ref), "\n")
end
io.stdout:write(" Fluid init. type: ", config.initCase, "\n")
if flow_options.initCase == Flow.Restart then
  io.stdout:write(" Restarting from iteration: ",
                  string.format(" %d",config.restartIter), "\n")
else
  io.stdout:write(" Fluid init. params: (",
                  string.format("%1.3f",config.initParams[1]), ",",
                  string.format("%1.3f",config.initParams[2]), ",",
                  string.format("%1.3f",config.initParams[3]), ",",
                  string.format("%1.3f",config.initParams[4]), ",",
                  string.format("%1.3f",config.initParams[5]), ")\n")
end
io.stdout:write(" Fluid body force: (",
                string.format("%1.3f",config.bodyForce[1]), ",",
                string.format("%1.3f",config.bodyForce[2]), ",",
                string.format("%1.3f",config.bodyForce[3]), ")\n")
io.stdout:write(" Linearly forced isotropic turbulence coefficient: ",
                string.format(" %f",config.turbForceCoeff), "\n")
print("")
print("------------------------- Particle Options --------------------------")
io.stdout:write(" Particle init. type: ", config.initParticles, "\n")
if particles_options.initCase == Particles.Restart then
  io.stdout:write(" Restarting from iteration: ",
                  string.format(" %d",config.restartParticleIter), "\n")
else
  io.stdout:write(" Initial temperature: ",
                  string.format(" %f",config.initialTemperature), "\n")
  io.stdout:write(" Mean particle diameter: ",
                  string.format(" %f",config.diameter_mean), "\n")
  io.stdout:write(" Diameter max deviation: ",
                  string.format(" %f",config.diameter_maxDeviation), "\n")
end
io.stdout:write(" Particle type (fixed or free): ", config.particleType, "\n")
io.stdout:write(" Two-way coupling: ", config.twoWayCoupling, "\n")
io.stdout:write(" Number of particles: ", string.format(" %d",config.num), "\n")
io.stdout:write(" Particle density: ",
                string.format(" %f",config.density), "\n")
io.stdout:write(" Coefficient of restitution: ",
                string.format(" %f",config.restitutionCoefficient), "\n")
io.stdout:write(" Convective coefficient: ",
                string.format(" %f",config.convectiveCoefficient), "\n")
io.stdout:write(" Heat capacity: ",
                string.format(" %f",config.heatCapacity), "\n")
io.stdout:write(" Absorptivity: ",
                string.format(" %f",config.absorptivity), "\n")
io.stdout:write(" Particle body force: (",
                string.format("%1.3f",config.bodyForceParticles[1]), ",",
                string.format("%1.3f",config.bodyForceParticles[2]), ",",
                string.format("%1.3f",config.bodyForceParticles[3]), ")\n")
print("")
print("------------------------ Radiation Options --------------------------")
io.stdout:write(" Radiation type: ", config.radiationType, "\n")
io.stdout:write(" Radiation intensity: ",
                string.format(" %f",config.radiationIntensity), "\n")
io.stdout:write(" Force zero avg. heat source: ", config.zeroAvgHeatSource, "\n")
print("")
print("------------------------- Time Integration --------------------------")
io.stdout:write(" Final physical time: ",
                string.format(" %f",config.final_time), "\n")
io.stdout:write(" Maximum number of iterations: ",
                string.format(" %d",config.max_iter), "\n")
if config.cfl > 0.0 then
io.stdout:write(" Courant–Friedrichs–Lewy (CFL) number: ",
                string.format(" %f",config.cfl), "\n")
else
io.stdout:write(" Fixed time step: ",
                string.format(" %f",config.delta_time), "\n")
end
print("")
print("-------------------------- Output Options ---------------------------")
io.stdout:write(" Restart files: ", config.wrtRestart, "\n")
io.stdout:write(" Restart output frequency (iterations): ",
                string.format(" %d",config.restartEveryTimeSteps), "\n")
io.stdout:write(" Volume solution files: ", config.wrtVolumeSolution, "\n")
io.stdout:write(" 1D slice output: ", config.wrt1DSlice, "\n")
io.stdout:write(" Particle tracking output ", config.wrtParticleEvolution, "\n")
io.stdout:write(" Solution output frequency (iterations): ",
                string.format(" %d",config.outputEveryTimeSteps), "\n")
io.stdout:write(" Header frequency (iterations): ",
                string.format(" %d",config.headerFrequency), "\n")
io.stdout:write(" Output format: ", config.outputFormat, "\n")
io.stdout:write(" Output directory: ", config.outputDirectory, "\n")
print("")
print("--------------------------- Start Solver ----------------------------")
print("")


-----------------------------------------------------------------------------
--[[                       USER DEFINED FUNCTIONS                        ]]--
-----------------------------------------------------------------------------

-- Norm of a vector
local ebb norm (v)
    return L.sqrt(L.dot(v, v))
end

-- Compute fluid dynamic viscosity from fluid temperature
local ebb GetDynamicViscosity (temperature)
  var viscosity = L.double(0.0)
  if fluid_options.viscosity_model == Viscosity.Constant then
    -- Constant
    viscosity = fluid_options.constant_visc
  elseif fluid_options.viscosity_model == Viscosity.PowerLaw then
    -- Power Law
    viscosity = fluid_options.powerlaw_visc_ref *
        L.pow(temperature/fluid_options.powerlaw_temp_ref, 0.75)
  elseif fluid_options.viscosity_model == Viscosity.Sutherland then
    -- Sutherland's Law
    viscosity = fluid_options.suth_visc_ref *
    L.pow((temperature/fluid_options.suth_temp_ref),(3.0/2.0))*
    ((fluid_options.suth_temp_ref + fluid_options.suth_s_ref)/
     (temperature + fluid_options.suth_s_ref))
  end
  return viscosity
end

-- Compute fluid flow sound speed based on temperature (a = sqrt(gamma*R*T))
local ebb GetSoundSpeed (temperature)
    return L.sqrt(fluid_options.gamma * fluid_options.gasConstant * temperature)
end

-- Function to retrieve particle area, volume and mass
-- These are Ebb user-defined functions that behave like a field
particles:NewFieldFunction('cross_section_area', ebb(p)
    return pi * L.pow(p.diameter, 2) / 4.0
end)
particles:NewFieldFunction('volume', ebb(p)
    return pi * L.pow(p.diameter, 3) / 6.0
end)
particles:NewFieldFunction('mass', ebb(p)
    return p.volume * p.density
end)

-----------------------------------------------------------------------------
--[[                              EBB MACROS                             ]]--
-----------------------------------------------------------------------------


local function GenerateTrilinearInterpolation(field_name)
  return ebb(dc, xyz)
    var c000 = dc.vertex.cell(-1, -1, -1)
    var c100 = dc.vertex.cell( 0, -1, -1)
    var c010 = dc.vertex.cell(-1,  0, -1)
    var c110 = dc.vertex.cell( 0,  0, -1)
    var c001 = dc.vertex.cell(-1, -1,  0)
    var c101 = dc.vertex.cell( 0, -1,  0)
    var c011 = dc.vertex.cell(-1,  0,  0)
    var c111 = dc.vertex.cell( 0,  0,  0)
    -- The following approach is valid for non-uniform grids, as it relies
    -- on the cell centers of the neighboring cells of the given dual cell
    -- (dc).
    -- WARNING: However, it poses a problem when periodicity is applied, as
    -- the built-in wrapping currently returns a cell which is on the
    -- opposite end of the grid, if the dual cell is in the periodic 
    -- boundary. Note that the field values are correctly retrieved through
    -- the wrapping, but not the positions used to define the weights of the
    -- interpolation
    --var dX = (xyz[0] - c000.center[0])/(c100.center[0] - c000.center[0])
    --var dY = (xyz[1] - c000.center[1])/(c010.center[1] - c000.center[1])
    --var dZ = (xyz[2] - c000.center[2])/(c001.center[2] - c000.center[2])
    -- WARNING: This assumes uniform mesh, and retrieves the position of the
    -- particle relative to the neighboring cells without resorting to the
    -- dual-cell itself, but purely based on grid origin and spacing
    -- See the other approch above (commented) for the generalization to
    -- non-uniform grids (with the current problem of not being usable if
    -- periodicity is enforced)
    var dX   = L.fmod((xyz[0] - grid_originX)/grid_dx + 0.5, 1.0)
    var dY   = L.fmod((xyz[1] - grid_originY)/grid_dy + 0.5, 1.0)
    var dZ   = L.fmod((xyz[2] - grid_originZ)/grid_dz + 0.5, 1.0)

    var oneMinusdX = 1.0 - dX
    var oneMinusdY = 1.0 - dY
    var oneMinusdZ = 1.0 - dZ
    var weight00 = c000.[field_name] * oneMinusdX + c100.[field_name] * dX 
    var weight10 = c010.[field_name] * oneMinusdX + c110.[field_name] * dX
    var weight01 = c001.[field_name] * oneMinusdX + c101.[field_name] * dX
    var weight11 = c011.[field_name] * oneMinusdX + c111.[field_name] * dX
    var weight0  = weight00 * oneMinusdY + weight10 * dY
    var weight1  = weight01 * oneMinusdY + weight11 * dY
    
    return weight0 * oneMinusdZ + weight1 * dZ
  end
end

local InterpolateTriRho = GenerateTrilinearInterpolation('rho')
local InterpolateTriVelocity = GenerateTrilinearInterpolation('velocity')
local InterpolateTriTemperature = GenerateTrilinearInterpolation('temperature')

-----------------------------------------------------------------------------
--[[                            EBB FUNCTIONS                            ]]--
-----------------------------------------------------------------------------

-------
-- FLOW
-------

-- Initialize flow variables
-- Cell center coordinates are stored in the grid field macro 'center'. 
-- Here, we use a field for convenience when outputting to file, but this is
-- to be removed after grid outputing is well defined from within the grid.t 
-- module. Similar story with the vertex coordinates (output only).
ebb Flow.InitializeCenterCoordinates (c : grid.cells)
    var xy = c.center
    c.centerCoordinates = L.vec3d({xy[0], xy[1], xy[2]})
end

ebb Flow.InitializeCellRindLayer (c : grid.cells)
    c.cellRindLayer = 0
end

-- Hard coding the vertices until we have access in grid.t
-- WARNING: Here, I am using the id numbers, but this is unsafe!
ebb Flow.InitializeVertexCoordinates (v : grid.vertices)
    var x = grid_originX + grid_dx * (L.double(L.xid(v)))
    var y = grid_originY + grid_dy * (L.double(L.yid(v)))
    var z = grid_originZ + grid_dz * (L.double(L.zid(v)))
    v.centerCoordinates = L.vec3d({x, y, z})
end

ebb Flow.InitializeVertexRindLayer (v : grid.vertices)
    v.vertexRindLayer = 0
end


ebb Flow.InitializePrimitives (c : grid.cells)
    if flow_options.initCase == Flow.TaylorGreen2DVortex then
      -- Define Taylor Green Vortex
      var taylorGreenDensity  = flow_options.initParams[0]
      var taylorGreenPressure = flow_options.initParams[1]
      var taylorGreenVelocity = flow_options.initParams[2]
      -- Initialize
      var xy = c.center
      var coorZ = 0
      c.rho = taylorGreenDensity
      c.velocity = 
          taylorGreenVelocity *
          L.vec3d({L.sin(xy[0]) * 
                   L.cos(xy[1]) *
                   L.cos(coorZ),
                 - L.cos(xy[0]) *
                   L.sin(xy[1]) *
                   L.cos(coorZ),
                   0})
      var factorA = L.cos(2.0*coorZ) + 2.0
      var factorB = L.cos(2.0*xy[0]) +
                    L.cos(2.0*xy[1])
      c.pressure = 
          taylorGreenPressure + 
          taylorGreenDensity * L.pow(taylorGreenVelocity,2) / 16 *
          factorA * factorB
    elseif flow_options.initCase == Flow.TaylorGreen3DVortex then
      -- Define Taylor Green Vortex
      var taylorGreenDensity  = flow_options.initParams[0]
      var taylorGreenPressure = flow_options.initParams[1]
      var taylorGreenVelocity = flow_options.initParams[2]
      -- Initialize
      var xy = c.center
      c.rho = taylorGreenDensity
      c.velocity = 
          taylorGreenVelocity *
          L.vec3d({L.sin(xy[0]) * 
                   L.cos(xy[1]) *
                   L.cos(xy[2]),
                 - L.cos(xy[0]) *
                   L.sin(xy[1]) *
                   L.cos(xy[2]),
                   0})
      var factorA = L.cos(2.0*xy[2]) + 2.0
      var factorB = L.cos(2.0*xy[0]) +
                    L.cos(2.0*xy[1])
      c.pressure = 
          taylorGreenPressure + 
          taylorGreenDensity * L.pow(taylorGreenVelocity,2) / 16 *
          factorA * factorB
    elseif flow_options.initCase == Flow.Uniform then
      c.rho         = flow_options.initParams[0]
      c.pressure    = flow_options.initParams[1]
      c.velocity[0] = flow_options.initParams[2]
      c.velocity[1] = flow_options.initParams[3]
      c.velocity[2] = flow_options.initParams[4]
      
    elseif flow_options.initCase == Flow.Perturbed then
      -- This initialization imposes a small random perturbation in 
      -- the density field used to start up forced turbulence cases
      c.rho         = flow_options.initParams[0]
      c.pressure    = flow_options.initParams[1]
      c.velocity[0] = flow_options.initParams[2] + (C.rand_unity()*1e-4)
      c.velocity[1] = flow_options.initParams[3] + (C.rand_unity()*1e-4)
      c.velocity[2] = flow_options.initParams[4] + (C.rand_unity()*1e-4)
    elseif flow_options.initCase == Flow.Restart then
     -- Do nothing here, since we have initialized using CSV restarts.
    end
end

ebb Flow.UpdateConservedFromPrimitive (c : grid.cells)

    -- Equation of state: T = p / ( R * rho )
    var tmpTemperature = c.pressure / (fluid_options.gasConstant * c.rho)
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
ebb Flow.InitializeTemporaries (c : grid.cells)
    c.rho_old         = c.rho
    c.rhoVelocity_old = c.rhoVelocity
    c.rhoEnergy_old   = c.rhoEnergy
    c.rho_new         = c.rho
    c.rhoVelocity_new = c.rhoVelocity
    c.rhoEnergy_new   = c.rhoEnergy
end

-- Initialize derivatives
ebb Flow.InitializeTimeDerivatives (c : grid.cells)
    c.rho_t         = L.double(0.0)
    c.rhoVelocity_t = L.vec3d({0.0, 0.0, 0.0})
    c.rhoEnergy_t   = L.double(0.0)
end

-----------
-- Inviscid
-----------

-- Initialize enthalpy and derivatives
ebb Flow.AddInviscidInitialize (c : grid.cells)
    c.rhoEnthalpy = c.rhoEnergy + c.pressure
end

-- Routine that computes the inviscid flux through the face of 
-- any two adjacent cells with a centered scheme. The left cell (c_l),
-- right cell (c_r), and coordinate direction (x = 0, y = 1, or z = 2)
-- are the inputs.
ebb Flow.CenteredInviscidFlux (c_l, c_r, direction)

    -- Diagonal terms of inviscid flux
    var rhoFactorDiagonal         = L.double(0.0)
    var rhoVelocityFactorDiagonal = L.vec3d({0.0, 0.0, 0.0})
    var rhoEnergyFactorDiagonal   = L.double(0.0)
    var fpdiag                    = L.double(0.0)

    rhoFactorDiagonal = 0.5 * ( c_l.rho * c_l.velocity[direction] +
                                c_r.rho * c_r.velocity[direction] )
    rhoVelocityFactorDiagonal = 0.5 *
                              ( c_l.rhoVelocity *
                                c_l.velocity[direction] +
                                c_r.rhoVelocity *
                                c_r.velocity[direction] )
    rhoEnergyFactorDiagonal = 0.5 *
                            ( c_l.rhoEnthalpy *
                              c_l.velocity[direction] +
                              c_r.rhoEnthalpy *
                              c_r.velocity[direction] )
    fpdiag += 0.5 * ( c_l.pressure + c_r.pressure )

    -- Skewed terms
    var rhoFactorSkew         = L.double(0.0)
    var rhoVelocityFactorSkew = L.vec3d({0.0, 0.0, 0.0})
    var rhoEnergyFactorSkew   = L.double(0.0)
    var tmp                   = L.double(0.0)

    tmp = 0.5 * c_r.velocity[direction]

    rhoFactorSkew         += c_l.rho * tmp
    rhoVelocityFactorSkew += c_l.rhoVelocity * tmp
    rhoEnergyFactorSkew   += c_l.rhoEnthalpy * tmp

    tmp = 0.5 * c_l.velocity[direction]

    rhoFactorSkew         += c_r.rho * tmp
    rhoVelocityFactorSkew += c_r.rhoVelocity * tmp
    rhoEnergyFactorSkew   += c_r.rhoEnthalpy * tmp

    -- Compute fluxes with prescribed splitting
    var s = spatial_stencil.split
    var rhoFlux_temp         = s * rhoFactorDiagonal +
                              (1-s) * rhoFactorSkew
    var rhoVelocityFlux_temp = s * rhoVelocityFactorDiagonal +
                              (1-s) * rhoVelocityFactorSkew
    var rhoEnergyFlux_temp   = s * rhoEnergyFactorDiagonal +
                              (1-s) * rhoEnergyFactorSkew
    rhoVelocityFlux_temp[direction] += fpdiag

    -- Return the fluxes in a 5D array
    return {rhoFlux_temp,
            rhoVelocityFlux_temp[0],
            rhoVelocityFlux_temp[1],
            rhoVelocityFlux_temp[2],
            rhoEnergyFlux_temp}
end

-- Compute inviscid fluxes in X direction. Include the first boundary
-- cell (c.xneg_depth == 1) to define left flux on first interior cell.
ebb Flow.AddInviscidGetFluxX (c : grid.cells)
    if c.in_interior or c.xneg_depth == 1 then
      
      -- Compute the inviscid flux with a centered scheme.
      -- Input the left and right cell states for this face and
      -- the direction index for the flux (x = 0, y = 1, or z = 2).
        var flux = Flow.CenteredInviscidFlux(c(0,0,0), c(1,0,0), 0)
        
        -- Store this flux in the cell to the left of the face.
        c.rhoFlux         =  flux[0]
        c.rhoVelocityFlux = {flux[1],flux[2],flux[3]}
        c.rhoEnergyFlux   =  flux[4]
        
    end
end

-- Compute inviscid fluxes in Y direction. Include the first boundary
-- cell (c.yneg_depth == 1) to define left flux on first interior cell.
ebb Flow.AddInviscidGetFluxY (c : grid.cells)
    if c.in_interior or c.yneg_depth == 1 then
      
      -- Compute the inviscid flux with a centered scheme.
      -- Input the left and right cell states for this face and
      -- the direction index for the flux (x = 0, y = 1, or z = 2).
      var flux = Flow.CenteredInviscidFlux(c(0,0,0), c(0,1,0), 1)
      
      -- Store this flux in the cell to the left of the face.
      c.rhoFlux         =  flux[0]
      c.rhoVelocityFlux = {flux[1],flux[2],flux[3]}
      c.rhoEnergyFlux   =  flux[4]
      
    end
end

-- Compute inviscid fluxes in Z direction. Include the first boundary
-- cell (c.zneg_depth == 1) to define left flux on first interior cell.
ebb Flow.AddInviscidGetFluxZ (c : grid.cells)
    if c.in_interior or c.zneg_depth == 1 then
      
      -- Compute the inviscid flux with a centered scheme.
      -- Input the left and right cell states for this face and
      -- the direction index for the flux (x = 0, y = 1, or z = 2).
      var flux = Flow.CenteredInviscidFlux(c(0,0,0), c(0,0,1), 2)
      
      -- Store this flux in the cell to the left of the face.
      c.rhoFlux         =  flux[0]
      c.rhoVelocityFlux = {flux[1],flux[2],flux[3]}
      c.rhoEnergyFlux   =  flux[4]
      
    end
end

-- Update conserved variables using flux values from previous part
-- write conserved variables, read flux variables
-- WARNING_START For non-uniform grids, the metrics used below 
-- (grid_dx, grid_dy, grid_dz) are not appropriate and should be changed
-- to reflect those expressed in the Python prototype code
-- WARNING_END
ebb Flow.AddInviscidUpdateUsingFluxX (c : grid.cells)
    c.rho_t += -(c( 0,0,0).rhoFlux -
                 c(-1,0,0).rhoFlux)/grid_dx
    c.rhoVelocity_t += -(c( 0,0,0).rhoVelocityFlux -
                         c(-1,0,0).rhoVelocityFlux)/grid_dx
    c.rhoEnergy_t += -(c( 0,0,0).rhoEnergyFlux -
                       c(-1,0,0).rhoEnergyFlux)/grid_dx
end
ebb Flow.AddInviscidUpdateUsingFluxY (c : grid.cells)
    c.rho_t += -(c(0, 0,0).rhoFlux -
                 c(0,-1,0).rhoFlux)/grid_dy
    c.rhoVelocity_t += -(c(0, 0,0).rhoVelocityFlux -
                         c(0,-1,0).rhoVelocityFlux)/grid_dy
    c.rhoEnergy_t += -(c(0, 0,0).rhoEnergyFlux -
                       c(0,-1,0).rhoEnergyFlux)/grid_dy
end
ebb Flow.AddInviscidUpdateUsingFluxZ (c : grid.cells)
    c.rho_t += -(c(0,0, 0).rhoFlux -
                 c(0,0,-1).rhoFlux)/grid_dz
    c.rhoVelocity_t += -(c(0,0, 0).rhoVelocityFlux -
                         c(0,0,-1).rhoVelocityFlux)/grid_dz
    c.rhoEnergy_t += -(c(0,0, 0).rhoEnergyFlux -
                       c(0,0,-1).rhoEnergyFlux)/grid_dz
end

----------
-- Viscous
----------

-- Compute viscous fluxes in X direction
ebb Flow.AddViscousGetFluxX (c : grid.cells)
    -- Consider first boundary element (c.xneg_depth == 1) to define left flux
    -- on first interior cell
    if c.in_interior or c.xneg_depth == 1 then
        var muFace = 0.5 * (GetDynamicViscosity(c(0,0,0).temperature) +
                            GetDynamicViscosity(c(1,0,0).temperature))
        var velocityFace    = L.vec3d({0.0, 0.0, 0.0})
        var velocityX_YFace = L.double(0)
        var velocityX_ZFace = L.double(0)
        var velocityY_YFace = L.double(0)
        var velocityZ_ZFace = L.double(0)

        -- Interpolate velocity and derivatives to face
        velocityFace = 0.5 * ( c(0,0,0).velocity + c(1,0,0).velocity )
        velocityX_YFace = 0.5 * ( c(0,0,0).velocityGradientY[0] +
                                  c(1,0,0).velocityGradientY[0] )
        velocityX_ZFace = 0.5 * ( c(0,0,0).velocityGradientZ[0] +
                                  c(1,0,0).velocityGradientZ[0] )
        velocityY_YFace = 0.5 * ( c(0,0,0).velocityGradientY[1] +
                                  c(1,0,0).velocityGradientY[1] )
        velocityZ_ZFace = 0.5 * ( c(0,0,0).velocityGradientZ[2] +
                                  c(1,0,0).velocityGradientZ[2] )

        -- Differentiate at face
        var velocityX_XFace   = L.double(0.0)
        var velocityY_XFace   = L.double(0.0)
        var velocityZ_XFace   = L.double(0.0)
        var temperature_XFace = L.double(0.0)

        velocityX_XFace   = 0.5*( c(1,0,0).velocity[0] - c(0,0,0).velocity[0] )
        velocityY_XFace   = 0.5*( c(1,0,0).velocity[1] - c(0,0,0).velocity[1] )
        velocityZ_XFace   = 0.5*( c(1,0,0).velocity[2] - c(0,0,0).velocity[2] )
        temperature_XFace = 0.5*( c(1,0,0).temperature - c(0,0,0).temperature )
       
        -- Half cell size due to the 0.5 above
        velocityX_XFace   /= (grid_dx*0.5)
        velocityY_XFace   /= (grid_dx*0.5)
        velocityZ_XFace   /= (grid_dx*0.5)
        temperature_XFace /= (grid_dx*0.5)

        -- Tensor components (at face)
        var sigmaXX = muFace * ( 4.0 * velocityX_XFace -
                                 2.0 * velocityY_YFace -
                                 2.0 * velocityZ_ZFace ) / 3.0
        var sigmaYX = muFace * ( velocityY_XFace + velocityX_YFace )
        var sigmaZX = muFace * ( velocityZ_XFace + velocityX_ZFace )
        var usigma  = velocityFace[0] * sigmaXX +
                      velocityFace[1] * sigmaYX +
                      velocityFace[2] * sigmaZX
        var cp = fluid_options.gamma * fluid_options.gasConstant / 
                 (fluid_options.gamma - 1.0)
        var heatFlux = - (cp*muFace/fluid_options.prandtl)*temperature_XFace

        -- Fluxes
        c.rhoVelocityFlux[0] = sigmaXX
        c.rhoVelocityFlux[1] = sigmaYX
        c.rhoVelocityFlux[2] = sigmaZX
        c.rhoEnergyFlux = usigma - heatFlux
        -- WARNING: Add SGS terms for LES

    end
end

-- Compute viscous fluxes in Y direction
ebb Flow.AddViscousGetFluxY (c : grid.cells)
    -- Consider first boundary element (c.yneg_depth == 1) to define down flux
    -- on first interior cell
    if c.in_interior or c.yneg_depth == 1 then
        var muFace = 0.5 * (GetDynamicViscosity(c(0,0,0).temperature) +
                            GetDynamicViscosity(c(0,1,0).temperature))
        var velocityFace    = L.vec3d({0.0, 0.0, 0.0})
        var velocityY_XFace = L.double(0)
        var velocityY_ZFace = L.double(0)
        var velocityX_XFace = L.double(0)
        var velocityZ_ZFace = L.double(0)

        -- Interpolate velocity and derivatives to face
        velocityFace = 0.5 * ( c(0,0,0).velocity + c(0,1,0).velocity )
        velocityY_XFace = 0.5 * ( c(0,0,0).velocityGradientX[1] +
                                  c(0,1,0).velocityGradientX[1] )
        velocityY_ZFace = 0.5 * ( c(0,0,0).velocityGradientZ[1] +
                                  c(0,1,0).velocityGradientZ[1] )
        velocityX_XFace = 0.5 * ( c(0,0,0).velocityGradientX[0] +
                                  c(0,1,0).velocityGradientX[0] )
        velocityZ_ZFace = 0.5 * ( c(0,0,0).velocityGradientZ[2] +
                                  c(0,1,0).velocityGradientZ[2] )
                             
        -- Differentiate at face
        var velocityX_YFace   = L.double(0.0)
        var velocityY_YFace   = L.double(0.0)
        var velocityZ_YFace   = L.double(0.0)
        var temperature_YFace = L.double(0.0)

        velocityX_YFace   = 0.5*( c(0,1,0).velocity[0] - c(0,0,0).velocity[0] )
        velocityY_YFace   = 0.5*( c(0,1,0).velocity[1] - c(0,0,0).velocity[1] )
        velocityZ_YFace   = 0.5*( c(0,1,0).velocity[2] - c(0,0,0).velocity[2] )
        temperature_YFace = 0.5*( c(0,1,0).temperature - c(0,0,0).temperature )
       
        -- Half cell size due to the 0.5 above
        velocityX_YFace   /= (grid_dy*0.5)
        velocityY_YFace   /= (grid_dy*0.5)
        velocityZ_YFace   /= (grid_dy*0.5)
        temperature_YFace /= (grid_dy*0.5)

        -- Tensor components (at face)
        var sigmaXY = muFace * ( velocityX_YFace + velocityY_XFace )
        var sigmaYY = muFace * ( 4.0 * velocityY_YFace -
                                 2.0 * velocityX_XFace -
                                 2.0 * velocityZ_ZFace ) / 3.0
        var sigmaZY = muFace * ( velocityZ_YFace + velocityY_ZFace )
        var usigma  = velocityFace[0] * sigmaXY +
                      velocityFace[1] * sigmaYY +
                      velocityFace[2] * sigmaZY
        var cp = fluid_options.gamma * fluid_options.gasConstant / 
                 (fluid_options.gamma - 1.0)
        var heatFlux = - (cp*muFace/fluid_options.prandtl)*temperature_YFace

        -- Fluxes
        c.rhoVelocityFlux[0] = sigmaXY
        c.rhoVelocityFlux[1] = sigmaYY
        c.rhoVelocityFlux[2] = sigmaZY
        c.rhoEnergyFlux = usigma - heatFlux
        -- WARNING: Add SGS terms for LES

    end
end

-- Compute viscous fluxes in Z direction
ebb Flow.AddViscousGetFluxZ (c : grid.cells)
    -- Consider first boundary element (c.zneg_depth == 1) to define down flux
    -- on first interior cell
    if c.in_interior or c.zneg_depth == 1 then
        var muFace = 0.5 * (GetDynamicViscosity(c(0,0,0).temperature) +
                            GetDynamicViscosity(c(0,0,1).temperature))
        var velocityFace    = L.vec3d({0.0, 0.0, 0.0})
        var velocityZ_XFace = L.double(0.0)
        var velocityZ_YFace = L.double(0.0)
        var velocityX_XFace = L.double(0.0)
        var velocityY_YFace = L.double(0.0)

        -- Interpolate velocity and derivatives to face
        velocityFace = 0.5 * ( c(0,0,0).velocity + c(0,0,1).velocity )
        velocityZ_XFace = 0.5 * ( c(0,0,0).velocityGradientX[2] +
                                  c(0,0,1).velocityGradientX[2] )
        velocityZ_YFace = 0.5 * ( c(0,0,0).velocityGradientY[2] +
                                  c(0,0,1).velocityGradientY[2] )
        velocityX_XFace = 0.5 * ( c(0,0,0).velocityGradientX[0] +
                                  c(0,0,1).velocityGradientX[0] )
        velocityY_YFace = 0.5 * ( c(0,0,0).velocityGradientY[1] +
                                  c(0,0,1).velocityGradientY[1] )

        -- Differentiate at face
        var velocityX_ZFace   = L.double(0.0)
        var velocityY_ZFace   = L.double(0.0)
        var velocityZ_ZFace   = L.double(0.0)
        var temperature_ZFace = L.double(0.0)

        velocityX_ZFace   = 0.5*( c(0,0,1).velocity[0] - c(0,0,0).velocity[0] )
        velocityY_ZFace   = 0.5*( c(0,0,1).velocity[1] - c(0,0,0).velocity[1] )
        velocityZ_ZFace   = 0.5*( c(0,0,1).velocity[2] - c(0,0,0).velocity[2] )
        temperature_ZFace = 0.5*( c(0,0,1).temperature - c(0,0,0).temperature )
       
        -- Half cell size due to the 0.5 above
        velocityX_ZFace   /= (grid_dz*0.5)
        velocityZ_ZFace   /= (grid_dz*0.5)
        velocityZ_ZFace   /= (grid_dz*0.5)
        temperature_ZFace /= (grid_dz*0.5)

        -- Tensor components (at face)
        var sigmaXZ = muFace * ( velocityX_ZFace + velocityZ_XFace )
        var sigmaYZ = muFace * ( velocityY_ZFace + velocityZ_YFace )
        var sigmaZZ = muFace * ( 4.0 * velocityZ_ZFace -
                                 2.0 * velocityX_XFace -
                                 2.0 * velocityY_YFace ) / 3.0
        var usigma  = velocityFace[0] * sigmaXZ +
                      velocityFace[1] * sigmaYZ +
                      velocityFace[2] * sigmaZZ
        var cp = fluid_options.gamma * fluid_options.gasConstant / 
                 (fluid_options.gamma - 1.0)
        var heatFlux = - (cp*muFace/fluid_options.prandtl)*temperature_ZFace

        -- Fluxes
        c.rhoVelocityFlux[0] = sigmaXZ
        c.rhoVelocityFlux[1] = sigmaYZ
        c.rhoVelocityFlux[2] = sigmaZZ
        c.rhoEnergyFlux = usigma - heatFlux
        -- WARNING: Add SGS terms for LES

    end
end

ebb Flow.AddViscousUpdateUsingFluxX (c : grid.cells)
    c.rhoVelocity_t += (c( 0,0,0).rhoVelocityFlux -
                        c(-1,0,0).rhoVelocityFlux)/grid_dx
    c.rhoEnergy_t   += (c( 0,0,0).rhoEnergyFlux -
                        c(-1,0,0).rhoEnergyFlux)/grid_dx
end

ebb Flow.AddViscousUpdateUsingFluxY (c : grid.cells)
    c.rhoVelocity_t += (c(0, 0,0).rhoVelocityFlux -
                        c(0,-1,0).rhoVelocityFlux)/grid_dy
    c.rhoEnergy_t   += (c(0, 0,0).rhoEnergyFlux -
                        c(0,-1,0).rhoEnergyFlux)/grid_dy
end

ebb Flow.AddViscousUpdateUsingFluxZ (c : grid.cells)
    c.rhoVelocity_t += (c(0,0, 0).rhoVelocityFlux -
                        c(0,0,-1).rhoVelocityFlux)/grid_dz
    c.rhoEnergy_t   += (c(0,0, 0).rhoEnergyFlux -
                        c(0,0,-1).rhoEnergyFlux)/grid_dz
end

---------------------
-- Particles coupling
---------------------

ebb Flow.AddParticlesCoupling (p : particles)
    if p.state == 1  and particles_options.twoWayCoupling == ON then
        -- WARNING: Assumes that deltaVelocityOverRelaxationTime and 
        -- deltaTemperatureTerm have been computed previously 
        -- (for example, when adding the flow coupling to the particles, 
        -- which should be called before in the time stepper)

        -- WARNING: Uniform grid assumption
        var cellVolume = grid_dx * grid_dy * grid_dz
        
        -- Retrieve cell containing this particle
        p.cell = grid.cell_locate(p.position)
        
        -- Add contribution to momentum and energy equations from the previously
        -- computed deltaVelocityOverRelaxationTime and deltaTemperatureTerm
        p.cell.rhoVelocity_t += -p.mass * p.deltaVelocityOverRelaxationTime / cellVolume
        p.cell.rhoEnergy_t   += -p.deltaTemperatureTerm / cellVolume
        
        -- In case we want to hold a fixed temperature by subtracting
        -- a constant heat flux from the fluid, compute the avg. 
        -- deltaTemperatureTerm to be adjusted later (note change in sign)
        Flow.averageHeatSource += p.deltaTemperatureTerm / cellVolume
    end
end

--------------
-- Holding Temperature Fixed in the presence of Radiation
--------------

ebb Flow.AddEnergySource (c : grid.cells)
  if radiation_options.zeroAvgHeatSource == ON then
    -- Remove a constant heat flux in all cells to balance with radiation.
    -- Note that this has been pre-computed before reaching this kernel (above).
    c.rhoEnergy_t += Flow.averageHeatSource
  end
end

--------------
-- Body Forces
--------------

ebb Flow.AddBodyForces (c : grid.cells)

    -- Add body forces (accelerations) to the momentum
    c.rhoVelocity_t += c.rho * flow_options.bodyForce

    -- Add source for forced isotropic turbulence (linear in velocity)
    c.rhoVelocity_t += c.rho * flow_options.turbForceCoeff * c.velocity

    -- Body force contribution to energy equations
    c.rhoEnergy_t += c.rho * L.dot(flow_options.bodyForce,c.velocity)
    Flow.averageHeatSource += -c.rho * L.dot(flow_options.bodyForce,c.velocity)
end

-------------------
-- Update functions
-------------------

-- Update flow variables using derivatives
Flow.UpdateFunctions = {}
function Flow.GenerateUpdateFunctions(relation, stage)
    -- Assumes 4th-order Runge-Kutta 
    local coeff_fun  = TimeIntegrator.coeff_function[stage]
    local coeff_time = TimeIntegrator.coeff_time[stage]
    local deltaTime  = TimeIntegrator.deltaTime
    if stage <= 3 then
        return ebb(r : relation)
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
        return ebb(r : relation)
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
    Flow.UpdateFunctions[sdx] = Flow.GenerateUpdateFunctions(grid.cells, sdx)
end

ebb Flow.UpdateAuxiliaryVelocity (c : grid.cells)
    var velocity = c.rhoVelocity / c.rho
    c.velocity = velocity
    c.kineticEnergy = 0.5 *  L.dot(velocity,velocity)
end

-- Helper function for updating the ghost fields to minimize repeated code
local ebb UpdateGhostFieldsHelper(c_bnd, c_int, sign, bnd_velocity, bnd_temperature)

  -- Temporary variables for computing new halo state
  var rho         = L.double(0.0)
  var temp_wall   = L.double(0.0)
  var temperature = L.double(0.0)
  var velocity    = L.vec3d({0.0, 0.0, 0.0})

  -- Compute the Cv for updating the Energy equation
  var cv = fluid_options.gasConstant / (fluid_options.gamma - 1.0)

  -- Compute the new velocity (including any wall conditions)
  for i = 0,3 do
    velocity[i] = c_int.rhoVelocity[i]/c_int.rho * sign[i] + bnd_velocity[i]
  end

  -- Compute the temperature for the halo cell (possibly adiabatic/isothermal)
  temp_wall = c_int.temperature
  if bnd_temperature > 0.0 then
    temp_wall = bnd_temperature
  end
  temperature = 2.0*temp_wall - c_int.temperature

  -- Recompute the density in the halo in case of a temperature change
  -- Pressure is a zero-order extrapolation
  rho = c_int.pressure / ( fluid_options.gasConstant * temperature )

  -- Update the boundary cell based on the values in the matching interior cell
  c_bnd.rhoBoundary         =  rho
  c_bnd.rhoVelocityBoundary =  rho*velocity
  c_bnd.rhoEnergyBoundary   =  rho * (cv * temperature +
                                      0.5*L.dot(velocity,velocity))
  c_bnd.velocityBoundary    =  velocity
  c_bnd.pressureBoundary    =  c_int.pressure
  c_bnd.temperatureBoundary =  temperature

end
ebb Flow.UpdateGhostFieldsStep1 (c : grid.cells)
    if c.xneg_depth > 0 then
        UpdateGhostFieldsHelper(c, c( 1,0,0), x_sign, xneg_velocity, xneg_temperature)
    end
    if c.xpos_depth > 0 then
        UpdateGhostFieldsHelper(c, c(-1,0,0), x_sign, xpos_velocity, xpos_temperature)
    end
    if c.yneg_depth > 0 then
        UpdateGhostFieldsHelper(c, c(0, 1,0), y_sign, yneg_velocity, yneg_temperature)
    end
    if c.ypos_depth > 0 then
        UpdateGhostFieldsHelper(c, c(0,-1,0), y_sign, ypos_velocity, ypos_temperature)
    end
    if c.zneg_depth > 0 then
        UpdateGhostFieldsHelper(c, c(0,0, 1), z_sign, zneg_velocity, zneg_temperature)
    end
    if c.zpos_depth > 0 then
        UpdateGhostFieldsHelper(c, c(0,0,-1), z_sign, zpos_velocity, zpos_temperature)
    end
end
ebb Flow.UpdateGhostFieldsStep2 (c : grid.cells)
    c.rho         = c.rhoBoundary
    c.rhoVelocity = c.rhoVelocityBoundary
    c.rhoEnergy   = c.rhoEnergyBoundary
    c.pressure    = c.pressureBoundary
    c.temperature = c.temperatureBoundary
end
function Flow.UpdateGhost()
    grid.cells.boundary:foreach(Flow.UpdateGhostFieldsStep1)
    grid.cells.boundary:foreach(Flow.UpdateGhostFieldsStep2)
end

-- Helper function for updating the ghost fields to minimize repeated code
local ebb UpdateGhostThermodynamicsHelper (c_bnd, c_int, bnd_temperature)

  -- Temporary variables for computing new halo state
  var temp_wall   = L.double(0.0)
  var temperature = L.double(0.0)

  -- Compute the temperature for the halo cell (possibly adiabatic/isothermal)
  temp_wall = c_int.temperature
  if bnd_temperature > 0.0 then
    temp_wall = bnd_temperature
  end
  temperature = 2.0*temp_wall - c_int.temperature

  -- Update the boundary cell based on the values in the matching interior cell
  c_bnd.pressureBoundary    = c_int.pressure
  c_bnd.temperatureBoundary = temperature

end
ebb Flow.UpdateGhostThermodynamicsStep1 (c : grid.cells)
  if c.xneg_depth > 0 then
    UpdateGhostThermodynamicsHelper(c, c( 1,0,0), xneg_temperature)
  end
  if c.xpos_depth > 0 then
    UpdateGhostThermodynamicsHelper(c, c(-1,0,0), xpos_temperature)
  end
  if c.yneg_depth > 0 then
    UpdateGhostThermodynamicsHelper(c, c(0, 1,0), yneg_temperature)
  end
  if c.ypos_depth > 0 then
    UpdateGhostThermodynamicsHelper(c, c(0,-1,0), ypos_temperature)
  end
  if c.zneg_depth > 0 then
    UpdateGhostThermodynamicsHelper(c, c(0,0, 1), zneg_temperature)
  end
  if c.zpos_depth > 0 then
    UpdateGhostThermodynamicsHelper(c, c(0,0,-1), zpos_temperature)
  end
end
ebb Flow.UpdateGhostThermodynamicsStep2 (c : grid.cells)
    if c.in_boundary then
        c.pressure    = c.pressureBoundary
        c.temperature = c.temperatureBoundary
    end
end
function Flow.UpdateGhostThermodynamics()
    grid.cells.boundary:foreach(Flow.UpdateGhostThermodynamicsStep1)
    grid.cells.boundary:foreach(Flow.UpdateGhostThermodynamicsStep2)
end

-- Helper function for updating the ghost fields to minimize repeated code
local ebb UpdateGhostVelocityHelper (c_bnd, c_int, sign, bnd_velocity)

  -- Update the boundary cell based on the values in the matching interior cell
  for i = 0,3 do
    c_bnd.velocityBoundary[i] = c_int.velocity[i] * sign[i] + bnd_velocity[i]
  end

end
ebb Flow.UpdateGhostVelocityStep1 (c : grid.cells)
  if c.xneg_depth > 0 then
    UpdateGhostVelocityHelper(c, c( 1,0,0), x_sign, xneg_velocity)
  end
  if c.xpos_depth > 0 then
    UpdateGhostVelocityHelper(c, c(-1,0,0), x_sign, xpos_velocity)
  end
  if c.yneg_depth > 0 then
    UpdateGhostVelocityHelper(c, c(0, 1,0), y_sign, yneg_velocity)
  end
  if c.ypos_depth > 0 then
    UpdateGhostVelocityHelper(c, c(0,-1,0), y_sign, ypos_velocity)
  end
  if c.zneg_depth > 0 then
    UpdateGhostVelocityHelper(c, c(0,0, 1), z_sign, zneg_velocity)
  end
  if c.zpos_depth > 0 then
    UpdateGhostVelocityHelper(c, c(0,0,-1), z_sign, zpos_velocity)
  end
end
ebb Flow.UpdateGhostVelocityStep2 (c : grid.cells)
    c.velocity = c.velocityBoundary
end
function Flow.UpdateGhostVelocity()
    grid.cells.boundary:foreach(Flow.UpdateGhostVelocityStep1)
    grid.cells.boundary:foreach(Flow.UpdateGhostVelocityStep2)
end

-- Helper function for updating the conservatives to minimize repeated code
local ebb UpdateGhostConservedHelper (c_bnd, c_int, sign, bnd_velocity,
                                        bnd_temperature)

  -- Temporary variables for computing new halo state
  var rho         = L.double(0.0)
  var temp_wall   = L.double(0.0)
  var temperature = L.double(0.0)
  var velocity    = L.vec3d({0.0, 0.0, 0.0})

  -- Compute the Cv for updating the Energy equation
  var cv = fluid_options.gasConstant / (fluid_options.gamma - 1.0)

  -- Compute the new velocity (including any wall conditions)
  var velocity = L.vec3d({0.0, 0.0, 0.0})
  for i = 0,3 do
    velocity[i] = c_int.rhoVelocity[i]/c_int.rho * sign[i] + bnd_velocity[i]
  end

  -- Compute the temperature for the halo cell (possibly adiabatic/isothermal)
  temp_wall = c_int.temperature
  if bnd_temperature > 0.0 then
    temp_wall = bnd_temperature
  end
  temperature = 2.0*temp_wall - c_int.temperature

  -- Recompute the density in the halo in case of a temperature change
  -- Pressure is a zero-order extrapolation
  rho = c_int.pressure / ( fluid_options.gasConstant * temperature )

  -- Update the boundary cell based on the values in the matching interior cell
  c_bnd.rhoBoundary         = rho
  c_bnd.rhoVelocityBoundary = rho*velocity
  c_bnd.rhoEnergyBoundary   = rho * (cv * temperature +
                                     0.5*L.dot(velocity,velocity))

end
ebb Flow.UpdateGhostConservedStep1 (c : grid.cells)
  if c.xneg_depth > 0 then
    UpdateGhostConservedHelper(c, c( 1,0,0), x_sign, xneg_velocity, xneg_temperature)
  end
  if c.xpos_depth > 0 then
    UpdateGhostConservedHelper(c, c(-1,0,0), x_sign, xpos_velocity, xpos_temperature)
  end
  if c.yneg_depth > 0 then
    UpdateGhostConservedHelper(c, c(0, 1,0), y_sign, yneg_velocity, yneg_temperature)
  end
  if c.ypos_depth > 0 then
    UpdateGhostConservedHelper(c, c(0,-1,0), y_sign, ypos_velocity, ypos_temperature)
  end
  if c.zneg_depth > 0 then
    UpdateGhostConservedHelper(c, c(0,0, 1), z_sign, zneg_velocity, zneg_temperature)
  end
  if c.zpos_depth > 0 then
    UpdateGhostConservedHelper(c, c(0,0,-1), z_sign, zpos_velocity, zpos_temperature)
  end
end
ebb Flow.UpdateGhostConservedStep2 (c : grid.cells)
    c.rho         = c.rhoBoundary
    c.rhoVelocity = c.rhoVelocityBoundary
    c.rhoEnergy   = c.rhoEnergyBoundary
end
function Flow.UpdateGhostConserved()
    grid.cells.boundary:foreach(Flow.UpdateGhostConservedStep1)
    grid.cells.boundary:foreach(Flow.UpdateGhostConservedStep2)
end

ebb Flow.UpdateAuxiliaryThermodynamics (c : grid.cells)
  var kineticEnergy = 0.5 * c.rho * L.dot(c.velocity,c.velocity)
  var pressure  = (fluid_options.gamma - 1.0) *( c.rhoEnergy - kineticEnergy )
  c.pressure    = pressure
  c.temperature = pressure / ( fluid_options.gasConstant * c.rho )
end

---------------------
-- Velocity gradients
---------------------

-- WARNING: non-uniform grid assumption
ebb Flow.ComputeVelocityGradientX (c : grid.cells)
  c.velocityGradientX = 0.5*(c(1,0,0).velocity - c(-1,0,0).velocity)/grid_dx
end

-- WARNING: non-uniform grid assumption
ebb Flow.ComputeVelocityGradientY (c : grid.cells)
  c.velocityGradientY = 0.5*(c(0,1,0).velocity - c(0,-1,0).velocity)/grid_dy
end

-- WARNING: non-uniform grid assumption
ebb Flow.ComputeVelocityGradientZ (c : grid.cells)
  c.velocityGradientZ = 0.5*(c(0,0,1).velocity - c(0,0,-1).velocity)/grid_dz
end

-- Helper function for updating the boundary gradients to minimize repeated code
local ebb UpdateGhostVelocityGradientHelper (c_bnd, c_int, sign)

  -- Apply sign change and copy gradients from interior to boundary
  for i = 0,3 do
    c_bnd.velocityGradientXBoundary[i] = sign[i] * c_int.velocityGradientX[i]
    c_bnd.velocityGradientYBoundary[i] = sign[i] * c_int.velocityGradientY[i]
    c_bnd.velocityGradientZBoundary[i] = sign[i] * c_int.velocityGradientZ[i]
  end

end
ebb Flow.UpdateGhostVelocityGradientStep1 (c : grid.cells)
    if c.xneg_depth > 0 then
      UpdateGhostVelocityGradientHelper(c, c( 1,0,0), x_sign)
    end
    if c.xpos_depth > 0 then
      UpdateGhostVelocityGradientHelper(c, c(-1,0,0), x_sign)
    end
    if c.yneg_depth > 0 then
      UpdateGhostVelocityGradientHelper(c, c(0, 1,0), y_sign)
    end
    if c.ypos_depth > 0 then
      UpdateGhostVelocityGradientHelper(c, c(0,-1,0), y_sign)
    end
    if c.zneg_depth > 0 then
      UpdateGhostVelocityGradientHelper(c, c(0,0, 1), z_sign)
    end
    if c.zpos_depth > 0 then
      UpdateGhostVelocityGradientHelper(c, c(0,0,-1), z_sign)
    end
end
ebb Flow.UpdateGhostVelocityGradientStep2 (c : grid.cells)
    if c.in_boundary then
        c.velocityGradientX = c.velocityGradientXBoundary
        c.velocityGradientY = c.velocityGradientYBoundary
        c.velocityGradientZ = c.velocityGradientZBoundary
    end
end

-- Calculation of spectral radii for clf-based delta time
local maxConvectiveSpectralRadius     = L.Global(L.double, 0.0)
local maxViscousSpectralRadius        = L.Global(L.double, 0.0)
local maxHeatConductionSpectralRadius = L.Global(L.double, 0.0)

-- WARNING: update cellVolume computation for non-uniform grids
local dXYZInverseSquare = L.Constant(L.double,
                                     1.0/grid_dx:get() * 1.0/grid_dx:get() +
                                     1.0/grid_dy:get() * 1.0/grid_dy:get() +
                                     1.0/grid_dz:get() * 1.0/grid_dz:get())
local ebb calculateConvectiveSpectralRadius     ( c : grid.cells )
  -- Convective spectral radii
  -- WARNING: uniform grid assumption
  c.convectiveSpectralRadius = 
   (L.fabs(c.velocity[0])/grid_dx  +
    L.fabs(c.velocity[1])/grid_dy  +
    L.fabs(c.velocity[2])/grid_dz  +
    GetSoundSpeed(c.temperature) * L.sqrt(dXYZInverseSquare))

  maxConvectiveSpectralRadius max= c.convectiveSpectralRadius    
end
local ebb calculateViscousSpectralRadius        ( c : grid.cells )
  -- Viscous spectral radii (including sgs model component)
  var dynamicViscosity = GetDynamicViscosity(c.temperature)
  var eddyViscosity = c.sgsEddyViscosity
  c.viscousSpectralRadius =
   (2.0 * ( dynamicViscosity + eddyViscosity ) /
    c.rho * dXYZInverseSquare) * 4.0

  maxViscousSpectralRadius max= c.viscousSpectralRadius       
end
local ebb calculateHeatConductionSpectralRadius ( c : grid.cells )
  var dynamicViscosity  = GetDynamicViscosity(c.temperature)

  -- Heat conduction spectral radii (including sgs model component)
  var cv = fluid_options.gasConstant / (fluid_options.gamma - 1.0)
  var cp = fluid_options.gamma * cv

  var kappa = cp / fluid_options.prandtl *  dynamicViscosity
  
  c.heatConductionSpectralRadius = 
     ((kappa + c.sgsEddyKappa) / (cv * c.rho) * dXYZInverseSquare) * 4.0
  maxHeatConductionSpectralRadius max= c.heatConductionSpectralRadius
end
function Flow.CalculateSpectralRadii(cells)
  cells:foreach(calculateConvectiveSpectralRadius)
  cells:foreach(calculateViscousSpectralRadius)
  cells:foreach(calculateHeatConductionSpectralRadius)
end


-------------
-- Statistics
-------------

-- WARNING: update cellVolume computation for non-uniform grids
local cellVolume = L.Constant(L.double,
                              grid_dx:get() * grid_dy:get() * grid_dz:get())
local ebb numberOfInteriorCells ( c : grid.cells )
  Flow.numberOfInteriorCells    += 1
end
local ebb areaInterior          ( c : grid.cells )
  Flow.areaInterior             += cellVolume
end
local ebb averagePressure       ( c : grid.cells )
  Flow.averagePressure          += c.pressure * cellVolume
end
local ebb averageTemperature    ( c : grid.cells )
  Flow.averageTemperature       += c.temperature * cellVolume
end
local ebb averageKineticEnergy  ( c : grid.cells )
  Flow.averageKineticEnergy     += c.kineticEnergy * cellVolume
end
local ebb minTemperature        ( c : grid.cells )
  Flow.minTemperature         min= c.temperature
end
local ebb maxTemperature        ( c : grid.cells )
  Flow.maxTemperature         max= c.temperature
end
function Flow.IntegrateQuantities(cells)
  cells:foreach(numberOfInteriorCells)
  cells:foreach(areaInterior         )
  cells:foreach(averagePressure      )
  cells:foreach(averageTemperature   )
  cells:foreach(averageKineticEnergy )
  cells:foreach(minTemperature       )
  cells:foreach(maxTemperature       )
end


---------
-- Output
---------

local function value_tostring(val)
  if type(val) == 'table' then
    local s = tostring(val[1])
    for i=2,#val do s = s..' '..tostring(val[i]) end
    return s
  end
  return tostring(val)
end

local function value_tostring_comma(val)
  if type(val) == 'table' then
    local s = tostring(val[1])
    for i=2,#val do s = s..', '..tostring(val[i]) end
    return s
  end
  return tostring(val)
end


----------------
-- Visualization
----------------

-- functions to draw particles and velocity for debugging purpose
ebb Flow.DrawFunction (c : grid.cells)
    --var xMax = L.double(grid_options.xWidth)
    --var yMax = L.double(grid_options.yWidth)
    --var zMax = L.double(grid_options.zWidth)
    var xMax = 1.0
    var yMax = 1.0
    var zMax = 1.0
    if c(0,0,0).center[0] < grid_originX+grid_dx then
      var posA : L.vec3d = { c(0,0,0).center[0]/xMax,
                             c(0,0,0).center[1]/yMax, 
                             c(0,0,0).center[2]/zMax }
      var posB : L.vec3d = { c(0,0,1).center[0]/xMax,
                             c(0,0,1).center[1]/yMax,
                             c(0,0,1).center[2]/zMax }
      var posC : L.vec3d = { c(0,1,1).center[0]/xMax,
                             c(0,1,1).center[1]/yMax, 
                             c(0,1,1).center[2]/zMax }
      var posD : L.vec3d = { c(0,1,0).center[0]/xMax,
                             c(0,1,0).center[1]/yMax,
                             c(0,1,0).center[2]/zMax }
      var value =
        (c(0,0,0).temperature + 
         c(0,1,0).temperature +
         c(0,0,1).temperature +
         c(0,1,1).temperature) / 4.0
      var minValue = Flow.minTemperature
      var maxValue = Flow.maxTemperature
      -- compute a display value in the range 0.0 to 1.0 from the value
      var scale = (value - minValue)/(maxValue - minValue)
      vdb.color((1.0-scale)*white)
      vdb.triangle(posA, posB, posC)
      vdb.triangle(posA, posD, posC)
    elseif c(0,0,0).center[1] < grid_originY+grid_dy then
      var posA : L.vec3d = { c(0,0,0).center[0]/xMax,
                             c(0,0,0).center[1]/yMax, 
                             c(0,0,0).center[2]/zMax }
      var posB : L.vec3d = { c(0,0,1).center[0]/xMax,
                             c(0,0,1).center[1]/yMax,
                             c(0,0,1).center[2]/zMax }
      var posC : L.vec3d = { c(1,0,1).center[0]/xMax,
                             c(1,0,1).center[1]/yMax, 
                             c(1,0,1).center[2]/zMax }
      var posD : L.vec3d = { c(1,0,0).center[0]/xMax,
                             c(1,0,0).center[1]/yMax,
                             c(1,0,0).center[2]/zMax }
      var value =
        (c(0,0,0).temperature + 
         c(1,0,0).temperature +
         c(0,0,1).temperature +
         c(1,0,1).temperature) / 4.0
      var minValue = Flow.minTemperature
      var maxValue = Flow.maxTemperature
      -- compute a display value in the range 0.0 to 1.0 from the value
      var scale = (value - minValue)/(maxValue - minValue)
      vdb.color((1.0-scale)*white)
      vdb.triangle(posA, posB, posC)
      vdb.triangle(posA, posD, posC)
    elseif c(0,0,0).center[2] < grid_originZ+grid_dz then
      var posA : L.vec3d = { c(0,0,0).center[0]/xMax,
                             c(0,0,0).center[1]/yMax, 
                             c(0,0,0).center[2]/zMax }
      var posB : L.vec3d = { c(0,1,0).center[0]/xMax,
                             c(0,1,0).center[1]/yMax,
                             c(0,1,0).center[2]/zMax }
      var posC : L.vec3d = { c(1,1,0).center[0]/xMax,
                             c(1,1,0).center[1]/yMax, 
                             c(1,1,0).center[2]/zMax }
      var posD : L.vec3d = { c(1,0,0).center[0]/xMax,
                             c(1,0,0).center[1]/yMax,
                             c(1,0,0).center[2]/zMax }
      var value =
        (c(0,0,0).temperature + 
         c(1,0,0).temperature +
         c(0,1,0).temperature +
         c(1,1,0).temperature) / 4.0
      var minValue = Flow.minTemperature
      var maxValue = Flow.maxTemperature
      -- compute a display value in the range 0.0 to 1.0 from the value
      var scale = (value - minValue)/(maxValue - minValue)
      vdb.color((1.0-scale)*white)
      vdb.triangle(posA, posB, posC)
      vdb.triangle(posA, posC, posD)
    end
end

------------
-- PARTICLES
------------

-- Locate particles in dual cells
ebb Particles.Locate (p : particles)
    p.dual_cell = grid.dual_locate(p.position)
end

-- Initialize temporaries for time stepper
ebb Particles.InitializeTemporaries (p : particles)
    if p.state == 1 then
        p.position_old    = p.position
        p.velocity_old    = p.velocity
        p.temperature_old = p.temperature
        p.position_new    = p.position
        p.velocity_new    = p.velocity
        p.temperature_new = p.temperature
    end
end

----------------
-- Flow Coupling
----------------

-- Initialize time derivative for each stage of time stepper
ebb Particles.InitializeTimeDerivatives (p : particles)
    if p.state == 1 then
        p.position_t = L.vec3d({0, 0, 0})
        p.velocity_t = L.vec3d({0, 0, 0})
        p.temperature_t = L.double(0)
    end
end

-- Update particle fields based on flow fields
ebb Particles.AddFlowCoupling (p: particles)
  if p.state == 1 then
    
    -- Locate the dual cell within which this particle is located
    p.dual_cell = grid.dual_locate(p.position)
    var flowDensity     = L.double(0)
    var flowVelocity    = L.vec3d({0, 0, 0})
    var flowTemperature = L.double(0)
    var flowDynamicViscosity = L.double(0)

    -- Trilinear interpolation for the flow quantities
    flowDensity     = InterpolateTriRho(p.dual_cell, p.position)
    flowVelocity    = InterpolateTriVelocity(p.dual_cell, p.position)
    flowTemperature = InterpolateTriTemperature(p.dual_cell, p.position)
    flowDynamicViscosity = GetDynamicViscosity(flowTemperature)
    
    -- Update the particle position using the current velocity
    if particles_options.particleType == Particles.Fixed then
      -- Don't move the particle
      elseif particles_options.particleType == Particles.Free then
      p.position_t    += p.velocity
    end
    
    -- Relaxation time for small particles
    -- - particles Reynolds number (set to zero for Stokesian)
    var particleReynoldsNumber = 0.0
    --(p.density * norm(flowVelocity - p.velocity) * p.diameter) / flowDynamicViscosity
    var relaxationTime =
    ( p.density * L.pow(p.diameter,2)/(18.0 * flowDynamicViscosity))/
    ( 1.0 + 0.15 * L.pow(particleReynoldsNumber,0.687) )
    
    p.deltaVelocityOverRelaxationTime = (flowVelocity - p.velocity) / relaxationTime
    
    p.deltaTemperatureTerm = pi * L.pow(p.diameter, 2) * particles_options.convective_coefficient * (flowTemperature - p.temperature)
    
    -- Update the particle velocity and temperature
    if particles_options.particleType == Particles.Fixed then
      p.velocity_t  = {0.0,0.0,0.0} -- Don't move the particle
      elseif particles_options.particleType == Particles.Free then
      p.velocity_t += p.deltaVelocityOverRelaxationTime
    end
    p.temperature_t += p.deltaTemperatureTerm / (p.mass * particles_options.heat_capacity)
    
  end
end

--------------
-- Body forces
--------------

ebb Particles.AddBodyForces (p : particles)
    if p.state == 1 and particles_options.particleType == Particles.Free then
        p.velocity_t += particles_options.bodyForce
    end
end

------------
-- Radiation
------------

ebb Particles.AddRadiation (p : particles)
    if p.state == 1 and radiation_options.radiationType == ON then
        -- Calculate absorbed radiation intensity considering optically thin
        -- particles, for a collimated radiation source with negligible 
        -- blackbody self radiation
        var absorbedRadiationIntensity =
          particles_options.absorptivity *
          radiation_options.radiationIntensity * p.cross_section_area

        -- Add contribution to particle temperature time evolution
        p.temperature_t += absorbedRadiationIntensity /
                           (p.mass * particles_options.heat_capacity)
    end
end

-- Set particle velocities to underlying flow velocity for initialization
ebb Particles.SetVelocitiesToFlow (p: particles)

    -- Locate the dual cell within which this particle is located
    p.dual_cell = grid.dual_locate(p.position)
    var flowDensity     = L.double(0)
    var flowVelocity    = L.vec3d({0, 0, 0})
    var flowTemperature = L.double(0)
    var flowDynamicViscosity = L.double(0)

    -- Trilinear interpolation
    flowDensity     = InterpolateTriRho(p.dual_cell, p.position)
    flowVelocity    = InterpolateTriVelocity(p.dual_cell, p.position)
    flowTemperature = InterpolateTriTemperature(p.dual_cell, p.position)
    flowDynamicViscosity = GetDynamicViscosity(flowTemperature)

    -- Update the particle velocity
    if (particles_options.particleType == Particles.Fixed) then
      p.velocity = {0.0,0.0,0.0} -- Don't move the particle
      elseif (particles_options.initParticles == Particles.Restart) then
      -- Do nothing, as we loaded the velocity from a restart
    elseif particles_options.particleType == Particles.Free then
      p.velocity = flowVelocity
    end

end

-- Update particle variables using derivatives
Particles.UpdateFunctions = {}
function Particles.GenerateUpdateFunctions(relation, stage)
    local coeff_fun  = TimeIntegrator.coeff_function[stage]
    local coeff_time = TimeIntegrator.coeff_time[stage]
    local deltaTime  = TimeIntegrator.deltaTime
    if stage <= 3 then
        return ebb(r : relation)
            if r.state == 1 then
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
        end
    elseif stage == 4 then
        return ebb(r : relation)
            if r.state == 1 then
              r.position = r.position_new +
                 coeff_fun * deltaTime * r.position_t
              r.velocity = r.velocity_new +
                 coeff_fun * deltaTime * r.velocity_t
              r.temperature = r.temperature_new +
                 coeff_fun * deltaTime * r.temperature_t
            end
        end
    end
end
for i = 1, 4 do
    Particles.UpdateFunctions[i] =
        Particles.GenerateUpdateFunctions(particles, i)
end

ebb Particles.UpdateAuxiliaryStep1 (p : particles)
    if p.state == 1 then

        -- Initialize position and velocity before we check for wall collisions
        
        p.position_ghost[0]   = p.position[0]
        p.position_ghost[1]   = p.position[1]
        p.position_ghost[2]   = p.position[2]
        p.velocity_ghost[0]   = p.velocity[0]
        p.velocity_ghost[1]   = p.velocity[1]
        p.velocity_ghost[2]   = p.velocity[2]
        p.velocity_t_ghost[0] = p.velocity_t[0]
        p.velocity_t_ghost[1] = p.velocity_t[1]
        p.velocity_t_ghost[2] = p.velocity_t[2]
        
        -- Check here for particles exiting the domain. For periodic
        -- boundaries, the particle is transported to the matching periodic
        -- face. For symmetry or wall boundaries, an elastic collision is
        -- assumed. To start, the collision is perfectly elastic.
        
        -- Left X boundary
        if p.position[0] < gridOriginInteriorX then
          if grid_options.xBCLeftParticles == Particles.Permeable then
            p.position_ghost[0] = p.position[0] + grid_options.xWidth
          elseif grid_options.xBCLeftParticles == Particles.Solid then

            -- Set the position to be on the wall
            p.position_ghost[0] = gridOriginInteriorX

            -- Apply an impulse to kick particle away from the wall
            var impulse = -(1.0+particles_options.restitution_coefficient)*p.velocity[0]
            if impulse <= 0 then
              p.velocity_ghost[0] += impulse
            end

            -- Add a contact force in case particle rests on the wall
            var contact_force = -1.0*p.velocity_t[0]

            -- To prevent sticky walls, only add contact force if current
            -- force would push the particle through the wall
            if contact_force > 0 then
              p.velocity_t_ghost[0] += contact_force
            end

          end
        end
        
        -- Right X boundary
        if p.position[0] > gridOriginInteriorX + grid_options.xWidth then
          if grid_options.xBCRightParticles == Particles.Permeable then
            p.position_ghost[0] = p.position[0] - grid_options.xWidth
          elseif grid_options.xBCRightParticles == Particles.Solid then

            -- Set the position to be on the wall
            p.position_ghost[0] = gridOriginInteriorX + grid_options.xWidth

            -- Apply an impulse to kick particle away from the wall
            var impulse = -(1.0+particles_options.restitution_coefficient)*p.velocity[0]
            if impulse >= 0 then
              p.velocity_ghost[0] += impulse
            end

            -- Add a contact force in case particle rests on the wall
            var contact_force = -1.0*p.velocity_t[0]

            -- To prevent sticky walls, only add contact force if current
            -- force would push the particle through the wall
            if contact_force < 0 then
              p.velocity_t_ghost[0] += contact_force
            end
    
          end
        end
        
        -- Left Y boundary
        if p.position[1] < gridOriginInteriorY then
          if grid_options.yBCLeftParticles == Particles.Permeable then
            p.position_ghost[1] = p.position[1] + grid_options.yWidth
          elseif grid_options.yBCLeftParticles == Particles.Solid then
          
            -- Set the position to be on the wall
            p.position_ghost[1] = gridOriginInteriorY
            
            -- Apply an impulse to kick particle away from the wall
            var impulse = -(1.0+particles_options.restitution_coefficient)*p.velocity[1]
            if impulse <= 0 then
            p.velocity_ghost[1] += impulse
            end
            
            -- Add a contact force in case particle rests on the wall
            var contact_force = -1.0*p.velocity_t[1]
            
            -- To prevent sticky walls, only add contact force if current
            -- force would push the particle through the wall
            if contact_force > 0 then
            p.velocity_t_ghost[1] += contact_force
            end
          
          end
          
        end
        
        -- Right Y boundary
        if p.position[1] > gridOriginInteriorY + grid_options.yWidth then
          if grid_options.yBCRightParticles == Particles.Permeable then
            p.position_ghost[1] = p.position[1] - grid_options.yWidth
          elseif grid_options.yBCRightParticles == Particles.Solid then

            -- Set the position to be on the wall
            p.position_ghost[1] = gridOriginInteriorY + grid_options.yWidth

            -- Apply an impulse to kick particle away from the wall
            var impulse = -(1.0+particles_options.restitution_coefficient)*p.velocity[1]
            if impulse >= 0 then
              p.velocity_ghost[1] += impulse
            end

            -- Add a contact force in case particle rests on the wall
            var contact_force = -1.0*p.velocity_t[1]

            -- To prevent sticky walls, only add contact force if current
            -- force would push the particle through the wall
            if contact_force < 0 then
              p.velocity_t_ghost[1] += contact_force
            end

          end
        end
        
        -- Left Z boundary
        if p.position[2] < gridOriginInteriorZ then
          if grid_options.zBCLeftParticles == Particles.Permeable then
            p.position_ghost[2] = p.position[2] + grid_options.zWidth
          elseif grid_options.zBCLeftParticles == Particles.Solid then

            -- Set the position to be on the wall
            p.position_ghost[2] = gridOriginInteriorZ

            -- Apply an impulse to kick particle away from the wall
            var impulse = -(1.0+particles_options.restitution_coefficient)*p.velocity[2]
            if impulse <= 0 then
              p.velocity_ghost[2] += impulse
            end

            -- Add a contact force in case particle rests on the wall
            var contact_force = -1.0*p.velocity_t[2]

            -- To prevent sticky walls, only add contact force if current
            -- force would push the particle through the wall
            if contact_force > 0 then
              p.velocity_t_ghost[2] += contact_force
            end

          end
        end
        
        -- Right Z boundary
        if p.position[2] > gridOriginInteriorZ + grid_options.zWidth then
          if grid_options.zBCRightParticles == Particles.Permeable then
            p.position_ghost[2] = p.position[2] - grid_options.zWidth
          elseif grid_options.zBCRightParticles == Particles.Solid then

            -- Set the position to be on the wall
            p.position_ghost[2] = gridOriginInteriorZ + grid_options.zWidth

            -- Apply an impulse to kick particle away from the wall
            var impulse = -(1.0+particles_options.restitution_coefficient)*p.velocity[2]
            if impulse >= 0 then
              p.velocity_ghost[2] += impulse
            end

            -- Add a contact force in case particle rests on the wall
            var contact_force = -1.0*p.velocity_t[2]

            -- To prevent sticky walls, only add contact force if current
            -- force would push the particle through the wall
            if contact_force < 0 then
              p.velocity_t_ghost[2] += contact_force
            end
    
          end
        end
        
    end
end
ebb Particles.UpdateAuxiliaryStep2 (p : particles)
    if p.state == 1 then
        p.position   = p.position_ghost
        p.velocity   = p.velocity_ghost
        p.velocity_t = p.velocity_t_ghost
    end
end

---------
-- Feeder
---------

-- Particles feeder
ebb Particles.Feed(p: particles)

    if p.state == 0 then

      p.position[0] = 0
      p.position[1] = 0
      p.position[2] = 0
      p.velocity[0] = 0
      p.velocity[1] = 0
      p.velocity[2] = 0
      p.state = 0

      -- Initialize based on feeder type
      if particles_options.feederType == 
           Particles.FeederAtStartTimeInRandomBox then

        -- Particles randomly distributed inside box limits defined 
        -- by options
        -- Specialize feederParams from options
        var centerX   = particles_options.feederParams[0]
        var centerY   = particles_options.feederParams[1]
        var centerZ   = particles_options.feederParams[2]
        var widthX    = particles_options.feederParams[3]
        var widthY    = particles_options.feederParams[4]
        var widthZ    = particles_options.feederParams[5]

        p.position[0] = centerX + (C.rand_unity()-0.5) * widthX
        p.position[1] = centerY + (C.rand_unity()-0.5) * widthY
        p.position[2] = centerZ + (C.rand_unity()-0.5) * widthZ
        p.state = 1
                        
      elseif particles_options.feederType == 
               Particles.FeederOverTimeInRandomBox then

        -- Specialize feederParams from options
        var injectorBox_centerX   = particles_options.feederParams[0]
        var injectorBox_centerY   = particles_options.feederParams[1]
        var injectorBox_centerZ   = particles_options.feederParams[2]
        var injectorBox_widthX    = particles_options.feederParams[3]
        var injectorBox_widthY    = particles_options.feederParams[4]
        var injectorBox_widthZ    = particles_options.feederParams[5]
        var injectorBox_velocityX = particles_options.feederParams[6]
        var injectorBox_velocityY = particles_options.feederParams[7]
        var injectorBox_velocityZ = particles_options.feederParams[8]
        var injectorBox_particlesPerTimeStep = particles_options.feederParams[9]
        -- Inject particle if matching timeStep requirements
        if L.floor(p.id/injectorBox_particlesPerTimeStep) ==
           TimeIntegrator.timeStep then
            p.position[0] = injectorBox_centerX +
                            (C.rand_unity()-0.5) * injectorBox_widthX
            p.position[1] = injectorBox_centerY +
                            (C.rand_unity()-0.5) * injectorBox_widthY
            p.position[2] = injectorBox_centerZ +
                            (C.rand_unity()-0.5) * injectorBox_widthZ
            p.velocity[0] = injectorBox_velocityX
            p.velocity[1] = injectorBox_velocityY
            p.velocity[2] = injectorBox_velocityZ
            p.state = 1
        end

      elseif particles_options.feederType == 
               Particles.FeederUQCase then

        -- Specialize feederParams from options
        -- Injector A
        var injectorA_centerX   = particles_options.feederParams[0]
        var injectorA_centerY   = particles_options.feederParams[1]
        var injectorA_centerZ   = particles_options.feederParams[2]
        var injectorA_widthX    = particles_options.feederParams[3]
        var injectorA_widthY    = particles_options.feederParams[4]
        var injectorA_widthZ    = particles_options.feederParams[5]
        var injectorA_velocityX = particles_options.feederParams[6]
        var injectorA_velocityY = particles_options.feederParams[7]
        var injectorA_velocityZ = particles_options.feederParams[8]
        var injectorA_particlesPerTimeStep = particles_options.feederParams[9]
        -- Injector B
        var injectorB_centerX   = particles_options.feederParams[10]
        var injectorB_centerY   = particles_options.feederParams[11]
        var injectorB_centerZ   = particles_options.feederParams[12]
        var injectorB_widthX    = particles_options.feederParams[13]
        var injectorB_widthY    = particles_options.feederParams[14]
        var injectorB_widthZ    = particles_options.feederParams[15]
        var injectorB_velocityX = particles_options.feederParams[16]
        var injectorB_velocityY = particles_options.feederParams[17]
        var injectorB_velocityZ = particles_options.feederParams[18]
        var injectorB_particlesPerTimeStep = particles_options.feederParams[19]
        var numberOfParticlesInA = 
             L.floor(particles_options.num*injectorA_particlesPerTimeStep/
             (injectorA_particlesPerTimeStep+injectorB_particlesPerTimeStep))
        var numberOfParticlesInB = 
             L.ceil(particles_options.num*injectorB_particlesPerTimeStep/
             (injectorA_particlesPerTimeStep+injectorB_particlesPerTimeStep))
        -- Inject particles at injectorA if matching timeStep requirements
        if L.floor(p.id/injectorA_particlesPerTimeStep) ==
          TimeIntegrator.timeStep then
            p.position[0] = injectorA_centerX +
                            (C.rand_unity()-0.5) * injectorA_widthX
            p.position[1] = injectorA_centerY +
                            (C.rand_unity()-0.5) * injectorA_widthY
            p.position[2] = injectorA_centerZ +
                            (C.rand_unity()-0.5) * injectorA_widthZ
            p.velocity[0] = injectorA_velocityX
            p.velocity[1] = injectorA_velocityY
            p.velocity[2] = injectorA_velocityZ
            p.state = 1
            p.groupID = 0
        end
        -- Inject particles at injectorB if matching timeStep requirements
        -- (if injectorA has injected this particle at this timeStep, it
        -- will get over-riden by injector B; this can only occur at the same
        -- timeStep, as otherwise p.state is already set to 1 and the program 
        -- will not enter this route)
        if L.floor((p.id-numberOfParticlesInA)/
                       injectorB_particlesPerTimeStep) ==
          TimeIntegrator.timeStep then
            p.position[0] = injectorB_centerX +
                            (C.rand_unity()-0.5) * injectorB_widthX
            p.position[1] = injectorB_centerY +
                            (C.rand_unity()-0.5) * injectorB_widthY
            p.position[2] = injectorB_centerZ +
                            (C.rand_unity()-0.5) * injectorB_widthZ
            p.velocity[0] = injectorB_velocityX
            p.velocity[1] = injectorB_velocityY
            p.velocity[2] = injectorB_velocityZ
            p.state = 1
            p.groupID = 1
        end

      end

    end

end

------------
-- Collector 
------------

-- Particles collector 
ebb Particles.Collect (p: particles)

    if p.state == 1 then

      if particles_options.collectorType == 
           Particles.CollectorOutOfBox then

        -- Specialize collectorParams from options
        var minX = particles_options.collectorParams[0]
        var minY = particles_options.collectorParams[1]
        var minZ = particles_options.collectorParams[2]
        var maxX = particles_options.collectorParams[3]
        var maxY = particles_options.collectorParams[4]
        var maxZ = particles_options.collectorParams[5]
        if p.position[0] < minX or
           p.position[0] > maxX or
           p.position[1] < minY or
           p.position[1] > maxY or
           p.position[2] < minZ or
           p.position[2] > maxZ then
          p.state = 2
        end
                       
      end

    end

end


-------------
-- Statistics
-------------

ebb Particles.IntegrateQuantities (p : particles)
    if p.state == 1 then
        Particles.averageTemperature += p.temperature
    end
end


----------------
-- Visualization
----------------

ebb Particles.DrawFunction (p : particles)
    --var xMax = L.double(grid_options.xWidth)
    --var yMax = L.double(grid_options.yWidth)
    --var zMax = L.double(grid_options.zWidth)
    var xMax = 1.0
    var yMax = 1.0
    var zMax = 1.0
    --var scale = p.temperature/particles_options.initialTemperature
    --var scale = 0.5 + 0.5*p.groupID
    --vdb.color(scale*blue)
    if p.groupID == 0 then
      vdb.color(red)
    elseif p.groupID == 1 then
      vdb.color(blue)
    else
      vdb.color(green)
    end
    var pos : L.vec3d = { p.position[0]/xMax,
                          p.position[1]/yMax,
                          p.position[2]/zMax }
    vdb.point(pos)
    var vel = p.velocity
    var v = L.vec3d({ vel[0], vel[1], vel[2] })
    vdb.line(pos, pos+0.1*v)
end


-----------------------------------------------------------------------------
--[[                                MAIN FUNCTIONS                       ]]--
-----------------------------------------------------------------------------

-------
-- FLOW
-------

function Flow.AddInviscid()
    grid.cells:foreach(Flow.AddInviscidInitialize)
    grid.cells:foreach(Flow.AddInviscidGetFluxX)
    grid.cells.interior:foreach(Flow.AddInviscidUpdateUsingFluxX)
    grid.cells:foreach(Flow.AddInviscidGetFluxY)
    grid.cells.interior:foreach(Flow.AddInviscidUpdateUsingFluxY)
    grid.cells:foreach(Flow.AddInviscidGetFluxZ)
    grid.cells.interior:foreach(Flow.AddInviscidUpdateUsingFluxZ)
end

function Flow.UpdateGhostVelocityGradient()
    grid.cells:foreach(Flow.UpdateGhostVelocityGradientStep1)
    grid.cells:foreach(Flow.UpdateGhostVelocityGradientStep2)
end

function Flow.AddViscous()
    grid.cells:foreach(Flow.AddViscousGetFluxX)
    grid.cells.interior:foreach(Flow.AddViscousUpdateUsingFluxX)
    grid.cells:foreach(Flow.AddViscousGetFluxY)
    grid.cells.interior:foreach(Flow.AddViscousUpdateUsingFluxY)
    grid.cells:foreach(Flow.AddViscousGetFluxZ)
    grid.cells.interior:foreach(Flow.AddViscousUpdateUsingFluxZ)
end

function Flow.Update(stage)
    grid.cells:foreach(Flow.UpdateFunctions[stage])
end

function Flow.ComputeVelocityGradients()
    grid.cells.interior:foreach(Flow.ComputeVelocityGradientX)
    grid.cells.interior:foreach(Flow.ComputeVelocityGradientY)
    grid.cells.interior:foreach(Flow.ComputeVelocityGradientZ)
end

function Flow.UpdateAuxiliaryVelocityConservedAndGradients()
    grid.cells.interior:foreach(Flow.UpdateAuxiliaryVelocity)
    Flow.UpdateGhostConserved()
    Flow.UpdateGhostVelocity()
    Flow.ComputeVelocityGradients()
end

function Flow.UpdateAuxiliary()
    Flow.UpdateAuxiliaryVelocityConservedAndGradients()
    grid.cells.interior:foreach(Flow.UpdateAuxiliaryThermodynamics)
    Flow.UpdateGhostThermodynamics()
end

------------
-- PARTICLES
------------

function Particles.Update(stage)
    particles:foreach(Particles.UpdateFunctions[stage])
end

function Particles.UpdateAuxiliary()
    particles:foreach(Particles.UpdateAuxiliaryStep1)
    particles:foreach(Particles.UpdateAuxiliaryStep2)
end

------------------
-- TIME INTEGRATOR
------------------

function TimeIntegrator.SetupTimeStep()
    --particles:foreach(Particles.Feed)
    -- for now, feeding of particles is disabled
    grid.cells:foreach(Flow.InitializeTemporaries)
    particles:foreach(Particles.InitializeTemporaries)
end

function TimeIntegrator.ConcludeTimeStep()
    particles:foreach(Particles.Collect)
end

function TimeIntegrator.InitializeTimeDerivatives()
    grid.cells:foreach(Flow.InitializeTimeDerivatives)
    particles:foreach(Particles.InitializeTimeDerivatives)
end

function TimeIntegrator.UpdateAuxiliary()
    Flow.UpdateAuxiliary()
    Particles.UpdateAuxiliary()
end

function TimeIntegrator.UpdateTime(timeOld, stage)
    TimeIntegrator.simTime:set(timeOld +
                               TimeIntegrator.coeff_time[stage] *
                               TimeIntegrator.deltaTime:get())
end

function TimeIntegrator.InitializeVariables()
    grid.cells:foreach(Flow.InitializeCenterCoordinates)
    grid.cells.interior:foreach(Flow.InitializeCellRindLayer)
    grid.vertices:foreach(Flow.InitializeVertexCoordinates)
    grid.vertices.interior:foreach(Flow.InitializeVertexRindLayer)
    grid.cells.interior:foreach(Flow.InitializePrimitives)
    grid.cells.interior:foreach(Flow.UpdateConservedFromPrimitive)
    Flow.UpdateAuxiliary()
    Flow.UpdateGhost()
    if particles_options.initParticles ~= Particles.Restart then
      particles:foreach(Particles.Feed)
    end
    particles:foreach(Particles.Locate)
    particles:foreach(Particles.SetVelocitiesToFlow)
end

function TimeIntegrator.ComputeDFunctionDt()
    Flow.AddInviscid()
    Flow.UpdateGhostVelocityGradient()
    Flow.AddViscous()
    Flow.averageHeatSource:set(0.0)
    grid.cells.interior:foreach(Flow.AddBodyForces)
    particles:foreach(Particles.AddFlowCoupling)
    particles:foreach(Particles.AddBodyForces)
    particles:foreach(Particles.AddRadiation)
    particles:foreach(Flow.AddParticlesCoupling)
    -- In case we want to hold flow temp fixed with radiation active
    --print(Flow.averageHeatSource:get())
    Flow.averageHeatSource:set(Flow.averageHeatSource:get()/
                               Flow.numberOfInteriorCells:get())
    grid.cells.interior:foreach(Flow.AddEnergySource)
end

function TimeIntegrator.UpdateSolution(stage)
    Flow.Update(stage)
    Particles.Update(stage)
end

function TimeIntegrator.AdvanceTimeStep()

    TimeIntegrator.SetupTimeStep()
    local timeOld = TimeIntegrator.simTime:get()
    for stage = 1, 4 do
        TimeIntegrator.InitializeTimeDerivatives()
        TimeIntegrator.ComputeDFunctionDt()
        TimeIntegrator.UpdateSolution(stage)
        TimeIntegrator.UpdateAuxiliary()
        TimeIntegrator.UpdateTime(timeOld, stage)
    end
    TimeIntegrator.ConcludeTimeStep()

    TimeIntegrator.timeStep:set(TimeIntegrator.timeStep:get() + 1)

end

function TimeIntegrator.CalculateDeltaTime()
  
  -- Check whether we are imposing a delta time or basing it on the CFL,
  -- i.e. a negative CFL was imposed in the config
  if TimeIntegrator.cfl < 0 then
    
    -- Impose a fixed time step from the config
    TimeIntegrator.deltaTime:set(TimeIntegrator.delta_time)
    
  else
  
    -- Calculate the convective, viscous, and heat spectral radii
    Flow.CalculateSpectralRadii(grid.cells)
    
    local maxV = maxViscousSpectralRadius:get()
    local maxH = maxHeatConductionSpectralRadius:get()
    local maxC = maxConvectiveSpectralRadius:get()
    
    -- Calculate diffusive spectral radius as the maximum between
    -- heat conduction and convective spectral radii
    local maxD = ( maxV > maxH ) and maxV or maxH
    
    -- Calculate global spectral radius as the maximum between the convective
    -- and diffusive spectral radii
    local spectralRadius = ( maxD > maxC ) and maxD or maxC
    
    -- Delta time using the CFL and max spectral radius for stability
    TimeIntegrator.deltaTime:set(TimeIntegrator.cfl / spectralRadius)
    
  end
  
end

-------------
-- STATISTICS
-------------

function Statistics.ResetSpatialAverages()
    Flow.numberOfInteriorCells:set(0)
    Flow.areaInterior:set(0)
    Flow.averagePressure:set(0.0)
    Flow.averageTemperature:set(0.0)
    Flow.averageKineticEnergy:set(0.0)
    Flow.minTemperature:set(math.huge)
    Flow.maxTemperature:set(-math.huge)
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
    Flow.IntegrateQuantities(grid.cells.interior)
    particles:foreach(Particles.IntegrateQuantities)
    Statistics.UpdateSpatialAverages(grid, particles)
end

-----
-- IO
-----

function IO.WriteConsoleOutput(timeStep)
  
  -- Output log headers at a specified frequency

  if timeStep % TimeIntegrator.headerFrequency == 0 then
    io.stdout:write("\n Current time step: ",
      string.format(" %2.6e",TimeIntegrator.deltaTime:get()), " s.\n")
    io.stdout:write(" Min Flow Temp: ",
      string.format("%11.6f",Flow.minTemperature:get()), " K.")
    io.stdout:write(" Max Flow Temp: ",
      string.format("%11.6f",Flow.maxTemperature:get()), " K.\n\n")
    io.stdout:write(string.format("%8s",'    Iter'),
      string.format("%12s",'   Time(s)'),
      string.format("%12s",'Avg Press'),
      string.format("%12s",'Avg Temp'),
      string.format("%12s",'Avg KE'),
      string.format("%12s",'Particle T'),'\n')
  end

  -- Check if we have particles (simply to avoid nan printed to screen)

  local particle_avg_temp = 0.0
  if particles_options.num > 0 then
    particle_avg_temp = Particles.averageTemperature:get()
  end

  -- Output the current stats to the console for this iteration
  
  io.stdout:write(string.format("%8d",timeStep),
                  string.format(" %11.6f",TimeIntegrator.simTime:get()),
                  string.format(" %11.6f",Flow.averagePressure:get()),
                  string.format(" %11.6f",Flow.averageTemperature:get()),
                  string.format(" %11.6f",Flow.averageKineticEnergy:get()),
                  string.format(" %11.6f",particle_avg_temp),'\n')

end

function IO.WriteFlowRestart(timeStep)
  
  -- Check if it is time to output a restart file
  if (timeStep % TimeIntegrator.restartEveryTimeSteps == 0 and
      IO.wrtRestart == ON) then
      
      -- Prepare the restart info file (.dat)
      
      local outputFileName = IO.outputFileNamePrefix .. "restart_" ..
      tostring(timeStep) .. ".dat"
      
      -- Open file
      
      local outputFile = io.output(outputFileName)
      
      -- We only need to write a few things to this info file (not fields)
      
      local nCells = grid.cells.velocity:Size()
      local nX = grid_options.xnum + 2*xBnum
      local nY = grid_options.ynum + 2*yBnum
      local nZ = grid_options.znum + 2*zBnum
      
      io.write('Soleil Flow Restart\n')
      local s = '' .. tostring(nX)
      s = s .. ' ' .. tostring(nY)
      s = s .. ' ' .. tostring(nZ)
      s = s .. ' ' .. tostring(timeStep)
      s = s .. ' ' .. tostring(TimeIntegrator.simTime:get()) .. '\n'
      io.write(s)
                           
     -- Close the restart file
     
     io.close()
     
     -- Write the restart CSV files for density, pressure, and velocity
     
     local fileName = IO.outputFileNamePrefix .. "restart_rho_" ..
     tostring(timeStep) .. ".csv"
     grid.cells.rho:SaveToCSV(fileName,{precision=16})
     
     fileName = IO.outputFileNamePrefix .. "restart_pressure_" ..
     tostring(timeStep) .. ".csv"
     grid.cells.pressure:SaveToCSV(fileName,{precision=16})
     
     fileName = IO.outputFileNamePrefix .. "restart_velocity_" ..
     tostring(timeStep) .. ".csv"
     grid.cells.velocity:SaveToCSV(fileName,{precision=16})
     
  end
  
end

-- Terra callback function to jointly dump multiple fields. Here, we use
-- it to write our Tecplot output (ASCII) for the flow phase.
-- Callback can dump fields to stdout/ err/ any kind of file.
-- Callback has read only access to requested fields.
local terra FlowTecplotTerra(dldarray : &dld.ctype, filename : &int8,
                             timeStep : uint8, timePhys : double,
                             nX : uint8, nY : uint8, nZ : uint8,
                             originX:double, originY:double, originZ:double,
                             dX : double, dY : double, dZ : double)

  -- Access the density, velocity, pressure, and temperature
  -- fields and set the appropriate dimensions and strides.

  var rho = dldarray[0]
  var s_r = rho.stride
  var dim = rho.dims

  var velocity = dldarray[1]
  var s_v      = velocity.stride
  var s_vec_v  = velocity.type.stride

  var pressure = dldarray[2]
  var s_p = pressure.stride

  var temperature = dldarray[3]
  var s_t = temperature.stride

  var halo_layer = dldarray[4]
  var s_h = halo_layer.stride

  -- Create one pointer that we will use repeatedly to access the
  -- data in the current sell based on our strides.

  var value : &double
  var halo  : &int
  var write : bool

  -- Get a file pointer and open up the Tecplot file for writing.

  var fp : &C.FILE
  fp = C.fopen(filename, "w")

  -- Write the Tecplot header for a cell-centered data file.

  C.fprintf(fp,"%s",'TITLE = "Soleil-X Flow Solution"\n')
  C.fprintf(fp,"%s %s",'VARIABLES = "X", "Y", "Z", "Density", "X-Velocity",',
            ' "Y-Velocity", "Z-Velocity", "Pressure", "Temperature"\n')
  C.fprintf(fp,"%s %d %s %f %s %d %s %d %s %d %s",'ZONE STRANDID=', timeStep+1,
            ' SOLUTIONTIME=', timePhys, ' I=', nX+1, ' J=', nY+1, ' K=', nZ+1,
            ' DATAPACKING=BLOCK VARLOCATION=([4-9]=CELLCENTERED)\n')

  -- First, write the x, y, and z coords for the vertices. Note that
  -- we are currently computing these on the fly to avoid issues with
  -- halo layers/periodic boundaries, etc.

  -- X Coordinates
  for k = 1, nZ+2 do
    for j = 1, nY+2 do
      for i = 1, nX+2 do
        C.fprintf(fp,"%.16f\n", originX + (dX * (i-1)))
      end
    end
  end

  -- Y Coordinates
  for k = 1, nZ+2 do
    for j = 1, nY+2 do
      for i = 1, nX+2 do
        C.fprintf(fp,"%.16f\n", originY + (dY * (j-1)))
      end
    end
  end

  -- Z Coordinates
  for k = 1, nZ+2 do
    for j = 1, nY+2 do
      for i = 1, nX+2 do
        C.fprintf(fp,"%.16f\n", originZ + (dZ * (k-1)))
      end
    end
  end

  -- Density
  for k = 0, dim[2] do
    for j = 0, dim[1] do
      for i = 0, dim[0] do
        halo  = [&int]([&uint8](halo_layer.address) + i*s_h[0] + j*s_h[1] +
                          k*s_h[2])
        value = [&double]([&uint8](rho.address) + i*s_r[0] + j*s_r[1] +
                          k*s_r[2])
        if (@halo == 0) then
          C.fprintf(fp,"%.16f\n",@value)
          write = true
        else write = false end
      end
    end
  end

  -- Velocity (i_vec = 0)
  for i_vec = 0,3 do
    for k = 0, dim[2] do
      for j = 0, dim[1] do
        for i = 0, dim[0] do
          halo  = [&int]([&uint8](halo_layer.address) + i*s_h[0] + j*s_h[1] +
                         k*s_h[2])
          value = [&double]([&uint8](velocity.address) + i*s_v[0] + j*s_v[1] +
                            k*s_v[2] + i_vec*s_vec_v[0])
          if (@halo == 0) then
            C.fprintf(fp,"%.16f\n",@value)
            write = true
          else write = false end
        end
      end
    end
  end

  -- Pressure
  for k = 0, dim[2] do
    for j = 0, dim[1] do
      for i = 0, dim[0] do
        halo  = [&int]([&uint8](halo_layer.address) + i*s_h[0] + j*s_h[1] +
                       k*s_h[2])
        value = [&double]([&uint8](pressure.address) + i*s_p[0] + j*s_p[1] +
                          k*s_p[2])
        if (@halo == 0) then
          C.fprintf(fp,"%.16f\n",@value)
          write = true
        else write = false end
      end
    end
  end

  -- Temperature
  for k = 0, dim[2] do
    for j = 0, dim[1] do
      for i = 0, dim[0] do
        halo  = [&int]([&uint8](halo_layer.address) + i*s_h[0] + j*s_h[1] +
                       k*s_h[2])
        value = [&double]([&uint8](temperature.address) + i*s_t[0] + j*s_t[1] +
                          k*s_t[2])
        if (@halo == 0) then
          C.fprintf(fp,"%.16f\n",@value)
          write = true
        else write = false end
      end
    end
  end

  -- Close the Tecplot file
  C.fclose(fp)

end

function IO.WriteFlowTecplotTerra(timeStep)
  
  -- Check if it is time to output to file
  if (timeStep % TimeIntegrator.outputEveryTimeSteps == 0 and
      IO.wrtVolumeSolution == ON) then
      
      -- Tecplot ASCII format
      local outputFileName = IO.outputFileNamePrefix .. "flow_" ..
      tostring(timeStep) .. ".dat"
      
      -- Use the terra callback to write the file (avoids Lua).
      grid.cells:DumpJointTerraFunction(FlowTecplotTerra,
                                        {'rho','velocity','pressure','temperature','cellRindLayer'},
                                        {outputFileName, timeStep, TimeIntegrator.simTime:get(),
                                        grid_options.xnum, grid_options.ynum, grid_options.znum,
                                        gridOriginInteriorX, gridOriginInteriorY, gridOriginInteriorZ,
                                        grid_options.xWidth / grid_options.xnum,
                                        grid_options.yWidth / grid_options.ynum,
                                        grid_options.zWidth / grid_options.znum})
                                        
  end
  
end

function IO.WriteFlowTecplotLua(timeStep)

-- Check if it is time to output to file
if (timeStep % TimeIntegrator.outputEveryTimeSteps == 0 and
    IO.wrtVolumeSolution == ON) then
    
  -- Tecplot ASCII format
  local outputFileName = IO.outputFileNamePrefix .. "flow_" ..
  tostring(timeStep) .. ".dat"

  -- Open file
  local outputFile = io.output(outputFileName)

  -- Write header
  io.write('TITLE = "Data"\n')
  io.write('VARIABLES = "X", "Y", "Z", "Density", "X-Velocity", "Y-Velocity",',
           '"Z-Velocity", "Pressure", "Temperature"\n')
  io.write('ZONE STRANDID=', timeStep+1, ' SOLUTIONTIME=',
           TimeIntegrator.simTime:get(), ' I=', grid_options.xnum+1, ' J=',
           grid_options.ynum+1, ' K=', grid_options.znum+1,
           ' DATAPACKING=BLOCK VARLOCATION=([4-9]=CELLCENTERED)\n')

  local s = ''
  local k = 0 -- Add a counter in order to remove space (hack for now)

  -- Here, we will recompute the coordinates just for output.
  -- This is being done as a workaround for the difference in 
  -- vertex handling between periodic and wall cases.
  local xCoord = {}          -- create the matrix
  local yCoord = {}          -- create the matrix
  local zCoord = {}          -- create the matrix
  local iVertex = 1
  for k =1,grid_options.znum+1 do
    for j=1,grid_options.ynum+1 do
      for i=1,grid_options.xnum+1 do
        xCoord[iVertex] = gridOriginInteriorX + (grid_options.xWidth /
                                                 grid_options.xnum * (i-1))
        yCoord[iVertex] = gridOriginInteriorY + (grid_options.yWidth /
                                                 grid_options.ynum  * (j-1))
        zCoord[iVertex] = gridOriginInteriorZ + (grid_options.zWidth /
                                                 grid_options.znum  * (k-1))
        iVertex = iVertex+1
      end
    end
  end
  local nVertex = iVertex-1

  -- Write the x-coordinates
  s = ''
  k = 1
  for i=1,nVertex do
    local t = tostring(xCoord[i])
    s = s .. ' ' .. t .. ''
    k = k + 1
    if k % 5 == 0 then
      s = s .. '\n'
      io.write("", s)
      s = ''
    end
  end
  io.write("", s)

  -- Write the y-coordinates
  s = ''
  k = 1
  for i=1,nVertex do
    local t = tostring(yCoord[i])
    s = s .. ' ' .. t .. ''
    k = k + 1
    if k % 5 == 0 then
      s = s .. '\n'
      io.write("", s)
      s = ''
    end
  end
  io.write("", s)

  -- Write the z-coordinates
  s = ''
  k = 1
  for i=1,nVertex do
    local t = tostring(zCoord[i])
    s = s .. ' ' .. t .. ''
    k = k + 1
    if k % 5 == 0 then
      s = s .. '\n'
      io.write("", s)
      s = ''
    end
  end
  io.write("", s)

  -- Now write density, velocity, pressure, temperature

  local function dump_with_cell_rind(field_name)
    s = ''
    k = 1
    grid.cells:DumpJoint({'cellRindLayer', field_name},
    function(ids, cell_rind, field_val)
      if cell_rind == 0 then
        s = s .. ' ' .. value_tostring(field_val) .. ''
        k = k + 1
      end
      if k % 5 == 0 then
        s = s .. '\n'
        io.write("", s)
        s = ''
      end
    end)
    io.write("", s)
  end
  local function dump_vec_component_with_cell_rind(field_name, dim_idx)
    s = ''
    k = 1
    grid.cells:DumpJoint({'cellRindLayer', field_name},
    function(ids, cell_rind, field_val)
      if cell_rind == 0 then
        s = s .. ' ' .. value_tostring(field_val[dim_idx]) .. ''
        k = k + 1
      end
      if k % 5 == 0 then
        s = s .. '\n'
        io.write("", s)
        s = ''
      end
    end)
    io.write("", s)
  end

  -- Now write density, velocity, pressure, and temperature

  dump_with_cell_rind('rho')
  local veclen = grid.cells.velocity:Type().N
  for j = 1,veclen do
    dump_vec_component_with_cell_rind('velocity', j)
  end
  dump_with_cell_rind('pressure')
  dump_with_cell_rind('temperature')

  -- close the file
  io.close()

end

end

function IO.WriteParticleRestart(timeStep)

  -- Check if it is time to output a particle restart file
  if (timeStep % TimeIntegrator.restartEveryTimeSteps == 0 and
  IO.wrtRestart == ON) then

    -- Write the restart CSV files for density, pressure, and velocity

    local fileName = IO.outputFileNamePrefix .. 'restart_particle_position_' ..
    tostring(timeStep) .. '.csv'
    particles.position:SaveToCSV(fileName,{precision=16})

    fileName = IO.outputFileNamePrefix .. 'restart_particle_velocity_' ..
    tostring(timeStep) .. '.csv'
    particles.velocity:SaveToCSV(fileName,{precision=16})

    fileName = IO.outputFileNamePrefix .. 'restart_particle_temperature_' ..
    tostring(timeStep) .. '.csv'
    particles.temperature:SaveToCSV(fileName,{precision=16})

    fileName = IO.outputFileNamePrefix .. 'restart_particle_diameter_' ..
    tostring(timeStep) .. '.csv'
    particles.diameter:SaveToCSV(fileName,{precision=16})

  end

end

-- terra callback to jointly dump multiple fields. Here, we use the
-- function to write our Tecplot output (ASCII) for the particle phase.
local terra ParticleTecplotTerra(dldarray : &dld.ctype, filename : &int8,
                                 timePhys : double)

  -- Access the density, velocity, pressure, and temperature
  -- fields and set the appropriate dimensions and strides.

  var position = dldarray[0]
  var s_p      = position.stride
  var dim      = position.dims
  var s_vec_p  = position.type.stride

  var velocity = dldarray[1]
  var s_v      = velocity.stride
  var s_vec_v  = velocity.type.stride

  var temperature = dldarray[2]
  var s_t = temperature.stride

  var diameter = dldarray[3]
  var s_d = diameter.stride

  -- Create one pointer that we will use repeatedly to access the
  -- data in the current sell based on our strides.

  var value : &double

  -- Get a file pointer and open up the Tecplot file for writing.

  var fp : &C.FILE
  fp = C.fopen(filename, "w")

  -- Write the Tecplot header for a cell-centered data file.

  C.fprintf(fp,"%s %s",'VARIABLES = "X", "Y", "Z", "X-Velocity",',
            ' "Y-Velocity", "Z-Velocity", "Temperature", "Diameter"\n')
  C.fprintf(fp,"%s %f %s",'ZONE SOLUTIONTIME=', timePhys, '\n')

  -- Write the position, velocity, temperature, and diameter for
  -- each particle on successive lines in order.

  for i = 0,dim[0] do
    for i_vec = 0,3 do
      value = [&double]([&uint8](position.address)+i*s_p[0]+i_vec*s_vec_p[0])
      C.fprintf(fp," %.16f ",@value)
    end
    for i_vec = 0,3 do
      value = [&double]([&uint8](velocity.address)+i*s_v[0]+i_vec*s_vec_v[0])
      C.fprintf(fp," %.16f ",@value)
    end
    value = [&double]([&uint8](temperature.address)+i*s_t[0])
    C.fprintf(fp," %.16f ",@value)
    value = [&double]([&uint8](diameter.address)+i*s_d[0])
    C.fprintf(fp," %.16f ",@value)
    C.fprintf(fp,"%s","\n")
  end

  -- Close the Tecplot file
  C.fclose(fp)

end

function IO.WriteParticleTecplotTerra(timeStep)
  
  -- Check if it is time to output to file
  if (timeStep % TimeIntegrator.outputEveryTimeSteps == 0 and
      IO.wrtVolumeSolution == ON) then
      
      -- Write a file for the particle positions
      local particleFileName = IO.outputFileNamePrefix .. "particles_" ..
      tostring(timeStep) .. ".dat"
      
      -- Use the terra callback to write the file (avoids Lua).
      particles:DumpJointTerraFunction(ParticleTecplotTerra,
                                        {'position','velocity','temperature','diameter'},
                                        {particleFileName, TimeIntegrator.simTime:get()})
                                        
  end
  
end

function IO.WriteParticleTecplotLua(timeStep)
  
  -- Check if it is time to output to file
  if (timeStep % TimeIntegrator.outputEveryTimeSteps == 0 and
      IO.wrtVolumeSolution == ON) then
      
  -- Write a file for the particle positions
  -- Tecplot ASCII format
  local particleFileName = IO.outputFileNamePrefix .. "particles_" ..
  tostring(timeStep) .. ".dat"

  -- Open file
  local particleFile = io.output(particleFileName)

  -- Compute the number of vertices to be written
  -- Write header
  io.write('VARIABLES = "X", "Y", "Z", "X-Velocity", "Y-Velocity", ',
           '"Z-Velocity", "Temperature", "Diameter"\n')
  io.write('ZONE SOLUTIONTIME=', TimeIntegrator.simTime:get(), '\n')

  local veclen = particles.position:Type().N
  particles:DumpJoint({'position', 'velocity', 'temperature', 'diameter'},
  function(ids, pos, vel, temp, diam)
    local s = ''
    s = s .. ' ' .. value_tostring(pos) .. ''
    s = s .. ' ' .. value_tostring(vel) .. ''
    s = s .. ' ' .. value_tostring(temp) ..
             ' ' .. value_tostring(diam) .. '\n'
    io.write("", s)
  end)

  -- Close the file.
  io.close()

  end
  
end

function IO.WriteParticleEvolution(timeStep)
  
  if (timeStep % TimeIntegrator.outputEveryTimeSteps == 0 and
      IO.wrtParticleEvolution == ON) then
    
    -- Prepare the particle evolution file name
    
    local particleEvolutionIndex  = IO.particleEvolutionIndex
    local outputFileName = IO.outputFileNamePrefix .. "evolution_particle_" ..
    tostring(particleEvolutionIndex) .. ".csv"
    
    -- Check if file already exists
    
    local fileDidNotExist = io.open(outputFileName,"r") == nil
    
    -- Open file
    
    local outputFile = io.open(outputFileName,"a")
    io.output(outputFile)
    
    -- CSV header
    
    if fileDidNotExist then
      io.write('"Time", "X", "Y", "Z", "X-Velocity", "Y-Velocity", "Z-Velocity", "Temperature", "Diameter"\n')
    end
    
    -- Check for the particle with 'index=particleIndex' and write its primitive variables
    
    local pos  = particles.position:DumpToList()
    local vel  = particles.velocity:DumpToList()
    local temp = particles.temperature:DumpToList()
    local diam = particles.diameter:DumpToList()
    
    local s =        value_tostring(TimeIntegrator.simTime:get())  .. ''
    s = s .. ', ' .. value_tostring_comma(pos[particleEvolutionIndex])  .. ''
    s = s .. ', ' .. value_tostring_comma(vel[particleEvolutionIndex])  .. ''
    s = s .. ', ' .. value_tostring(temp[particleEvolutionIndex]) .. ''
    s = s .. ', ' .. value_tostring(diam[particleEvolutionIndex]) .. '\n'
    
    io.write("", s)
    
    -- Close the file
    
    io.close()
    
    --end
  end
  
end

function IO.WriteX0SliceVec (timeStep, field, filename)
  
  if (timeStep % TimeIntegrator.outputEveryTimeSteps == 0 and
      IO.wrt1DSlice == ON) then

      -- Open file
      local outputFile = io.output(IO.outputFileNamePrefix .. filename)
      
      -- CSV header
      io.write('y, ' .. field .. '_1, ' .. field .. '_2, ' .. field .. '_3\n')
      
      -- Check for the vertical center of the domain and write the x-vel
      grid.cells:DumpJoint({ 'centerCoordinates', field },
                           function(ids, cellCenter, field)
                           local s = ''
                           local x = cellCenter[1]
                           local y = cellCenter[2]
                           local z = cellCenter[3]
                           if    x < (gridOriginInteriorX
                                      --               + grid_options.xWidth/2.0
                                      -- Modification in order to avoid not finding cells when grid_options.xnum is a pair number. A tolerance of "x-gridSize/1000.0" is added.
                                      + grid_options.xWidth/2.0 + (grid_options.xWidth / grid_options.xnum) / 1000.0
                                      + grid_options.xWidth / (2.0*grid_options.xnum))
                           and x > (gridOriginInteriorX
                                    + grid_options.xWidth/2.0
                                    - grid_options.xWidth / (2.0*grid_options.xnum))
                           and y < (gridOriginInteriorY + grid_options.yWidth)
                           and y > (gridOriginInteriorY)
                           and z < (gridOriginInteriorZ + grid_options.zWidth)
                           and z > (gridOriginInteriorZ)
                           then
                           s = tostring(y) .. ', ' .. tostring(field[1]) .. ', '
                                                   .. tostring(field[2]) .. ', '
                                                   .. tostring(field[3]) .. '\n'
                           io.write(s)
                           end
                           end)
                           
                           -- Close the file
                           io.close()
                           
  end
  
end

function IO.WriteY0SliceVec (timeStep, field, filename)
  
  if (timeStep % TimeIntegrator.outputEveryTimeSteps == 0 and
      IO.wrt1DSlice == ON) then
      
      -- Open file
      local outputFile = io.output(IO.outputFileNamePrefix .. filename)
      
      -- CSV header
      io.write('x, ' .. field .. '\n')
      
      -- Check for the vertical center of the domain and write the x-vel
      grid.cells:DumpJoint({ 'centerCoordinates', field },
                           function(ids, cellCenter, field)
                           local s = ''
                           local x = cellCenter[1]
                           local y = cellCenter[2]
                           local z = cellCenter[3]
                           if    y < (gridOriginInteriorY
                                      --               + grid_options.yWidth/2.0
                                      -- Modification in order to avoid not finding cells when grid_options.ynum is a pair number. A tolerance of "y-gridSize/1000.0" is added.
                                      + grid_options.yWidth/2.0 + (grid_options.yWidth / grid_options.ynum) / 1000.0
                                      + grid_options.yWidth / (2.0*grid_options.ynum))
                           and y > (gridOriginInteriorY
                                    + grid_options.yWidth/2.0
                                    - grid_options.yWidth / (2.0*grid_options.ynum))
                           and x < (gridOriginInteriorX + grid_options.xWidth)
                           and x > (gridOriginInteriorX)
                           and z < (gridOriginInteriorZ + grid_options.zWidth)
                           and z > (gridOriginInteriorZ)
                           then
                           s = tostring(y) .. ', ' .. tostring(field[1]) .. ', '
                                                   .. tostring(field[2]) .. ', '
                                                   .. tostring(field[3]) .. '\n'
                           io.write(s)
                           end
                           end)
                           
                           -- Close the file
                           io.close()
                           
  end
  
end

function IO.WriteX0Slice (timeStep, field, filename)
  
  if (timeStep % TimeIntegrator.outputEveryTimeSteps == 0 and
      IO.wrt1DSlice == ON) then

  -- Open file
  local outputFile = io.output(IO.outputFileNamePrefix .. filename)
  
  -- CSV header
  io.write('y, ' .. field .. '\n')
  
  -- Check for the vertical center of the domain and write the x-vel
  grid.cells:DumpJoint({ 'centerCoordinates', field },
  function(ids, cellCenter, field)
    local s = ''
    local x = cellCenter[1]
    local y = cellCenter[2]
    local z = cellCenter[3]
    if    x < (gridOriginInteriorX
              --               + grid_options.xWidth/2.0
              -- Modification in order to avoid not finding cells when grid_options.xnum is a pair number. A tolerance of "x-gridSize/1000.0" is added.
              + grid_options.xWidth/2.0 + (grid_options.xWidth / grid_options.xnum) / 1000.0
              + grid_options.xWidth / (2.0*grid_options.xnum))
    and x > (gridOriginInteriorX
            + grid_options.xWidth/2.0
            - grid_options.xWidth / (2.0*grid_options.xnum))
      and y < (gridOriginInteriorY + grid_options.yWidth)
      and y > (gridOriginInteriorY)
      and z < (gridOriginInteriorZ + grid_options.zWidth)
      and z > (gridOriginInteriorZ)
    then
      s = tostring(y) .. ', ' .. tostring(field) .. '\n'
      io.write(s)
    end
  end)
  
  -- Close the file
  io.close()
  
  end
  
end

function IO.WriteY0Slice (timeStep, field, filename)
  
  
  if (timeStep % TimeIntegrator.outputEveryTimeSteps == 0 and
      IO.wrt1DSlice == ON) then
      
      -- Open file
      local outputFile = io.output(IO.outputFileNamePrefix .. filename)
      
      -- CSV header
      io.write('x, ' .. field .. '\n')

  -- Check for the vertical center of the domain and write the x-vel
  grid.cells:DumpJoint({ 'centerCoordinates', field },
  function(ids, cellCenter, field)
    local s = ''
    local x = cellCenter[1]
    local y = cellCenter[2]
    local z = cellCenter[3]
     if    y < (gridOriginInteriorY
                --               + grid_options.yWidth/2.0
                -- Modification in order to avoid not finding cells when grid_options.ynum is a pair number. A tolerance of "y-gridSize/1000.0" is added.
                + grid_options.yWidth/2.0 + (grid_options.yWidth / grid_options.ynum) / 1000.0
                + grid_options.yWidth / (2.0*grid_options.ynum))
      and y > (gridOriginInteriorY
               + grid_options.yWidth/2.0
               - grid_options.yWidth / (2.0*grid_options.ynum))
      and x < (gridOriginInteriorX + grid_options.xWidth)
      and x > (gridOriginInteriorX)
      and z < (gridOriginInteriorZ + grid_options.zWidth)
      and z > (gridOriginInteriorZ)
    then
      s = tostring(x) .. ', ' .. tostring(field) .. '\n'
      io.write(s)
    end
  end)
  
  -- Close the file
  io.close()
  
  end
                       
end

function IO.WriteOutput(timeStep)

  -- Write the console output to the screen

  IO.WriteConsoleOutput(timeStep)

  -- Write the flow restart files

  IO.WriteFlowRestart(timeStep)
  
  -- Write the particle restart files
  
  IO.WriteParticleRestart(timeStep)

  -- Write the volume solution files for visualization

  if IO.outputFormat == IO.Tecplot then
    IO.WriteFlowTecplotTerra(timeStep)
    --IO.WriteFlowTecplotLua(timeStep)
    IO.WriteParticleTecplotTerra(timeStep)
    --IO.WriteParticleTecplotLua(timeStep)
  else
    print("Output format not defined. No output written to disk.")
  end

  -- Write center line profiles to CSV files

  IO.WriteX0SliceVec (timeStep, 'velocity',    'velocity_x0.csv')
  IO.WriteY0SliceVec (timeStep, 'velocity',    'velocity_y0.csv')
  IO.WriteX0Slice (timeStep, 'temperature', 'temperature_x0.csv')
  IO.WriteY0Slice (timeStep, 'temperature', 'temperature_y0.csv')

  -- Write a file for the evolution in time of particle i
  
  IO.WriteParticleEvolution(timeStep)

end

----------------
-- VISUALIZATION
----------------

function Visualization.Draw()
    vdb.vbegin()
    vdb.frame()
    grid.cells:foreach(Flow.DrawFunction)
    particles:foreach(Particles.DrawFunction)
    vdb.vend()
end

-----------------------------------------------------------------------------
--[[                            MAIN EXECUTION                           ]]--
-----------------------------------------------------------------------------

-- Initialize all variables

TimeIntegrator.InitializeVariables()
Statistics.ComputeSpatialAverages()
IO.WriteOutput(TimeIntegrator.timeStep:get())

-- Main iteration loop

while ((TimeIntegrator.simTime:get()  < TimeIntegrator.final_time) and
       (TimeIntegrator.timeStep:get() < TimeIntegrator.max_iter))  do

    TimeIntegrator.CalculateDeltaTime()
    TimeIntegrator.AdvanceTimeStep()
    Statistics.ComputeSpatialAverages()
    IO.WriteOutput(TimeIntegrator.timeStep:get())
    
    -- Visualize the simulation with VDB if requested
    if vdb_options.visualize == ON then
      Visualization.Draw()
    end
end

print("")
print("--------------------------- Exit Success ----------------------------")
print("")
