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
local Visualization = {};

-----------------------------------------------------------------------------
--[[         Global variables used for specialization within kernels     ]]--
-----------------------------------------------------------------------------

-- Flow type
Flow.Uniform             = L.NewGlobal(L.int, 0)
Flow.TaylorGreen2DVortex = L.NewGlobal(L.int, 1)
Flow.TaylorGreen3DVortex = L.NewGlobal(L.int, 2)

-- Particles feeder
Particles.FeederAtStartTimeInRandomBox = L.NewGlobal(L.int, 0)
Particles.FeederOverTimeInRandomBox    = L.NewGlobal(L.int, 1)
Particles.FeederUQCase                 = L.NewGlobal(L.int, 2)

-- Particles collector
Particles.CollectorNone     = L.NewGlobal(L.int, 0)
Particles.CollectorOutOfBox = L.NewGlobal(L.int, 1)

-----------------------------------------------------------------------------
--[[                       COLORS FOR VISUALIZATION                      ]]--
-----------------------------------------------------------------------------
local unity = L.NewVector(L.float,{1.0,1.0,1.0})
local red   = L.NewVector(L.float,{1.0,0.0,0.0})
local green = L.NewVector(L.float,{0.0,1.0,0.0})
local blue  = L.NewVector(L.float,{0.0,0.0,1.0})
local white = L.NewVector(L.float,{1.0,1.0,1.0})

-----------------------------------------------------------------------------
--[[                             OPTIONS                                 ]]--
-----------------------------------------------------------------------------

local grid_options = {
xnum = 64,
ynum = 64,
znum = 64,
origin = {0.0, 0.0, 0.0},
xWidth = twoPi,
--xWidth = 2*twoPi,
yWidth = twoPi,
zWidth = twoPi,
--xWidth = 1.0,
--yWidth = 1.0,
--zWidth = 1.0/64.0,
--xBCLeft  = 'wall',
xBCLeftVel = {0.0, 0.0, 0.0},
--xBCRight = 'wall',
xBCRightVel = {0.0, 0.0, 0.0},
--yBCLeft  = 'wall',
yBCLeftVel = {0.0, 0.0, 0.0},
--yBCRight = 'wall',
yBCRightVel = {33.179, 0.0, 0.0},
--zBCLeft  = 'symmetry',
zBCLeftVel = {0.0, 0.0, 0.0},
--zBCRight = 'symmetry',
zBCRightVel = {0.0, 0.0, 0.0},
--xBCLeft  = 'periodic',
--xBCRight = 'periodic',
--yBCLeft  = 'periodic',
--yBCRight = 'periodic',
--zBCLeft  = 'periodic',
--zBCRight = 'periodic',
xBCLeft  = 'symmetry',
xBCRight = 'symmetry',
yBCLeft  = 'symmetry',
yBCRight = 'symmetry',
zBCLeft  = 'symmetry',
zBCRight = 'symmetry',
}

local spatial_stencil = {
--  Splitting parameter
    split = 0.5,
--  Order 2
--    order = 2,
--    size = 2,
--    numInterpolateCoeffs = 2,
--    interpolateCoeffs = L.NewVector(L.double, {0, 0.5}),
--    numFirstDerivativeCoeffs = 2,
--    firstDerivativeCoeffs = L.NewVector(L.double, {0, 0.5}),
--    firstDerivativeModifiedWaveNumber = 1.0,
--    secondDerivativeModifiedWaveNumber = 4.0,
--  Order 6
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
-- The sign variables define the necessary reflections for the
-- different types of BCs. The wall velocity is specified above,
-- and then the velocity adjustment is calculated here and applied
-- to the boundaries below. WARNING: we are assuming uniform meshes
-- here as well, as the 2.0 results in a simple average to match
-- the specified wall velocity when the BC is applied.

-- Define offsets, signs, and velocities for the x BCs

local xSignX, xSignY, xSignZ
local xBCLeftVelX, xBCLeftVelY, xBCLeftVelZ
local xBCRightVelX, xBCRightVelY, xBCRightVelZ
-- Offset liszt functions
local XOffsetPeriodic = liszt function(boundaryPointDepth)
  return 0
end
local XOffsetDummyPeriodic = liszt function(boundaryPointDepth)
  return grid_options.xnum
end
local XOffsetSymmetry = liszt function(boundaryPointDepth)
  return 2*boundaryPointDepth-1
end
if grid_options.xBCLeft  == "periodic" and 
   grid_options.xBCRight == "periodic" then
  XOffset = XOffsetPeriodic
  xSignX = 1
  xSignY = 1
  xSignZ = 1
  xBCLeftVelX  = 0.0
  xBCLeftVelY  = 0.0
  xBCLeftVelZ  = 0.0
  xBCRightVelX = 0.0
  xBCRightVelY = 0.0
  xBCRightVelZ = 0.0
elseif grid_options.xBCLeft == "symmetry" and
       grid_options.xBCRight == "symmetry" then
  XOffset = XOffsetSymmetry
  xSignX = -1
  xSignY = 1
  xSignZ = 1
  xBCLeftVelX  = 0.0
  xBCLeftVelY  = 0.0
  xBCLeftVelZ  = 0.0
  xBCRightVelX = 0.0
  xBCRightVelY = 0.0
  xBCRightVelZ = 0.0
elseif grid_options.xBCLeft  == "wall" and
       grid_options.xBCRight == "wall" then
  XOffset = XOffsetSymmetry
  xSignX = -1
  xSignY = -1
  xSignZ = -1
  xBCLeftVelX  = 2.0*grid_options.xBCLeftVel[1]
  xBCLeftVelY  = 2.0*grid_options.xBCLeftVel[2]
  xBCLeftVelZ  = 2.0*grid_options.xBCLeftVel[3]
  xBCRightVelX = 2.0*grid_options.xBCRightVel[1]
  xBCRightVelY = 2.0*grid_options.xBCRightVel[2]
  xBCRightVelZ = 2.0*grid_options.xBCRightVel[3]
else
  error("Boundary conditions in x not implemented")
end

-- Define offsets, signs, and velocities for the y BCs

local ySignX, ySignY, ySignZ
local yBCLeftVelX, yBCLeftVelY, yBCLeftVelZ
local yBCRightVelX, yBCRightVelY, yBCRightVelZ
local YOffsetDummyPeriodic = liszt function(boundaryPointDepth)
  return 0
end
local YOffsetSymmetry = liszt function(boundaryPointDepth)
  return 2*boundaryPointDepth-1
end
if grid_options.yBCLeft  == "periodic" and 
   grid_options.yBCRight == "periodic" then
  YOffset = YOffsetDummyPeriodic
  ySignX = 1
  ySignY = 1
  ySignZ = 1
  yBCLeftVelX = 0.0
  yBCLeftVelY = 0.0
  yBCLeftVelZ = 0.0
  yBCRightVelX = 0.0
  yBCRightVelY = 0.0
  yBCRightVelZ = 0.0
elseif grid_options.yBCLeft  == "symmetry" and
       grid_options.yBCRight == "symmetry" then
  YOffset = YOffsetSymmetry
  ySignX = 1
  ySignY = -1
  ySignZ = 1
  yBCLeftVelX = 0.0
  yBCLeftVelY = 0.0
  yBCLeftVelZ = 0.0
  yBCRightVelX = 0.0
  yBCRightVelY = 0.0
  yBCRightVelZ = 0.0
elseif grid_options.yBCLeft  == "wall" and
       grid_options.yBCRight == "wall" then
  YOffset = YOffsetSymmetry
  ySignX = -1
  ySignY = -1
  ySignZ = -1
  yBCLeftVelX  = 2.0*grid_options.yBCLeftVel[1]
  yBCLeftVelY  = 2.0*grid_options.yBCLeftVel[2]
  yBCLeftVelZ  = 2.0*grid_options.yBCLeftVel[3]
  yBCRightVelX = 2.0*grid_options.yBCRightVel[1]
  yBCRightVelY = 2.0*grid_options.yBCRightVel[2]
  yBCRightVelZ = 2.0*grid_options.yBCRightVel[3]
else
  error("Boundary conditions in y not implemented")
end

-- Define offsets, signs, and velocities for the z BCs

local zSignX, zSignY, zSignZ
local zBCLeftVelX, zBCLeftVelY, zBCLeftVelZ
local zBCRightVelX, zBCRightVelY, zBCRightVelZ
-- Offset liszt functions
local ZOffsetPeriodic = liszt function(boundaryPointDepth)
  return 0
end
local ZOffsetDummyPeriodic = liszt function(boundaryPointDepth)
  return 0
end
local ZOffsetSymmetry = liszt function(boundaryPointDepth)
  return 2*boundaryPointDepth-1
end
if grid_options.zBCLeft  == "periodic" and 
   grid_options.zBCRight == "periodic" then
  ZOffset = ZOffsetPeriodic
  zSignX = 1
  zSignY = 1
  zSignZ = 1
  zBCLeftVelX = 0.0
  zBCLeftVelY = 0.0
  zBCLeftVelZ = 0.0
  zBCRightVelX = 0.0
  zBCRightVelY = 0.0
  zBCRightVelZ = 0.0
elseif grid_options.zBCLeft == "symmetry" and
       grid_options.zBCRight == "symmetry" then
  ZOffset = ZOffsetSymmetry
  zSignX = 1
  zSignY = 1
  zSignZ = -1
  zBCLeftVelX = 0.0
  zBCLeftVelY = 0.0
  zBCLeftVelZ = 0.0
  zBCRightVelX = 0.0
  zBCRightVelY = 0.0
  zBCRightVelZ = 0.0
elseif grid_options.zBCLeft  == "wall" and
       grid_options.zBCRight == "wall" then
  ZOffset = ZOffsetSymmetry
  zSignX = -1
  zSignY = -1
  zSignZ = -1
  zBCLeftVelX  = 2.0*grid_options.zBCLeftVel[1]
  zBCLeftVelY  = 2.0*grid_options.zBCLeftVel[2]
  zBCLeftVelZ  = 2.0*grid_options.zBCLeftVel[3]
  zBCRightVelX = 2.0*grid_options.zBCRightVel[1]
  zBCRightVelY = 2.0*grid_options.zBCRightVel[2]
  zBCRightVelZ = 2.0*grid_options.zBCRightVel[3]
else
  error("Boundary conditions in z not implemented")
end

-- Time integrator
TimeIntegrator.coeff_function       = {1/6, 1/3, 1/3, 1/6}
TimeIntegrator.coeff_time           = {0.5, 0.5, 1, 1}
TimeIntegrator.simTime              = L.NewGlobal(L.double,0)
TimeIntegrator.final_time           = 20.00001
TimeIntegrator.max_iter             = 500000
TimeIntegrator.timeStep             = L.NewGlobal(L.int,0)
TimeIntegrator.cfl                  = 2.0
TimeIntegrator.outputEveryTimeSteps = 100
TimeIntegrator.headerFrequency      = 20
TimeIntegrator.deltaTime            = L.NewGlobal(L.double, 0.01)

local fluid_options = {
    gasConstant = 200.4128,
    gamma = 1.4,
    dynamic_viscosity_ref = 0.001,
    dynamic_viscosity_temp_ref = 1.0,
    prandtl = 0.7
    --gasConstant = 287.058,
    --gamma = 1.4,
    --dynamic_viscosity_ref = 1.7893e-05,
    --dynamic_viscosity_temp_ref = 288.15,
    --prandtl = 0.72
}

local flow_options = {
    --initCase = Flow.TaylorGreen2DVortex,
    --initParams = L.NewGlobal(L.vector(L.double,3),
    --                           {1,100,2}),
    initCase = Flow.TaylorGreen3DVortex,
    initParams = L.NewGlobal(L.vector(L.double,3),
                               {1,100,2}),
    --initCase = Flow.Uniform,
    --initParams = L.NewGlobal(L.vector(L.double,5),
    --                           {0.000525805,43.4923,0.0,0.0,0.0}),
    --bodyForce = L.NewGlobal(L.vec3d, {0,0.01,0.0})
    bodyForce = L.NewGlobal(L.vec3d, {0,0.0,0})
}

local particles_options = {
    -- Feeder is defined by a type and a set of parameters
    -- feedeerParams is a vector of double values whose meaning
    -- differs for each feederType. Please refer to the 
    -- Particles.Feed kernel where it is specialized
    --
    -- Feed all particles at start randomly
    -- distributed on a box defined by its center and sides
    --feederType = Particles.FeederAtStartTimeInRandomBox,
    --feederParams = L.NewGlobal(L.vector(L.double,6),
    --                           {pi,pi,pi,2*pi,2*pi,2*pi}), 
    
    -- Feeding a given number of particles every timestep randomly
    -- distributed on a box defined by its center and sides
    --feederType = Particles.FeederOverTimeInRandomBox,
    --feederParams = L.NewGlobal(L.vector(L.double,10),
    --                           {pi,pi,pi, -- centerCoor
    --                            2*pi,2*pi,2*pi, -- widthCoor
    --                            2.0,2.0,0, -- velocity
    --                            0.5, -- particlesPerTimestep
    --                           }), 
    --
    ---- UQCase
    feederType = Particles.FeederUQCase,
    feederParams = L.NewGlobal(L.vector(L.double,20),
                       {pi/4,  pi/2,pi,0.1*pi,0.1*pi,pi/2,4, 4,0,0.3,
                        pi/4,3*pi/2,pi,0.1*pi,0.1*pi,pi/2,4,-4,0,0.8}),
    
    -- Collector is defined by a type and a set of parameters
    -- collectorParams is a vector of double values whose meaning
    -- differs for each collectorType. Please refer to the 
    -- Particles.Collect kernel where it is specialized
    --
    -- Do not collect particles (freely move within the domain)
    --collectorType = Particles.CollectorNone,
    --collectorParams = L.NewGlobal(L.vector(L.double,1),{0}),
    
    -- Collect all particles that exit a box defined by its Cartesian 
    -- min/max coordinates
    collectorType = Particles.CollectorOutOfBox,
    collectorParams = L.NewGlobal(L.vector(L.double,6),{0.5,0.5,0.5,12,6,6}),

    num = 0,
    convective_coefficient = L.NewGlobal(L.double, 0.7), -- W m^-2 K^-1
    heat_capacity = L.NewGlobal(L.double, 0.7), -- J Kg^-1 K^-1
    initialTemperature = 20,
    density = 1000,
    diameter_mean = 0.03,
    diameter_maxDeviation = 0.02,
    bodyForce = L.NewGlobal(L.vec3d, {0,-0.0,0}),
    --bodyForce = L.NewGlobal(L.vec3d, {0,-1.1,0}),
    emissivity = 0.5,
    absorptivity = 0.5 -- Equal to emissivity in thermal equilibrium
                       -- (Kirchhoff law of thermal radiation)
}

local radiation_options = {
    radiationIntensity = 0.0
    --10.0
}

-- IO
-- Choose an output format (0 is the native Python, 1, is for Tecplot)
--IO.outputFormat = 0 -- Python
IO.outputFormat = 1 -- Tecplot
IO.outputFileNamePrefix = "../soleilOutput/output"

-----------------------------------------------------------------------------
--[[                       GRID/PARTICLES RELATIONS                      ]]--
-----------------------------------------------------------------------------

-- Check boundary type consistency
if ( grid_options.xBCLeft  == 'periodic' and 
     grid_options.xBCRight ~= 'periodic' ) or 
   ( grid_options.xBCLeft  ~= 'periodic' and 
     grid_options.xBCRight == 'periodic' ) then
    error("Boundary conditions in x should match periodicity")
end
if ( grid_options.yBCLeft  == 'periodic' and 
     grid_options.yBCRight ~= 'periodic' ) or 
   ( grid_options.yBCLeft  ~= 'periodic' and 
     grid_options.yBCRight == 'periodic' ) then
    error("Boundary conditions in y should match periodicity")
end
if ( grid_options.zBCLeft  == 'periodic' and 
     grid_options.zBCRight ~= 'periodic' ) or 
   ( grid_options.zBCLeft  ~= 'periodic' and 
     grid_options.zBCRight == 'periodic' ) then
    error("Boundary conditions in z should match periodicity")
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

local bnum = spatial_stencil.order/2
if xBCPeriodic then
  xBnum = 0
else
  xBnum = bnum
end
if yBCPeriodic then
  yBnum = 0
else
  yBnum = bnum
end
if zBCPeriodic then
  zBnum = 0
else
  zBnum = bnum
end
local xBw   = grid_options.xWidth/grid_options.xnum * xBnum
local yBw   = grid_options.yWidth/grid_options.ynum * yBnum
local zBw   = grid_options.zWidth/grid_options.znum * zBnum
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
local grid_originX = L.NewGlobal(L.double, grid:xOrigin())
local grid_originY = L.NewGlobal(L.double, grid:yOrigin())
local grid_originZ = L.NewGlobal(L.double, grid:zOrigin())
local grid_dx = L.NewGlobal(L.double, grid:xCellWidth())
local grid_dy = L.NewGlobal(L.double, grid:yCellWidth())
local grid_dz = L.NewGlobal(L.double, grid:zCellWidth())

-- Create a field for the center coords of the dual cells (i.e., vertices)
grid.dual_cells:NewField('centerCoordinates', L.vec3d):
LoadConstant(L.NewVector(L.double, {0, 0, 0}))

-- Conserved variables
grid.cells:NewField('rho', L.double):
LoadConstant(0)
grid.cells:NewField('rhoVelocity', L.vec3d):
LoadConstant(L.NewVector(L.double, {0, 0, 0}))
grid.cells:NewField('rhoEnergy', L.double):
LoadConstant(0)

-- Primitive variables
grid.cells:NewField('centerCoordinates', L.vec3d):
LoadConstant(L.NewVector(L.double, {0, 0, 0}))
grid.cells:NewField('velocity', L.vec3d):
LoadConstant(L.NewVector(L.double, {0, 0, 0}))
grid.cells:NewField('velocityGradientX', L.vec3d):
LoadConstant(L.NewVector(L.double, {0, 0, 0}))
grid.cells:NewField('velocityGradientY', L.vec3d):
LoadConstant(L.NewVector(L.double, {0, 0, 0}))
grid.cells:NewField('velocityGradientZ', L.vec3d):
LoadConstant(L.NewVector(L.double, {0, 0, 0}))
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
grid.cells:NewField('rhoVelocityBoundary', L.vec3d):
LoadConstant(L.NewVector(L.double, {0, 0, 0}))
grid.cells:NewField('rhoEnergyBoundary', L.double):
LoadConstant(0)
grid.cells:NewField('velocityBoundary', L.vec3d):
LoadConstant(L.NewVector(L.double, {0, 0, 0}))
grid.cells:NewField('pressureBoundary', L.double):
LoadConstant(0)
grid.cells:NewField('temperatureBoundary', L.double):
LoadConstant(0)
grid.cells:NewField('velocityGradientXBoundary', L.vec3d):
LoadConstant(L.NewVector(L.double, {0, 0, 0}))
grid.cells:NewField('velocityGradientYBoundary', L.vec3d):
LoadConstant(L.NewVector(L.double, {0, 0, 0}))
grid.cells:NewField('velocityGradientZBoundary', L.vec3d):
LoadConstant(L.NewVector(L.double, {0, 0, 0}))

-- scratch (temporary) fields
-- intermediate value and copies
grid.cells:NewField('rho_old', L.double):
LoadConstant(0)
grid.cells:NewField('rhoVelocity_old', L.vec3d):
LoadConstant(L.NewVector(L.double, {0, 0, 0}))
grid.cells:NewField('rhoEnergy_old', L.double):
LoadConstant(0)
grid.cells:NewField('rho_new', L.double):
LoadConstant(0)
grid.cells:NewField('rhoVelocity_new', L.vec3d):
LoadConstant(L.NewVector(L.double, {0, 0, 0}))
grid.cells:NewField('rhoEnergy_new', L.double):
LoadConstant(0)
-- time derivatives
grid.cells:NewField('rho_t', L.double):
LoadConstant(0)
grid.cells:NewField('rhoVelocity_t', L.vec3d):
LoadConstant(L.NewVector(L.double, {0, 0, 0}))
grid.cells:NewField('rhoEnergy_t', L.double):
LoadConstant(0)
-- fluxes
grid.cells:NewField('rhoFlux', L.double):
LoadConstant(0)
grid.cells:NewField('rhoVelocityFlux', L.vec3d):
LoadConstant(L.NewVector(L.double, {0, 0, 0}))
grid.cells:NewField('rhoEnergyFlux', L.double):
LoadConstant(0)


-- Declare and initialize particle relation and fields over the particle

local particles = L.NewRelation(particles_options.num, 'particles')

particles:NewField('dual_cell', grid.dual_cells):
LoadConstant(0)
particles:NewField('cell', grid.cells):
LoadConstant(0)
particles:NewField('position', L.vec3d):
LoadConstant(L.NewVector(L.double, {0, 0, 0}))
particles:NewField('velocity', L.vec3d):
LoadConstant(L.NewVector(L.double, {0, 0, 0}))
particles:NewField('temperature', L.double):
LoadConstant(particles_options.initialTemperature)
particles:NewField('position_ghost', L.vec3d):
LoadConstant(L.NewVector(L.double, {0, 0, 0}))

particles:NewField('diameter', L.double):
-- Initialize to random distribution with given mean value and maximum 
-- deviation from it
Load(function(i)
    return cmath.rand_unity() * particles_options.diameter_maxDeviation +
           particles_options.diameter_mean
end)
particles:NewField('density', L.double):
LoadConstant(particles_options.density)

particles:NewField('deltaVelocityOverRelaxationTime', L.vec3d):
LoadConstant(L.NewVector(L.double, {0, 0, 0}))
particles:NewField('deltaTemperatureTerm', L.double):
LoadConstant(0)
-- state of a particle:
--   - a particle not yet fed has a state = 0
--   - a particle fed into the domain and active has a state = 1
--   - a particle already collected has a state =2
particles:NewField('state', L.int):
LoadConstant(0)
-- ID field
particles:NewField('id', L.int):
-- Initialize to random distribution with given mean value and maximum 
-- deviation from it
Load(function(i)
    return i
end)
-- grouID: differentiates particles within a given distribution
-- For example, when multiple injectors are used
particles:NewField('groupID', L.int):
LoadConstant(0)

-- scratch (temporary) fields
-- intermediate values and copies
particles:NewField('position_old', L.vec3d):
LoadConstant(L.NewVector(L.double, {0, 0, 0}))
particles:NewField('velocity_old', L.vec3d):
LoadConstant(L.NewVector(L.double, {0, 0, 0}))
particles:NewField('temperature_old', L.double):
LoadConstant(0)
particles:NewField('position_new', L.vec3d):
LoadConstant(L.NewVector(L.double, {0, 0, 0}))
particles:NewField('velocity_new', L.vec3d):
LoadConstant(L.NewVector(L.double, {0, 0, 0}))
particles:NewField('temperature_new', L.double):
LoadConstant(0)
-- derivatives
particles:NewField('position_t', L.vec3d):
LoadConstant(L.NewVector(L.double, {0, 0, 0}))
particles:NewField('velocity_t', L.vec3d):
LoadConstant(L.NewVector(L.double, {0, 0, 0}))
particles:NewField('temperature_t', L.double):
LoadConstant(0)


-- Statistics quantities

-- Note: - numberOfInteriorCells and areaInterior could be defined as variables
-- from grid instead of Flow. Here Flow is used to avoid adding things to grid
-- externally
Flow.numberOfInteriorCells = L.NewGlobal(L.int, 0)
Flow.areaInterior = L.NewGlobal(L.double, 0)
Flow.averagePressure = L.NewGlobal(L.double, 0.0)
Flow.averageTemperature = L.NewGlobal(L.double, 0.0)
Flow.averageKineticEnergy = L.NewGlobal(L.double, 0.0)
Flow.minTemperature = L.NewGlobal(L.double, 0)
Flow.maxTemperature = L.NewGlobal(L.double, 0)
Particles.averageTemperature= L.NewGlobal(L.double, 0.0)

-----------------------------------------------------------------------------
--[[                 CONSOLE OUTPUT AFTER PREPROCESSING                  ]]--
-----------------------------------------------------------------------------

print("\n")
print("---------------------------------------------------------------------")
print("|    _____    ___     _      ____   _____   _                       |")
print("|   / ____|  / __ \\  | |    | ___| |_   _| | |                      |")
print("|  | (___   | |  | | | |    | |_     | |   | |  Stanford University |")
print("|   \\___ \\  | |  | | | |    |  _|    | |   | |        PSAAP 2       |")
print("|   ____) | | |__| | | |__  | |__   _| |_  | |__                    |")
print("|  |_____/   \\____/  |____| |____| |_____| |____|                   |")
print("|                                                                   |")
print("---------------------------------------------------------------------")
print("| Copyright (C) 2013-2014 ...                                       |")
print("| Soleil is a turbulence/particle/radiation solver written in       |")
print("| the Liszt DSL and executed by the Legion runtime.                 |")
--print("| Soleil is distributed in the hope that it will be useful,       |")
--print("| but WITHOUT ANY WARRANTY; without even the implied warranty of  |")
--print("| MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the    |")
--print("| XXXX XXXX XXXX License (version X.X) for more details.          |")
print("---------------------------------------------------------------------")
print("\n")
print("-------------------------- Grid Definition --------------------------")
print("Grid x dimension: ", grid_options.xnum)
print("Grid y dimension: ", grid_options.ynum)
print("Grid z dimension: ", grid_options.znum)
print("xBoundaryDepth()", grid:xBoundaryDepth())
print("yBoundaryDepth()", grid:yBoundaryDepth())
print("zBoundaryDepth()", grid:zBoundaryDepth())
print("grid xOrigin()", grid:xOrigin())
print("grid yOrigin()", grid:yOrigin())
print("grid zOrigin()", grid:zOrigin())
print("grid xWidth()", grid:xWidth())
print("grid yWidth()", grid:yWidth())
print("grid zWidth()", grid:zWidth())

print("\n")
print("--------------------------- Start Solver ----------------------------")

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

--local Rho = L.NewMacro(function(r)
--    return liszt `r.rho
--end)
local Rho = liszt function(r)
    return r.rho
end

local Velocity = L.NewMacro(function(r)
    return liszt `r.velocity
end)

local Temperature = L.NewMacro(function(r)
    return liszt `r.temperature
end)

local InterpolateTrilinear = L.NewMacro(function(dc, xyz, Field)
    return liszt quote
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
        var dX   = cmath.fmod((xyz[0] - grid_originX)/grid_dx + 0.5, 1.0)
        var dY   = cmath.fmod((xyz[1] - grid_originY)/grid_dy + 0.5, 1.0)
        var dZ   = cmath.fmod((xyz[2] - grid_originZ)/grid_dz + 0.5, 1.0)

        var oneMinusdX = 1.0 - dX
        var oneMinusdY = 1.0 - dY
        var oneMinusdZ = 1.0 - dZ
        var weight00 = Field(c000) * oneMinusdX + Field(c100) * dX 
        var weight10 = Field(c010) * oneMinusdX + Field(c110) * dX
        var weight01 = Field(c001) * oneMinusdX + Field(c101) * dX
        var weight11 = Field(c011) * oneMinusdX + Field(c111) * dX
        var weight0  = weight00 * oneMinusdY + weight10 * dY
        var weight1  = weight01 * oneMinusdY + weight11 * dY
    in
        weight0 * oneMinusdZ + weight1 * dZ
    end
end)

-----------------------------------------------------------------------------
--[[                            LISZT KERNELS                            ]]--
-----------------------------------------------------------------------------

-------
-- FLOW
-------

-- Initialize flow variables
-- Cell center coordinates are stored in the grid field macro 'center'. 
-- Here, we use a field for convenience when outputting to file, but this is
-- to be removed after grid outputing is well defined from within the grid.t 
-- module. Similar story with the vertex coordinates (output only).
Flow.InitializeCenterCoordinates = liszt kernel(c : grid.cells)
    var xy = c.center
    c.centerCoordinates = L.vec3d({xy[0], xy[1], xy[2]})
end

Flow.InitializeVertexCoordinates = liszt kernel(c : grid.dual_cells)
    var xy = c.center
    c.centerCoordinates = L.vec3d({xy[0], xy[1], xy[2]})
end

Flow.InitializePrimitives = liszt kernel(c : grid.cells)
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
          taylorGreenDensity * cmath.pow(taylorGreenVelocity,2) / 16 *
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
          taylorGreenDensity * cmath.pow(taylorGreenVelocity,2) / 16 *
          factorA * factorB
    elseif flow_options.initCase == Flow.Uniform then
      c.rho         = flow_options.initParams[0]
      c.pressure    = flow_options.initParams[1]
      c.velocity[0] = flow_options.initParams[2]
      c.velocity[1] = flow_options.initParams[3]
      c.velocity[2] = flow_options.initParams[4]
    end
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


-- Initialize derivatives
Flow.InitializeTimeDerivatives = liszt kernel(c : grid.cells)
    c.rho_t = L.double(0)
    c.rhoVelocity_t = L.vec3d({0, 0, 0})
    c.rhoEnergy_t = L.double(0)
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
    -- Consider first boundary element (c.xneg_depth == 1) to define left flux
    -- on first interior cell
    if c.in_interior or c.xneg_depth == 1 then
        var directionIdx = 0
        var numInterpolateCoeffs  = spatial_stencil.numInterpolateCoeffs
        var interpolateCoeffs     = spatial_stencil.interpolateCoeffs
        var numFirstDerivativeCoeffs = spatial_stencil.numFirstDerivativeCoeffs
        var firstDerivativeCoeffs    = spatial_stencil.firstDerivativeCoeffs

        -- Diagonal terms
        var rhoFactorDiagonal = L.double(0)
        var rhoVelocityFactorDiagonal = L.vec3d({0.0, 0.0, 0.0})
        var rhoEnergyFactorDiagonal   = L.double(0.0)
        var fpdiag = L.double(0.0)
        for ndx = 1, numInterpolateCoeffs do
            rhoFactorDiagonal += interpolateCoeffs[ndx] *
                          ( c(1-ndx,0,0).rho *
                            c(1-ndx,0,0).velocity[directionIdx] +
                            c(ndx,0,0).rho *
                            c(ndx,0,0).velocity[directionIdx] )
            rhoVelocityFactorDiagonal += interpolateCoeffs[ndx] *
                                   ( c(1-ndx,0,0).rhoVelocity *
                                     c(1-ndx,0,0).velocity[directionIdx] +
                                     c(ndx,0,0).rhoVelocity *
                                     c(ndx,0,0).velocity[directionIdx] )
            rhoEnergyFactorDiagonal += interpolateCoeffs[ndx] *
                                 ( c(1-ndx,0,0).rhoEnthalpy *
                                   c(1-ndx,0,0).velocity[directionIdx] +
                                   c(ndx,0,0).rhoEnthalpy *
                                   c(ndx,0,0).velocity[directionIdx] )
            fpdiag += interpolateCoeffs[ndx] *
                    ( c(1-ndx,0,0).pressure +
                      c(ndx,0,0).pressure )
        end

        -- Skewed terms
        var rhoFactorSkew         = L.double(0)
        var rhoVelocityFactorSkew = L.vec3d({0.0, 0.0, 0.0})
        var rhoEnergyFactorSkew   = L.double(0.0)
        -- mdx = -N+1,...,0
        for mdx = 2-numFirstDerivativeCoeffs, 1 do
          var tmp = L.double(0)
          for ndx = 1, mdx+numFirstDerivativeCoeffs do
            tmp += firstDerivativeCoeffs[ndx-mdx] * 
                   c(ndx,0,0).velocity[directionIdx]
          end

          rhoFactorSkew         += c(mdx,0,0).rho * tmp
          rhoVelocityFactorSkew += c(mdx,0,0).rhoVelocity * tmp
          rhoEnergyFactorSkew   += c(mdx,0,0).rhoEnthalpy * tmp
        end
        --  mdx = 1,...,N
        for mdx = 1,numFirstDerivativeCoeffs do
          var tmp = L.double(0)
          for ndx = mdx-numFirstDerivativeCoeffs+1, 1 do
            tmp += firstDerivativeCoeffs[mdx-ndx] * 
                   c(ndx,0,0).velocity[directionIdx]
          end

          rhoFactorSkew         += c(mdx,0,0).rho * tmp
          rhoVelocityFactorSkew += c(mdx,0,0).rhoVelocity * tmp
          rhoEnergyFactorSkew   += c(mdx,0,0).rhoEnthalpy * tmp
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
    -- Consider first boundary element (c.yneg_depth == 1) to define down flux
    -- on first interior cell
    if c.in_interior or c.yneg_depth == 1 then
        var directionIdx = 1
        var numInterpolateCoeffs  = spatial_stencil.numInterpolateCoeffs
        var interpolateCoeffs     = spatial_stencil.interpolateCoeffs
        var numFirstDerivativeCoeffs = spatial_stencil.numFirstDerivativeCoeffs
        var firstDerivativeCoeffs    = spatial_stencil.firstDerivativeCoeffs
        var rhoFactorDiagonal = L.double(0)

        -- Diagonal terms
        var rhoVelocityFactorDiagonal = L.vec3d({0.0, 0.0, 0.0})
        var rhoEnergyFactorDiagonal   = L.double(0.0)
        var fpdiag = L.double(0.0)
        for ndx = 1, numInterpolateCoeffs do
            rhoFactorDiagonal += interpolateCoeffs[ndx] *
                          ( c(0,1-ndx,0).rho *
                            c(0,1-ndx,0).velocity[directionIdx] +
                            c(0,ndx,0).rho *
                            c(0,ndx,0).velocity[directionIdx] )
            rhoVelocityFactorDiagonal += interpolateCoeffs[ndx] *
                                   ( c(0,1-ndx,0).rhoVelocity *
                                     c(0,1-ndx,0).velocity[directionIdx] +
                                     c(0,ndx,0).rhoVelocity *
                                     c(0,ndx,0).velocity[directionIdx] )
            rhoEnergyFactorDiagonal += interpolateCoeffs[ndx] *
                                 ( c(0,1-ndx,0).rhoEnthalpy *
                                   c(0,1-ndx,0).velocity[directionIdx] +
                                   c(0,ndx,0).rhoEnthalpy *
                                   c(0,ndx,0).velocity[directionIdx] )
            fpdiag += interpolateCoeffs[ndx] *
                    ( c(0,1-ndx,0).pressure +
                      c(0,ndx,0).pressure )
        end

        -- Skewed terms
        var rhoFactorSkew     = L.double(0)
        var rhoVelocityFactorSkew     = L.vec3d({0.0, 0.0, 0.0})
        var rhoEnergyFactorSkew       = L.double(0.0)
        -- mdx = -N+1,...,0
        for mdx = 2-numFirstDerivativeCoeffs, 1 do
          var tmp = L.double(0)
          for ndx = 1, mdx+numFirstDerivativeCoeffs do
            tmp += firstDerivativeCoeffs[ndx-mdx] * 
                   c(0,ndx,0).velocity[directionIdx]
          end

          rhoFactorSkew         += c(0,mdx,0).rho * tmp
          rhoVelocityFactorSkew += c(0,mdx,0).rhoVelocity * tmp
          rhoEnergyFactorSkew   += c(0,mdx,0).rhoEnthalpy * tmp
        end
        --  mdx = 1,...,N
        for mdx = 1,numFirstDerivativeCoeffs do
          var tmp = L.double(0)
          for ndx = mdx-numFirstDerivativeCoeffs+1, 1 do
            tmp += firstDerivativeCoeffs[mdx-ndx] * 
                   c(0,ndx,0).velocity[directionIdx]
          end

          rhoFactorSkew         += c(0,mdx,0).rho * tmp
          rhoVelocityFactorSkew += c(0,mdx,0).rhoVelocity * tmp
          rhoEnergyFactorSkew   += c(0,mdx,0).rhoEnthalpy * tmp
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

-- Compute inviscid fluxes in Z direction
Flow.AddInviscidGetFluxZ =  liszt kernel(c : grid.cells)
    -- Consider first boundary element (c.zneg_depth == 1) to define down flux
    -- on first interior cell
    if c.in_interior or c.zneg_depth == 1 then
        var directionIdx = 2
        var numInterpolateCoeffs     = spatial_stencil.numInterpolateCoeffs
        var interpolateCoeffs        = spatial_stencil.interpolateCoeffs
        var numFirstDerivativeCoeffs = spatial_stencil.numFirstDerivativeCoeffs
        var firstDerivativeCoeffs    = spatial_stencil.firstDerivativeCoeffs
        var rhoFactorDiagonal        = L.double(0)

        -- Diagonal terms
        var rhoVelocityFactorDiagonal = L.vec3d({0.0, 0.0, 0.0})
        var rhoEnergyFactorDiagonal   = L.double(0.0)
        var fpdiag = L.double(0.0)
        for ndx = 1, numInterpolateCoeffs do
            rhoFactorDiagonal += interpolateCoeffs[ndx] *
                          ( c(0,0,1-ndx).rho *
                            c(0,0,1-ndx).velocity[directionIdx] +
                            c(0,0,  ndx).rho *
                            c(0,0,  ndx).velocity[directionIdx] )
            rhoVelocityFactorDiagonal += interpolateCoeffs[ndx] *
                                   ( c(0,0,1-ndx).rhoVelocity *
                                     c(0,0,1-ndx).velocity[directionIdx] +
                                     c(0,0,  ndx).rhoVelocity *
                                     c(0,0,  ndx).velocity[directionIdx] )
            rhoEnergyFactorDiagonal += interpolateCoeffs[ndx] *
                                 ( c(0,0,1-ndx).rhoEnthalpy *
                                   c(0,0,1-ndx).velocity[directionIdx] +
                                   c(0,0,  ndx).rhoEnthalpy *
                                   c(0,0,  ndx).velocity[directionIdx] )
            fpdiag += interpolateCoeffs[ndx] *
                    ( c(0,0,1-ndx).pressure +
                      c(0,0,  ndx).pressure )
        end

        -- Skewed terms
        var rhoFactorSkew             = L.double(0)
        var rhoVelocityFactorSkew     = L.vec3d({0.0, 0.0, 0.0})
        var rhoEnergyFactorSkew       = L.double(0.0)
        -- mdx = -N+1,...,0
        for mdx = 2-numFirstDerivativeCoeffs, 1 do
          var tmp = L.double(0)
          for ndx = 1, mdx+numFirstDerivativeCoeffs do
            tmp += firstDerivativeCoeffs[ndx-mdx] * 
                   c(0,0,ndx).velocity[directionIdx]
          end

          rhoFactorSkew         += c(0,0,mdx).rho * tmp
          rhoVelocityFactorSkew += c(0,0,mdx).rhoVelocity * tmp
          rhoEnergyFactorSkew   += c(0,0,mdx).rhoEnthalpy * tmp
        end
        --  mdx = 1,...,N
        for mdx = 1,numFirstDerivativeCoeffs do
          var tmp = L.double(0)
          for ndx = mdx-numFirstDerivativeCoeffs+1, 1 do
            tmp += firstDerivativeCoeffs[mdx-ndx] * 
                   c(0,0,ndx).velocity[directionIdx]
          end

          rhoFactorSkew         += c(0,0,mdx).rho * tmp
          rhoVelocityFactorSkew += c(0,0,mdx).rhoVelocity * tmp
          rhoEnergyFactorSkew   += c(0,0,mdx).rhoEnthalpy * tmp
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
-- WARNING_START For non-uniform grids, the metrics used below 
-- (grid_dx, grid_dy, grid_dz) are not  appropriate and should be changed 
-- to reflect those expressed in the Python prototype code
-- WARNING_END
Flow.AddInviscidUpdateUsingFluxX = liszt kernel(c : grid.cells)
    c.rho_t -= (c( 0,0,0).rhoFlux -
                c(-1,0,0).rhoFlux)/grid_dx
    c.rhoVelocity_t -= (c( 0,0,0).rhoVelocityFlux -
                        c(-1,0,0).rhoVelocityFlux)/grid_dx
    c.rhoEnergy_t -= (c( 0,0,0).rhoEnergyFlux -
                      c(-1,0,0).rhoEnergyFlux)/grid_dx
end
Flow.AddInviscidUpdateUsingFluxY = liszt kernel(c : grid.cells)
    c.rho_t -= (c(0, 0,0).rhoFlux -
                c(0,-1,0).rhoFlux)/grid_dy
    c.rhoVelocity_t -= (c(0, 0,0).rhoVelocityFlux -
                        c(0,-1,0).rhoVelocityFlux)/grid_dy
    c.rhoEnergy_t -= (c(0, 0,0).rhoEnergyFlux -
                      c(0,-1,0).rhoEnergyFlux)/grid_dy
end
Flow.AddInviscidUpdateUsingFluxZ = liszt kernel(c : grid.cells)
    c.rho_t -= (c(0,0, 0).rhoFlux -
                c(0,0,-1).rhoFlux)/grid_dz
    c.rhoVelocity_t -= (c(0,0, 0).rhoVelocityFlux -
                        c(0,0,-1).rhoVelocityFlux)/grid_dz
    c.rhoEnergy_t -= (c(0,0, 0).rhoEnergyFlux -
                      c(0,0,-1).rhoEnergyFlux)/grid_dz
end

----------
-- Viscous
----------

-- Compute viscous fluxes in X direction
Flow.AddViscousGetFluxX =  liszt kernel(c : grid.cells)
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
        var numInterpolateCoeffs  = spatial_stencil.numInterpolateCoeffs
        var interpolateCoeffs     = spatial_stencil.interpolateCoeffs
        -- Interpolate velocity and derivatives to face
        for ndx = 1, numInterpolateCoeffs do
            velocityFace += interpolateCoeffs[ndx] *
                          ( c(1-ndx,0,0).velocity +
                            c(  ndx,0,0).velocity )
            velocityX_YFace += interpolateCoeffs[ndx] *
                               ( c(1-ndx,0,0).velocityGradientY[0] +
                                 c(  ndx,0,0).velocityGradientY[0] )
            velocityX_ZFace += interpolateCoeffs[ndx] *
                               ( c(1-ndx,0,0).velocityGradientZ[0] +
                                 c(  ndx,0,0).velocityGradientZ[0] )
            velocityY_YFace += interpolateCoeffs[ndx] *
                               ( c(1-ndx,0,0).velocityGradientY[1] +
                                 c(  ndx,0,0).velocityGradientY[1] )
            velocityZ_ZFace += interpolateCoeffs[ndx] *
                               ( c(1-ndx,0,0).velocityGradientZ[2] +
                                 c(  ndx,0,0).velocityGradientZ[2] )
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
            ( c(ndx,0,0).velocity[0] - c(1-ndx,0,0).velocity[0] )
          velocityY_XFace += firstDerivativeCoeffs[ndx] *
            ( c(ndx,0,0).velocity[1] - c(1-ndx,0,0).velocity[1] )
          velocityZ_XFace += firstDerivativeCoeffs[ndx] *
            ( c(ndx,0,0).velocity[2] - c(1-ndx,0,0).velocity[2] )
          temperature_XFace += firstDerivativeCoeffs[ndx] *
            ( c(ndx,0,0).temperature - c(1-ndx,0,0).temperature )
        end
       
        velocityX_XFace   /= grid_dx
        velocityY_XFace   /= grid_dx
        velocityZ_XFace   /= grid_dx
        temperature_XFace /= grid_dx

        -- Tensor components (at face)
        var sigmaXX = muFace * ( 4.0 * velocityX_XFace -
                                 2.0 * velocityY_YFace -
                                 2.0 * velocityZ_ZFace ) / 3.0
        var sigmaYX = muFace * ( velocityY_XFace + velocityX_YFace )
        var sigmaZX = muFace * ( velocityZ_XFace + velocityX_ZFace )
        var usigma = velocityFace[0] * sigmaXX +
                     velocityFace[1] * sigmaYX +
                     velocityFace[2] * sigmaZX
        var cp = fluid_options.gamma * fluid_options.gasConstant / 
                 (fluid_options.gamma - 1.0)
        var heatFlux = - cp / fluid_options.prandtl * 
                         muFace * temperature_XFace

        -- Fluxes
        c.rhoVelocityFlux[0] = sigmaXX
        c.rhoVelocityFlux[1] = sigmaYX
        c.rhoVelocityFlux[2] = sigmaZX
        c.rhoEnergyFlux = usigma - heatFlux
        -- WARNING: Add SGS terms for LES

    end
end

-- Compute viscous fluxes in Y direction
Flow.AddViscousGetFluxY =  liszt kernel(c : grid.cells)
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
        var numInterpolateCoeffs  = spatial_stencil.numInterpolateCoeffs
        var interpolateCoeffs     = spatial_stencil.interpolateCoeffs
        -- Interpolate velocity and derivatives to face
        for ndx = 1, numInterpolateCoeffs do
            velocityFace += interpolateCoeffs[ndx] *
                          ( c(0,1-ndx,0).velocity +
                            c(0,ndx,0).velocity )
            velocityY_XFace += interpolateCoeffs[ndx] *
                               ( c(0,1-ndx,0).velocityGradientX[1] +
                                 c(0,  ndx,0).velocityGradientX[1] )
            velocityY_ZFace += interpolateCoeffs[ndx] *
                               ( c(0,1-ndx,0).velocityGradientZ[1] +
                                 c(0,  ndx,0).velocityGradientZ[1] )
            velocityX_XFace += interpolateCoeffs[ndx] *
                               ( c(0,1-ndx,0).velocityGradientX[0] +
                                 c(0,  ndx,0).velocityGradientX[0] )
            velocityZ_ZFace += interpolateCoeffs[ndx] *
                               ( c(0,1-ndx,0).velocityGradientZ[2] +
                                 c(0,  ndx,0).velocityGradientZ[2] )
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
            ( c(0,ndx,0).velocity[0] - c(0,1-ndx,0).velocity[0] )
          velocityY_YFace += firstDerivativeCoeffs[ndx] *
            ( c(0,ndx,0).velocity[1] - c(0,1-ndx,0).velocity[1] )
          velocityZ_YFace += firstDerivativeCoeffs[ndx] *
            ( c(0,ndx,0).velocity[2] - c(0,1-ndx,0).velocity[2] )
          temperature_YFace += firstDerivativeCoeffs[ndx] *
            ( c(0,ndx,0).temperature - c(0,1-ndx,0).temperature )
        end
       
        velocityX_YFace   /= grid_dy
        velocityY_YFace   /= grid_dy
        velocityZ_YFace   /= grid_dy
        temperature_YFace /= grid_dy

        -- Tensor components (at face)
        var sigmaXY = muFace * ( velocityX_YFace + velocityY_XFace )
        var sigmaYY = muFace * ( 4.0 * velocityY_YFace -
                                 2.0 * velocityX_XFace -
                                 2.0 * velocityZ_ZFace ) / 3.0
        var sigmaZY = muFace * ( velocityZ_YFace + velocityY_ZFace )
        var usigma = velocityFace[0] * sigmaXY +
                     velocityFace[1] * sigmaYY +
                     velocityFace[2] * sigmaZY
        var cp = fluid_options.gamma * fluid_options.gasConstant / 
                 (fluid_options.gamma - 1.0)
        var heatFlux = - cp / fluid_options.prandtl * 
                         muFace * temperature_YFace

        -- Fluxes
        c.rhoVelocityFlux[0] = sigmaXY
        c.rhoVelocityFlux[1] = sigmaYY
        c.rhoVelocityFlux[2] = sigmaZY
        c.rhoEnergyFlux = usigma - heatFlux
        -- WARNING: Add SGS terms for LES

    end
end

-- Compute viscous fluxes in Z direction
Flow.AddViscousGetFluxZ =  liszt kernel(c : grid.cells)
    -- Consider first boundary element (c.zneg_depth == 1) to define down flux
    -- on first interior cell
    if c.in_interior or c.zneg_depth == 1 then
        var muFace = 0.5 * (GetDynamicViscosity(c(0,0,0).temperature) +
                            GetDynamicViscosity(c(0,0,1).temperature))
        var velocityFace    = L.vec3d({0.0, 0.0, 0.0})
        var velocityZ_XFace = L.double(0)
        var velocityZ_YFace = L.double(0)
        var velocityX_XFace = L.double(0)
        var velocityY_YFace = L.double(0)
        var numInterpolateCoeffs  = spatial_stencil.numInterpolateCoeffs
        var interpolateCoeffs     = spatial_stencil.interpolateCoeffs
        -- Interpolate velocity and derivatives to face
        for ndx = 1, numInterpolateCoeffs do
            velocityFace += interpolateCoeffs[ndx] *
                          ( c(0,0,1-ndx).velocity +
                            c(0,0,  ndx).velocity )
            velocityZ_XFace += interpolateCoeffs[ndx] *
                               ( c(0,0,1-ndx).velocityGradientX[2] +
                                 c(0,0,  ndx).velocityGradientX[2] )
            velocityZ_YFace +=  interpolateCoeffs[ndx] *
                               ( c(0,0,1-ndx).velocityGradientY[2] +
                                 c(0,0,  ndx).velocityGradientY[2] )
            velocityX_XFace += interpolateCoeffs[ndx] *
                               ( c(0,0,1-ndx).velocityGradientX[0] +
                                 c(0,0,  ndx).velocityGradientX[0] )
            velocityY_YFace += interpolateCoeffs[ndx] *
                               ( c(0,0,1-ndx).velocityGradientY[1] +
                                 c(0,0,  ndx).velocityGradientY[1] )
        end

        -- Differentiate at face
        var velocityX_ZFace = L.double(0)
        var velocityY_ZFace = L.double(0)
        var velocityZ_ZFace = L.double(0)
        var temperature_ZFace = L.double(0)
        var numFirstDerivativeCoeffs = spatial_stencil.numFirstDerivativeCoeffs
        var firstDerivativeCoeffs    = spatial_stencil.firstDerivativeCoeffs
        for ndx = 1, numFirstDerivativeCoeffs do
          velocityX_ZFace += firstDerivativeCoeffs[ndx] *
            ( c(0,0,ndx).velocity[0] - c(0,0,1-ndx).velocity[0] )
          velocityY_ZFace += firstDerivativeCoeffs[ndx] *
            ( c(0,0,ndx).velocity[1] - c(0,0,1-ndx).velocity[1] )
          velocityZ_ZFace += firstDerivativeCoeffs[ndx] *
            ( c(0,0,ndx).velocity[2] - c(0,0,1-ndx).velocity[2] )
          temperature_ZFace += firstDerivativeCoeffs[ndx] *
            ( c(0,0,ndx).temperature - c(0,0,1-ndx).temperature )
        end
       
        velocityX_ZFace   /= grid_dz
        velocityZ_ZFace   /= grid_dz
        velocityZ_ZFace   /= grid_dz
        temperature_ZFace /= grid_dz

        -- Tensor components (at face)
        var sigmaXZ = muFace * ( velocityX_ZFace + velocityZ_XFace )
        var sigmaYZ = muFace * ( velocityY_ZFace + velocityZ_YFace )
        var sigmaZZ = muFace * ( 4.0 * velocityZ_ZFace -
                                 2.0 * velocityX_XFace -
                                 2.0 * velocityY_YFace ) / 3.0
        var usigma = velocityFace[0] * sigmaXZ +
                     velocityFace[1] * sigmaYZ +
                     velocityFace[2] * sigmaZZ
        var cp = fluid_options.gamma * fluid_options.gasConstant / 
                 (fluid_options.gamma - 1.0)
        var heatFlux = - cp / fluid_options.prandtl * 
                         muFace * temperature_ZFace

        -- Fluxes
        c.rhoVelocityFlux[0] = sigmaXZ
        c.rhoVelocityFlux[1] = sigmaYZ
        c.rhoVelocityFlux[2] = sigmaZZ
        c.rhoEnergyFlux = usigma - heatFlux
        -- WARNING: Add SGS terms for LES

    end
end

Flow.AddViscousUpdateUsingFluxX = liszt kernel(c : grid.cells)
    c.rhoVelocity_t += (c(0,0,0).rhoVelocityFlux -
                        c(-1,0,0).rhoVelocityFlux)/grid_dx
    c.rhoEnergy_t   += (c(0,0,0).rhoEnergyFlux -
                        c(-1,0,0).rhoEnergyFlux)/grid_dx
end

Flow.AddViscousUpdateUsingFluxY = liszt kernel(c : grid.cells)
    c.rhoVelocity_t += (c(0,0,0).rhoVelocityFlux -
                        c(0,-1,0).rhoVelocityFlux)/grid_dy
    c.rhoEnergy_t   += (c(0,0,0).rhoEnergyFlux -
                        c(0,-1,0).rhoEnergyFlux)/grid_dy
end

Flow.AddViscousUpdateUsingFluxZ = liszt kernel(c : grid.cells)
    c.rhoVelocity_t += (c(0,0, 0).rhoVelocityFlux -
                        c(0,0,-1).rhoVelocityFlux)/grid_dz
    c.rhoEnergy_t   += (c(0,0, 0).rhoEnergyFlux -
                        c(0,0,-1).rhoEnergyFlux)/grid_dz
end

---------------------
-- Particles coupling
---------------------

Flow.AddParticlesCoupling = liszt kernel(p : particles)
    if p.state == 1 then
        -- WARNING: Assumes that deltaVelocityOverRelaxationTime and 
        -- deltaTemperatureTerm have been computed previously 
        -- (for example, when adding the flow coupling to the particles, 
        -- which should be called before in the time stepper)

        -- Retrieve cell containing this particle
        p.cell = grid.cell_locate(p.position)
        -- Add contribution to momentum and energy equations from the previously
        -- computed deltaVelocityOverRelaxationTime and deltaTemperatureTerm
        p.cell.rhoVelocity_t -= p.mass * p.deltaVelocityOverRelaxationTime
        p.cell.rhoEnergy_t   -= p.deltaTemperatureTerm
    end
end

--------------
-- Body Forces
--------------

Flow.AddBodyForces = liszt kernel(c : grid.cells)
    -- Add body forces to momentum equation
    c.rhoVelocity_t += c.rho *
                       flow_options.bodyForce
end

-----------------
-- Update kernels
-----------------

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

Flow.UpdateAuxiliaryVelocity = liszt kernel(c : grid.cells)
    var velocity = c.rhoVelocity / c.rho
    c.velocity = velocity
    c.kineticEnergy = 0.5 *  L.dot(velocity,velocity)
end

Flow.UpdateGhostFieldsStep1 = liszt kernel(c : grid.cells)
    if c.xneg_depth > 0 then
        var xoffset = XOffset(c.xneg_depth)
        c.rhoBoundary            =   c(xoffset,0,0).rho
        c.rhoVelocityBoundary[0] =   c(xoffset,0,0).rhoVelocity[0] * xSignX + xBCLeftVelX
        c.rhoVelocityBoundary[1] =   c(xoffset,0,0).rhoVelocity[1] * xSignY + xBCLeftVelY
        c.rhoVelocityBoundary[2] =   c(xoffset,0,0).rhoVelocity[2] * xSignZ + xBCLeftVelZ
        c.rhoEnergyBoundary      =   c(xoffset,0,0).rhoEnergy
        c.velocityBoundary[0]    =   c(xoffset,0,0).velocity[0] * xSignX + xBCLeftVelX
        c.velocityBoundary[1]    =   c(xoffset,0,0).velocity[1] * xSignY + xBCLeftVelY
        c.velocityBoundary[2]    =   c(xoffset,0,0).velocity[2] * xSignZ + xBCLeftVelZ
        c.pressureBoundary       =   c(xoffset,0,0).pressure
        c.temperatureBoundary    =   c(xoffset,0,0).temperature
    end
    if c.xpos_depth > 0 then
        var xoffset = XOffset(c.xpos_depth)
        c.rhoBoundary            =   c(-xoffset,0,0).rho
        c.rhoVelocityBoundary[0] =   c(-xoffset,0,0).rhoVelocity[0] * xSignX + xBCRightVelX
        c.rhoVelocityBoundary[1] =   c(-xoffset,0,0).rhoVelocity[1] * xSignY + xBCRightVelY
        c.rhoVelocityBoundary[2] =   c(-xoffset,0,0).rhoVelocity[2] * xSignZ + xBCRightVelZ
        c.rhoEnergyBoundary      =   c(-xoffset,0,0).rhoEnergy
        c.velocityBoundary[0]    =   c(-xoffset,0,0).velocity[0] * xSignX + xBCRightVelX
        c.velocityBoundary[1]    =   c(-xoffset,0,0).velocity[1] * xSignY + xBCRightVelY
        c.velocityBoundary[2]    =   c(-xoffset,0,0).velocity[2] * xSignZ + xBCRightVelZ
        c.pressureBoundary       =   c(-xoffset,0,0).pressure
        c.temperatureBoundary    =   c(-xoffset,0,0).temperature
    end
    if c.yneg_depth > 0 then
        var yoffset = YOffset(c.yneg_depth)
        c.rhoBoundary            =   c(0,yoffset,0).rho
        c.rhoVelocityBoundary[0] =   c(0,yoffset,0).rhoVelocity[0] * ySignX + yBCLeftVelX
        c.rhoVelocityBoundary[1] =   c(0,yoffset,0).rhoVelocity[1] * ySignY + yBCLeftVelY
        c.rhoVelocityBoundary[2] =   c(0,yoffset,0).rhoVelocity[2] * ySignZ + yBCLeftVelZ
        c.rhoEnergyBoundary      =   c(0,yoffset,0).rhoEnergy
        c.velocityBoundary[0]    =   c(0,yoffset,0).velocity[0] * ySignX + yBCLeftVelX
        c.velocityBoundary[1]    =   c(0,yoffset,0).velocity[1] * ySignY + yBCLeftVelY
        c.velocityBoundary[2]    =   c(0,yoffset,0).velocity[2] * ySignZ + yBCLeftVelZ
        c.pressureBoundary       =   c(0,yoffset,0).pressure
        c.temperatureBoundary    =   c(0,yoffset,0).temperature
    end
    if c.ypos_depth > 0 then
        var yoffset = YOffset(c.ypos_depth)
        c.rhoBoundary            =   c(0,-yoffset,0).rho
        c.rhoVelocityBoundary[0] =   c(0,-yoffset,0).rhoVelocity[0] * ySignX + yBCRightVelX
        c.rhoVelocityBoundary[1] =   c(0,-yoffset,0).rhoVelocity[1] * ySignY + yBCRightVelY
        c.rhoVelocityBoundary[2] =   c(0,-yoffset,0).rhoVelocity[2] * ySignZ + yBCRightVelZ
        c.rhoEnergyBoundary      =   c(0,-yoffset,0).rhoEnergy
        c.velocityBoundary[0]    =   c(0,-yoffset,0).velocity[0] * ySignX + yBCRightVelX
        c.velocityBoundary[1]    =   c(0,-yoffset,0).velocity[1] * ySignY + yBCRightVelY
        c.velocityBoundary[2]    =   c(0,-yoffset,0).velocity[2] * ySignZ + yBCRightVelZ
        c.pressureBoundary       =   c(0,-yoffset,0).pressure
        c.temperatureBoundary    =   c(0,-yoffset,0).temperature
    end
    if c.zneg_depth > 0 then
        var zoffset = ZOffset(c.zneg_depth)
        c.rhoBoundary            =   c(0,0,zoffset).rho
        c.rhoVelocityBoundary[0] =   c(0,0,zoffset).rhoVelocity[0] * zSignX + zBCLeftVelX
        c.rhoVelocityBoundary[1] =   c(0,0,zoffset).rhoVelocity[1] * zSignY + zBCLeftVelY
        c.rhoVelocityBoundary[2] =   c(0,0,zoffset).rhoVelocity[2] * zSignZ + zBCLeftVelZ
        c.rhoEnergyBoundary      =   c(0,0,zoffset).rhoEnergy
        c.velocityBoundary[0]    =   c(0,0,zoffset).velocity[0] * zSignX + zBCLeftVelX
        c.velocityBoundary[1]    =   c(0,0,zoffset).velocity[1] * zSignY + zBCLeftVelY
        c.velocityBoundary[2]    =   c(0,0,zoffset).velocity[2] * zSignZ + zBCLeftVelZ
        c.pressureBoundary       =   c(0,0,zoffset).pressure
        c.temperatureBoundary    =   c(0,0,zoffset).temperature
    end
    if c.zpos_depth > 0 then
        var zoffset = ZOffset(c.zpos_depth)
        c.rhoBoundary            =   c(0,0,-zoffset).rho
        c.rhoVelocityBoundary[0] =   c(0,0,-zoffset).rhoVelocity[0] * zSignX + zBCRightVelX
        c.rhoVelocityBoundary[1] =   c(0,0,-zoffset).rhoVelocity[1] * zSignY + zBCRightVelY
        c.rhoVelocityBoundary[2] =   c(0,0,-zoffset).rhoVelocity[2] * zSignZ + zBCRightVelZ
        c.rhoEnergyBoundary      =   c(0,0,-zoffset).rhoEnergy
        c.velocityBoundary[0]    =   c(0,0,-zoffset).velocity[0] * zSignX + zBCRightVelX
        c.velocityBoundary[1]    =   c(0,0,-zoffset).velocity[1] * zSignY + zBCRightVelY
        c.velocityBoundary[2]    =   c(0,0,-zoffset).velocity[2] * zSignZ + zBCRightVelZ
        c.pressureBoundary       =   c(0,0,-zoffset).pressure
        c.temperatureBoundary    =   c(0,0,-zoffset).temperature
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
    if c.xneg_depth > 0 then
        var xoffset = XOffset(c.xneg_depth)
        c.pressureBoundary       =   c(xoffset,0,0).pressure
        c.temperatureBoundary    =   c(xoffset,0,0).temperature
    end
    if c.xpos_depth > 0 then
        var xoffset = XOffset(c.xpos_depth)
        c.pressureBoundary       =   c(-xoffset,0,0).pressure
        c.temperatureBoundary    =   c(-xoffset,0,0).temperature
    end
    if c.yneg_depth > 0 then
        var yoffset = YOffset(c.yneg_depth)
        c.pressureBoundary       =   c(0,yoffset,0).pressure
        c.temperatureBoundary    =   c(0,yoffset,0).temperature
    end
    if c.ypos_depth > 0 then
        var yoffset = YOffset(c.ypos_depth)
        c.pressureBoundary       =   c(0,-yoffset,0).pressure
        c.temperatureBoundary    =   c(0,-yoffset,0).temperature
    end
    if c.zpos_depth > 0 then
        var zoffset = ZOffset(c.zpos_depth)
        c.pressureBoundary       =   c(0,0,-zoffset).pressure
        c.temperatureBoundary    =   c(0,0,-zoffset).temperature
    end
    if c.zneg_depth > 0 then
        var zoffset = ZOffset(c.zneg_depth)
        c.pressureBoundary       =   c(0,0,zoffset).pressure
        c.temperatureBoundary    =   c(0,0,zoffset).temperature
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
    if c.xneg_depth > 0 then
        var xoffset = XOffset(c.xneg_depth)
        c.velocityBoundary[0] =   c(xoffset,0,0).velocity[0] * xSignX + xBCLeftVelX
        c.velocityBoundary[1] =   c(xoffset,0,0).velocity[1] * xSignY + xBCLeftVelY
        c.velocityBoundary[2] =   c(xoffset,0,0).velocity[2] * xSignZ + xBCLeftVelZ
    end
    if c.xpos_depth > 0 then
        var xoffset = XOffset(c.xpos_depth)
        c.velocityBoundary[0] =   c(-xoffset,0,0).velocity[0] * xSignX + xBCRightVelX
        c.velocityBoundary[1] =   c(-xoffset,0,0).velocity[1] * xSignY + xBCRightVelY
        c.velocityBoundary[2] =   c(-xoffset,0,0).velocity[2] * xSignZ + xBCRightVelZ
    end
    if c.yneg_depth > 0 then
        var yoffset = YOffset(c.yneg_depth)
        c.velocityBoundary[0] =   c(0,yoffset,0).velocity[0] * ySignX + yBCLeftVelX
        c.velocityBoundary[1] =   c(0,yoffset,0).velocity[1] * ySignY + yBCLeftVelY
        c.velocityBoundary[2] =   c(0,yoffset,0).velocity[2] * ySignZ + yBCLeftVelZ
    end
    if c.ypos_depth > 0 then
        var yoffset = YOffset(c.ypos_depth)
        c.velocityBoundary[0] =   c(0,-yoffset,0).velocity[0] * ySignX + yBCRightVelX
        c.velocityBoundary[1] =   c(0,-yoffset,0).velocity[1] * ySignY + yBCRightVelY
        c.velocityBoundary[2] =   c(0,-yoffset,0).velocity[2] * ySignZ + yBCRightVelZ
    end
    if c.zneg_depth > 0 then
        var zoffset = ZOffset(c.zneg_depth)
        c.velocityBoundary[0] =   c(0,0,zoffset).velocity[0] * zSignX + zBCLeftVelX
        c.velocityBoundary[1] =   c(0,0,zoffset).velocity[1] * zSignY + zBCLeftVelY
        c.velocityBoundary[2] =   c(0,0,zoffset).velocity[2] * zSignZ + zBCLeftVelZ
    end
    if c.zpos_depth > 0 then
        var zoffset = ZOffset(c.zpos_depth)
        c.velocityBoundary[0] =   c(0,0,-zoffset).velocity[0] * zSignX + zBCRightVelX
        c.velocityBoundary[1] =   c(0,0,-zoffset).velocity[1] * zSignY + zBCRightVelY
        c.velocityBoundary[2] =   c(0,0,-zoffset).velocity[2] * zSignZ + zBCRightVelZ
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
    if c.xneg_depth > 0 then
        var xoffset = XOffset(c.xneg_depth)
        c.rhoBoundary            =   c(xoffset,0,0).rho
        c.rhoVelocityBoundary[0] =   c(xoffset,0,0).rhoVelocity[0] * xSignX + xBCLeftVelX
        c.rhoVelocityBoundary[1] =   c(xoffset,0,0).rhoVelocity[1] * xSignY + xBCLeftVelY
        c.rhoVelocityBoundary[2] =   c(xoffset,0,0).rhoVelocity[2] * xSignZ + xBCLeftVelZ
        c.rhoEnergyBoundary      =   c(xoffset,0,0).rhoEnergy
    end
    if c.xpos_depth > 0 then
        var xoffset = XOffset(c.xpos_depth)
        c.rhoBoundary            =   c(-xoffset,0,0).rho
        c.rhoVelocityBoundary[0] =   c(-xoffset,0,0).rhoVelocity[0] * xSignX + xBCRightVelX
        c.rhoVelocityBoundary[1] =   c(-xoffset,0,0).rhoVelocity[1] * xSignY + xBCRightVelY
        c.rhoVelocityBoundary[2] =   c(-xoffset,0,0).rhoVelocity[2] * xSignZ + xBCRightVelZ
        c.rhoEnergyBoundary      =   c(-xoffset,0,0).rhoEnergy
    end
    if c.yneg_depth > 0 then
        var yoffset = YOffset(c.yneg_depth)
        c.rhoBoundary            =   c(0,yoffset,0).rho
        c.rhoVelocityBoundary[0] =   c(0,yoffset,0).rhoVelocity[0] * ySignX + yBCLeftVelX
        c.rhoVelocityBoundary[1] =   c(0,yoffset,0).rhoVelocity[1] * ySignY + yBCLeftVelY
        c.rhoVelocityBoundary[2] =   c(0,yoffset,0).rhoVelocity[2] * ySignZ + yBCLeftVelZ
        c.rhoEnergyBoundary      =   c(0,yoffset,0).rhoEnergy
    end
    if c.ypos_depth > 0 then
        var yoffset = YOffset(c.ypos_depth)
        c.rhoBoundary            =   c(0,-yoffset,0).rho
        c.rhoVelocityBoundary[0] =   c(0,-yoffset,0).rhoVelocity[0] * ySignX + yBCRightVelX
        c.rhoVelocityBoundary[1] =   c(0,-yoffset,0).rhoVelocity[1] * ySignY + yBCRightVelY
        c.rhoVelocityBoundary[2] =   c(0,-yoffset,0).rhoVelocity[2] * ySignZ + yBCRightVelZ
        c.rhoEnergyBoundary      =   c(0,-yoffset,0).rhoEnergy
    end
    if c.zneg_depth > 0 then
        var zoffset = ZOffset(c.zneg_depth)
        c.rhoBoundary            =   c(0,0,zoffset).rho
        c.rhoVelocityBoundary[0] =   c(0,0,zoffset).rhoVelocity[0] * zSignX + zBCLeftVelX
        c.rhoVelocityBoundary[1] =   c(0,0,zoffset).rhoVelocity[1] * zSignY + zBCLeftVelY
        c.rhoVelocityBoundary[2] =   c(0,0,zoffset).rhoVelocity[2] * zSignZ + zBCLeftVelZ
        c.rhoEnergyBoundary      =   c(0,0,zoffset).rhoEnergy
    end
    if c.zpos_depth > 0 then
        var zoffset = ZOffset(c.zpos_depth)
        c.rhoBoundary            =   c(0,0,-zoffset).rho
        c.rhoVelocityBoundary[0] =   c(0,0,-zoffset).rhoVelocity[0] * zSignX + zBCRightVelX
        c.rhoVelocityBoundary[1] =   c(0,0,-zoffset).rhoVelocity[1] * zSignY + zBCRightVelY
        c.rhoVelocityBoundary[2] =   c(0,0,-zoffset).rhoVelocity[2] * zSignZ + zBCRightVelZ
        c.rhoEnergyBoundary      =   c(0,0,-zoffset).rhoEnergy
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

---------------------
-- Velocity gradients
---------------------

Flow.ComputeVelocityGradientX = liszt kernel(c : grid.cells)
    var numFirstDerivativeCoeffs = spatial_stencil.numFirstDerivativeCoeffs
    var firstDerivativeCoeffs    = spatial_stencil.firstDerivativeCoeffs
    var tmp = L.vec3d({0.0, 0.0, 0.0})
    for ndx = 1, numFirstDerivativeCoeffs do
      tmp += firstDerivativeCoeffs[ndx] * 
              ( c(ndx,0,0).velocity -
                c(-ndx,0,0).velocity )
    end
    c.velocityGradientX = tmp / grid_dx
end

Flow.ComputeVelocityGradientY = liszt kernel(c : grid.cells)
    var numFirstDerivativeCoeffs = spatial_stencil.numFirstDerivativeCoeffs
    var firstDerivativeCoeffs    = spatial_stencil.firstDerivativeCoeffs
    var tmp = L.vec3d({0.0, 0.0, 0.0})
    for ndx = 1, numFirstDerivativeCoeffs do
      tmp += firstDerivativeCoeffs[ndx] * 
              ( c(0,ndx,0).velocity -
                c(0,-ndx,0).velocity )
    end
    c.velocityGradientY = tmp / grid_dy
end

Flow.ComputeVelocityGradientZ = liszt kernel(c : grid.cells)
    var numFirstDerivativeCoeffs = spatial_stencil.numFirstDerivativeCoeffs
    var firstDerivativeCoeffs    = spatial_stencil.firstDerivativeCoeffs
    var tmp = L.vec3d({0.0, 0.0, 0.0})
    for ndx = 1, numFirstDerivativeCoeffs do
      tmp += firstDerivativeCoeffs[ndx] * 
              ( c(0,0,ndx).velocity -
                c(0,0,-ndx).velocity )
    end
    c.velocityGradientZ = tmp / grid_dz
end

Flow.UpdateGhostVelocityGradientStep1 = liszt kernel(c : grid.cells)
    if c.xneg_depth > 0 then
        var xoffset = XOffset(c.xneg_depth)
        c.velocityGradientXBoundary[0] = - c(xoffset,0,0).velocityGradientX[0]
        c.velocityGradientXBoundary[1] =   c(xoffset,0,0).velocityGradientX[1]
        c.velocityGradientXBoundary[2] =   c(xoffset,0,0).velocityGradientX[2]
        c.velocityGradientYBoundary[0] = - c(xoffset,0,0).velocityGradientY[0]
        c.velocityGradientYBoundary[1] =   c(xoffset,0,0).velocityGradientY[1]
        c.velocityGradientYBoundary[2] =   c(xoffset,0,0).velocityGradientY[2]
        c.velocityGradientZBoundary[0] = - c(xoffset,0,0).velocityGradientZ[0]
        c.velocityGradientZBoundary[1] =   c(xoffset,0,0).velocityGradientZ[1]
        c.velocityGradientZBoundary[2] =   c(xoffset,0,0).velocityGradientZ[2]
    end
    if c.xpos_depth > 0 then
        var xoffset = XOffset(c.xpos_depth)
        c.velocityGradientXBoundary[0] = - c(-xoffset,0,0).velocityGradientX[0]
        c.velocityGradientXBoundary[1] =   c(-xoffset,0,0).velocityGradientX[1]
        c.velocityGradientXBoundary[2] =   c(-xoffset,0,0).velocityGradientX[2]
        c.velocityGradientYBoundary[0] = - c(-xoffset,0,0).velocityGradientY[0]
        c.velocityGradientYBoundary[1] =   c(-xoffset,0,0).velocityGradientY[1]
        c.velocityGradientYBoundary[2] =   c(-xoffset,0,0).velocityGradientY[2]
        c.velocityGradientZBoundary[0] = - c(-xoffset,0,0).velocityGradientZ[0]
        c.velocityGradientZBoundary[1] =   c(-xoffset,0,0).velocityGradientZ[1]
        c.velocityGradientZBoundary[2] =   c(-xoffset,0,0).velocityGradientZ[2]
    end
    if c.yneg_depth > 0 then
        var yoffset = YOffset(c.yneg_depth)
        c.velocityGradientXBoundary[0] =   c(0,yoffset,0).velocityGradientX[0]
        c.velocityGradientXBoundary[1] = - c(0,yoffset,0).velocityGradientX[1]
        c.velocityGradientXBoundary[2] =   c(0,yoffset,0).velocityGradientX[2]
        c.velocityGradientYBoundary[0] =   c(0,yoffset,0).velocityGradientY[0]
        c.velocityGradientYBoundary[1] = - c(0,yoffset,0).velocityGradientY[1]
        c.velocityGradientYBoundary[2] =   c(0,yoffset,0).velocityGradientY[2]
        c.velocityGradientZBoundary[0] =   c(0,yoffset,0).velocityGradientZ[0]
        c.velocityGradientZBoundary[1] = - c(0,yoffset,0).velocityGradientZ[1]
        c.velocityGradientZBoundary[2] =   c(0,yoffset,0).velocityGradientZ[2]
    end
    if c.ypos_depth > 0 then
        var yoffset = YOffset(c.ypos_depth)
        c.velocityGradientXBoundary[0] =   c(0,-yoffset,0).velocityGradientX[0]
        c.velocityGradientXBoundary[1] = - c(0,-yoffset,0).velocityGradientX[1]
        c.velocityGradientXBoundary[2] =   c(0,-yoffset,0).velocityGradientX[2]
        c.velocityGradientYBoundary[0] =   c(0,-yoffset,0).velocityGradientY[0]
        c.velocityGradientYBoundary[1] = - c(0,-yoffset,0).velocityGradientY[1]
        c.velocityGradientYBoundary[2] =   c(0,-yoffset,0).velocityGradientY[2]
        c.velocityGradientZBoundary[0] =   c(0,-yoffset,0).velocityGradientZ[0]
        c.velocityGradientZBoundary[1] = - c(0,-yoffset,0).velocityGradientZ[1]
        c.velocityGradientZBoundary[2] =   c(0,-yoffset,0).velocityGradientZ[2]
    end
    if c.zneg_depth > 0 then
        var zoffset = ZOffset(c.zneg_depth)
        c.velocityGradientXBoundary[0] =   c(0,0,zoffset).velocityGradientX[0]
        c.velocityGradientXBoundary[1] =   c(0,0,zoffset).velocityGradientX[1]
        c.velocityGradientXBoundary[2] = - c(0,0,zoffset).velocityGradientX[2]
        c.velocityGradientYBoundary[0] =   c(0,0,zoffset).velocityGradientY[0]
        c.velocityGradientYBoundary[1] =   c(0,0,zoffset).velocityGradientY[1]
        c.velocityGradientYBoundary[2] = - c(0,0,zoffset).velocityGradientY[2]
        c.velocityGradientZBoundary[0] =   c(0,0,zoffset).velocityGradientZ[0]
        c.velocityGradientZBoundary[1] =   c(0,0,zoffset).velocityGradientZ[1]
        c.velocityGradientZBoundary[2] = - c(0,0,zoffset).velocityGradientZ[2]
    end
    if c.zpos_depth > 0 then
        var zoffset = YOffset(c.zpos_depth)
        c.velocityGradientXBoundary[0] =   c(0,0,-zoffset).velocityGradientX[0]
        c.velocityGradientXBoundary[1] =   c(0,0,-zoffset).velocityGradientX[1]
        c.velocityGradientXBoundary[2] = - c(0,0,-zoffset).velocityGradientX[2]
        c.velocityGradientYBoundary[0] =   c(0,0,-zoffset).velocityGradientY[0]
        c.velocityGradientYBoundary[1] =   c(0,0,-zoffset).velocityGradientY[1]
        c.velocityGradientYBoundary[2] = - c(0,0,-zoffset).velocityGradientY[2]
        c.velocityGradientZBoundary[0] =   c(0,0,-zoffset).velocityGradientZ[0]
        c.velocityGradientZBoundary[1] =   c(0,0,-zoffset).velocityGradientZ[1]
        c.velocityGradientZBoundary[2] = - c(0,0,-zoffset).velocityGradientZ[2]
    end
end
Flow.UpdateGhostVelocityGradientStep2 = liszt kernel(c : grid.cells)
    if c.in_boundary then
        c.velocityGradientX = c.velocityGradientXBoundary
        c.velocityGradientY = c.velocityGradientYBoundary
        c.velocityGradientZ = c.velocityGradientZBoundary
    end
end

-- Calculation of spectral radii for clf-based delta time
local maxConvectiveSpectralRadius = L.NewGlobal(L.double, 0)
local maxViscousSpectralRadius  = L.NewGlobal(L.double, 0)
local maxHeatConductionSpectralRadius  = L.NewGlobal(L.double, 0)
Flow.CalculateSpectralRadii = liszt kernel(c : grid.cells)
    var dXYZInverseSquare = 1.0/grid_dx * 1.0/grid_dx +
                            1.0/grid_dy * 1.0/grid_dy +
                            1.0/grid_dz * 1.0/grid_dz
    -- Convective spectral radii
    c.convectiveSpectralRadius = 
       (cmath.fabs(c.velocity[0])/grid_dx  +
        cmath.fabs(c.velocity[1])/grid_dy  +
        cmath.fabs(c.velocity[2])/grid_dz  +
        GetSoundSpeed(c.temperature) * cmath.sqrt(dXYZInverseSquare)) *
       spatial_stencil.firstDerivativeModifiedWaveNumber
    
    -- Viscous spectral radii (including sgs model component)
    var dynamicViscosity = GetDynamicViscosity(c.temperature)
    var eddyViscosity = c.sgsEddyViscosity
    c.viscousSpectralRadius = 
       (2.0 * ( dynamicViscosity + eddyViscosity ) /
        c.rho * dXYZInverseSquare) *
       spatial_stencil.secondDerivativeModifiedWaveNumber
    
    -- Heat conduction spectral radii (including sgs model 
    -- component)
    var cv = fluid_options.gasConstant / 
             (fluid_options.gamma - 1.0)
    var cp = fluid_options.gamma * cv
    var kappa = cp / fluid_options.prandtl *  dynamicViscosity
    
    c.heatConductionSpectralRadius = 
       ((kappa + c.sgsEddyKappa) / (cv * c.rho) * dXYZInverseSquare) *
       spatial_stencil.secondDerivativeModifiedWaveNumber

    maxConvectiveSpectralRadius     max= c.convectiveSpectralRadius
    maxViscousSpectralRadius        max= c.viscousSpectralRadius
    maxHeatConductionSpectralRadius max= c.heatConductionSpectralRadius

end

-------------
-- Statistics
-------------

Flow.IntegrateQuantities = liszt kernel(c : grid.cells)
    -- WARNING: update cellVolume computation for non-uniform grids
    --var cellVolume = c.xCellWidth() * c.yCellWidth() * c.zCellWidth()
    var cellVolume = grid_dx * grid_dy * grid_dz
    Flow.numberOfInteriorCells += 1
    Flow.areaInterior += cellVolume
    Flow.averagePressure += c.pressure * cellVolume
    Flow.averageTemperature += c.temperature * cellVolume
    Flow.averageKineticEnergy += c.kineticEnergy * cellVolume
    Flow.minTemperature min= c.temperature
    Flow.maxTemperature max= c.temperature
end

---------
-- Output
---------

-- Write cells field to output file
Flow.WriteField = function (outputFileNamePrefix,xSize,ySize,zSize,field)
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
        io.write("# ", xSize, " ", ySize, " ", zSize, " ", N, " ", veclen, "\n")
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
        io.write("# ", xSize, " ", ySize, " ", zSize, " ", N, " ", 1, "\n")
        for i=1,N do
            local t = tostring(values[i]):gsub('ULL', ' ')
            -- i-1 to return to 0 indexing
            io.write("", i-1, ' ', t,"\n")
        end
    end
    io.close()
end

----------------
-- Visualization
----------------

-- kernels to draw particles and velocity for debugging purpose
Flow.DrawKernel = liszt kernel (c : grid.cells)
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
Particles.Locate = liszt kernel(p : particles)
    p.dual_cell = grid.dual_locate(p.position)
end

-- Initialize temporaries for time stepper
Particles.InitializeTemporaries = liszt kernel(p : particles)
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
Particles.InitializeTimeDerivatives = liszt kernel(p : particles)
    if p.state == 1 then
        p.position_t = L.vec3d({0, 0, 0})
        p.velocity_t = L.vec3d({0, 0, 0})
        p.temperature_t = L.double(0)
    end
end

-- Update particle fields based on flow fields
Particles.AddFlowCoupling = liszt kernel(p: particles)
    if p.state == 1 then
      p.dual_cell = grid.dual_locate(p.position)
      var flowDensity     = L.double(0)
      var flowVelocity    = L.vec3d({0, 0, 0})
      var flowTemperature = L.double(0)
      var flowDynamicViscosity = L.double(0)
      flowDensity     = InterpolateTrilinear(p.dual_cell, p.position, Rho)
      flowVelocity    = InterpolateTrilinear(p.dual_cell, p.position, Velocity)
      flowTemperature = InterpolateTrilinear(p.dual_cell, p.position, Temperature)
      flowDynamicViscosity = GetDynamicViscosity(flowTemperature)
      p.position_t    += p.velocity
      -- Relaxation time for small particles 
      -- - particles Reynolds number (set to zero for Stokesian)
      var particleReynoldsNumber =
        (p.density * norm(flowVelocity - p.velocity) * p.diameter) / 
        flowDynamicViscosity
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
end

--------------
-- Body forces
--------------

Particles.AddBodyForces= liszt kernel(p : particles)
    if p.state == 1 then
        p.velocity_t += particles_options.bodyForce
    end
end

------------
-- Radiation
------------

Particles.AddRadiation = liszt kernel(p : particles)
    if p.state == 1 then
        -- Calculate absorbed radiation intensity considering optically thin
        -- particles, for a collimated radiation source with negligible 
        -- blackbody self radiation
        var absorbedRadiationIntensity =
          particles_options.absorptivity *
          radiation_options.radiationIntensity *
          p.area / 4.0

        -- Add contribution to particle temperature time evolution
        p.temperature_t += absorbedRadiationIntensity /
                           (p.mass * particles_options.heat_capacity)
    end
end

-- Set particle velocities to underlying flow velocity for initialization
Particles.SetVelocitiesToFlow = liszt kernel(p: particles)
    p.dual_cell = grid.dual_locate(p.position)
    var flow_density     = L.double(0)
    var flow_velocity    = L.vec3d({0, 0, 0})
    var flow_temperature = L.double(0)
    var flowDynamicViscosity = L.double(0)
    flow_velocity    = InterpolateTrilinear(p.dual_cell, p.position, Velocity)
    p.velocity = flow_velocity
end

-- Update particle variables using derivatives
Particles.UpdateKernels = {}
function Particles.GenerateUpdateKernels(relation, stage)
    local coeff_fun  = TimeIntegrator.coeff_function[stage]
    local coeff_time = TimeIntegrator.coeff_time[stage]
    local deltaTime  = TimeIntegrator.deltaTime
    if stage <= 3 then
        return liszt kernel(r : relation)
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
        return liszt kernel(r : relation)
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
    Particles.UpdateKernels[i] = Particles.GenerateUpdateKernels(particles, i)
end

Particles.UpdateAuxiliaryStep1 = liszt kernel(p : particles)
    if p.state == 1 then
        p.position_ghost[0] = p.position[0]
        p.position_ghost[1] = p.position[1]
        p.position_ghost[2] = p.position[2]
        if p.position[0] < gridOriginInteriorX then
            p.position_ghost[0] = p.position[0] + grid_options.xWidth
        end
        if p.position[0] > gridOriginInteriorX + grid_options.xWidth then
            p.position_ghost[0] = p.position[0] - grid_options.xWidth
        end
        if p.position[1] < gridOriginInteriorY then
            p.position_ghost[1] = p.position[1] + grid_options.yWidth
        end
        if p.position[1] > gridOriginInteriorY + grid_options.yWidth then
            p.position_ghost[1] = p.position[1] - grid_options.yWidth
        end
        if p.position[2] < gridOriginInteriorZ then
            p.position_ghost[2] = p.position[2] + grid_options.zWidth
        end
        if p.position[2] > gridOriginInteriorZ + grid_options.zWidth then
            p.position_ghost[2] = p.position[2] - grid_options.zWidth
        end
    end
end
Particles.UpdateAuxiliaryStep2 = liszt kernel(p : particles)
    if p.state == 1 then
        p.position = p.position_ghost
    end
end

---------
-- Feeder
---------

-- Particles feeder
Particles.Feed = liszt kernel(p: particles)

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

        p.position[0] = centerX + (cmath.rand_unity()-0.5) * widthX
        p.position[1] = centerY + (cmath.rand_unity()-0.5) * widthY
        p.position[2] = centerZ + (cmath.rand_unity()-0.5) * widthZ
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
        if cmath.floor(p.id/injectorBox_particlesPerTimeStep) ==
           TimeIntegrator.timeStep then
            p.position[0] = injectorBox_centerX +
                            (cmath.rand_unity()-0.5) * injectorBox_widthX
            p.position[1] = injectorBox_centerY +
                            (cmath.rand_unity()-0.5) * injectorBox_widthY
            p.position[2] = injectorBox_centerZ +
                            (cmath.rand_unity()-0.5) * injectorBox_widthZ
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
             cmath.floor(particles_options.num*injectorA_particlesPerTimeStep/
             (injectorA_particlesPerTimeStep+injectorB_particlesPerTimeStep))
        var numberOfParticlesInB = 
             cmath.ceil(particles_options.num*injectorB_particlesPerTimeStep/
             (injectorA_particlesPerTimeStep+injectorB_particlesPerTimeStep))
        -- Inject particles at injectorA if matching timeStep requirements
        if cmath.floor(p.id/injectorA_particlesPerTimeStep) ==
           TimeIntegrator.timeStep then
            p.position[0] = injectorA_centerX +
                            (cmath.rand_unity()-0.5) * injectorA_widthX
            p.position[1] = injectorA_centerY +
                            (cmath.rand_unity()-0.5) * injectorA_widthY
            p.position[2] = injectorA_centerZ +
                            (cmath.rand_unity()-0.5) * injectorA_widthZ
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
        if cmath.floor((p.id-numberOfParticlesInA)/
                       injectorB_particlesPerTimeStep) ==
           TimeIntegrator.timeStep then
            p.position[0] = injectorB_centerX +
                            (cmath.rand_unity()-0.5) * injectorB_widthX
            p.position[1] = injectorB_centerY +
                            (cmath.rand_unity()-0.5) * injectorB_widthY
            p.position[2] = injectorB_centerZ +
                            (cmath.rand_unity()-0.5) * injectorB_widthZ
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
Particles.Collect = liszt kernel(p: particles)

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

Particles.IntegrateQuantities = liszt kernel(p : particles)
    if p.state == 1 then
        Particles.averageTemperature += p.temperature
    end
end

-------------
-- Output
-------------

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

----------------
-- Visualization
----------------

Particles.DrawKernel = liszt kernel (p : particles)
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
    Flow.AddInviscidInitialize(grid.cells)
    Flow.AddInviscidGetFluxX(grid.cells)
    Flow.AddInviscidUpdateUsingFluxX(grid.cells.interior)
    Flow.AddInviscidGetFluxY(grid.cells)
    Flow.AddInviscidUpdateUsingFluxY(grid.cells.interior)
    Flow.AddInviscidGetFluxZ(grid.cells)
    Flow.AddInviscidUpdateUsingFluxZ(grid.cells.interior)
end

function Flow.UpdateGhostVelocityGradient()
    Flow.UpdateGhostVelocityGradientStep1(grid.cells)
    Flow.UpdateGhostVelocityGradientStep2(grid.cells)
end

function Flow.AddViscous()
    Flow.AddViscousGetFluxX(grid.cells)
    Flow.AddViscousUpdateUsingFluxX(grid.cells.interior)
    Flow.AddViscousGetFluxY(grid.cells)
    Flow.AddViscousUpdateUsingFluxY(grid.cells.interior)
    Flow.AddViscousGetFluxZ(grid.cells)
    Flow.AddViscousUpdateUsingFluxZ(grid.cells.interior)
end

function Flow.Update(stage)
    Flow.UpdateKernels[stage](grid.cells)
end

function Flow.ComputeVelocityGradients()
    Flow.ComputeVelocityGradientX(grid.cells.interior)
    Flow.ComputeVelocityGradientY(grid.cells.interior)
    Flow.ComputeVelocityGradientZ(grid.cells.interior)
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

------------
-- PARTICLES
------------

function Particles.Update(stage)
    Particles.UpdateKernels[stage](particles)
end

function Particles.UpdateAuxiliary()
    Particles.UpdateAuxiliaryStep1(particles)
    Particles.UpdateAuxiliaryStep2(particles)
end

------------------
-- TIME INTEGRATOR
------------------

function TimeIntegrator.SetupTimeStep()
    Particles.Feed(particles)
    Flow.InitializeTemporaries(grid.cells)
    Particles.InitializeTemporaries(particles)
end

function TimeIntegrator.ConcludeTimeStep()
    Particles.Collect(particles)
end

function TimeIntegrator.InitializeTimeDerivatives()
    Flow.InitializeTimeDerivatives(grid.cells)
    Particles.InitializeTimeDerivatives(particles)
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
    Flow.InitializeCenterCoordinates(grid.cells)
    Flow.InitializeVertexCoordinates(grid.dual_cells)
    Flow.InitializePrimitives(grid.cells.interior)
    Flow.UpdateConservedFromPrimitive(grid.cells.interior)
    Flow.UpdateGhost()
    Flow.UpdateAuxiliary()

    Particles.Feed(particles)
    --Particles.Locate(particles)
    --Particles.SetVelocitiesToFlow(particles)
end

function TimeIntegrator.ComputeDFunctionDt()
    Flow.AddInviscid()
    Flow.UpdateGhostVelocityGradient()
    Flow.AddViscous()
    Flow.AddBodyForces(grid.cells.interior)
    Particles.AddFlowCoupling(particles)
    --Flow.AddParticlesCoupling(particles)
    Particles.AddBodyForces(particles)
    --Particles.AddRadiation(particles)
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

    TimeIntegrator.deltaTime:set(TimeIntegrator.cfl / spectralRadius)
    --TimeIntegrator.deltaTime:set(0.01)

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
    Particles.IntegrateQuantities(particles)
    Statistics.UpdateSpatialAverages(grid, particles)
end

-----
-- IO
-----

function IO.WriteOutput(timeStep)

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

    -- Check if we have particles (avoid nan printed to screen)

    local particle_avg_temp = 0.0
    if particles_options.num > 0 then
      particle_avg_temp = Particles.averageTemperature:get()
    end

    -- Ouput the current stats to the console for this iteration

    io.stdout:write(string.format("%8d",timeStep),
    string.format(" %11.6f",TimeIntegrator.simTime:get()),
    string.format(" %11.6f",Flow.averagePressure:get()),
    string.format(" %11.6f",Flow.averageTemperature:get()),
    string.format(" %11.6f",Flow.averageKineticEnergy:get()),
    string.format(" %11.6f",particle_avg_temp),'\n')

    -- Check if it is time to output to file
    if timeStep % TimeIntegrator.outputEveryTimeSteps == 0 then

        -- Native Python output format
        if IO.outputFormat == 0 then
            --print("Time to output")
            local outputFileName = IO.outputFileNamePrefix .. "_" ..
              tostring(timeStep)
            Flow.WriteField(outputFileName .. "_flow",
              grid:xSize(), grid:ySize(), grid:zSize(),
              grid.cells.temperature)
            --Flow.WriteField(outputFileName .. "_flow",
            --  grid:xSize(), grid:ySize(), grid:zSize(),
            --  grid.cells.rho)
            Flow.WriteField(outputFileName .. "_flow",
              grid:xSize(), grid:ySize(), grid:zSize(),
              grid.cells.pressure)
            Flow.WriteField(outputFileName .. "_flow",
              grid:xSize(), grid:ySize(), grid:zSize(),
              grid.cells.kineticEnergy)
            Particles.WriteField(outputFileName .. "_particles",
              particles.position)
            Particles.WriteField(outputFileName .. "_particles",
              particles.velocity)
            Particles.WriteField(outputFileName .. "_particles",
              particles.temperature)
            Particles.WriteField(outputFileName .. "_particles",
              particles.state)
            Particles.WriteField(outputFileName .. "_particles",
              particles.id)
            Particles.WriteField(outputFileName .. "_particles",
              particles.groupID)

        elseif IO.outputFormat == 1 then

            -- Tecplot ASCII format
            local outputFileName = IO.outputFileNamePrefix .. "_" ..
              tostring(timeStep) .. ".dat"

            -- Open file
            local outputFile = io.output(outputFileName)

            -- Write header
            io.write('TITLE = "Data"\n')
            io.write('VARIABLES = "X", "Y", "Z", "Density", "X-Velocity", "Y-Velocity", "Z-Velocity", "Pressure", "Temperature"\n')
            io.write('ZONE STRANDID=', timeStep, ' SOLUTIONTIME=', TimeIntegrator.simTime:get(), ' I=', grid:xSize()+1, ' J=', grid:ySize()+1, ' K=', grid:zSize()+1, ' DATAPACKING=BLOCK VARLOCATION=([4-9]=CELLCENTERED)\n')

            --grid.dual_cells:NewField('boundary_for_output', L.bool)
            --grid.dual_cells.boundary_for_output:Load(false)
            --local record_boundary = liszt kernel(dc : grid.dual_cells)
            --  dc.boundary_for_output =
            --end

            --for i=1,n do
            --  for ...
            --    xyz = F(i,j,k)
            --end

            local s = ''

            -- Write data
            local values = grid.dual_cells.centerCoordinates:DumpToList()
            local N      = grid.dual_cells.centerCoordinates:Size()
            local veclen = grid.dual_cells.centerCoordinates:Type().N

            -- Need to dump all x coords (fastest), then y, then z
            for j=1,veclen do
                s = ''
                for i=1,N do
                    local t = tostring(values[i][j]):gsub('ULL',' ')
                    s = s .. ' ' .. t .. ''
                    if i % 5 == 0 then
                        s = s .. '\n'
                        io.write("", s, "\n")
                        s = ''
                    end
                end
                -- i-1 to return to 0 indexing
                io.write("", s, "\n")
            end

            -- Now write density, velocity, pressure, temperature

            values = grid.cells.rho:DumpToList()
            N      = grid.cells.rho:Size()
            --for j=1,veclen do
            s = ''
            for i=1,N do
              local t = tostring(values[i]):gsub('ULL',' ')
              s = s .. ' ' .. t .. ''
              if i % 5 == 0 then
                s = s .. '\n'
io.write("", s, "\n")
s = ''
end
            end
            -- i-1 to return to 0 indexing
            io.write("", s, "\n")
            --end

            values = grid.cells.velocity:DumpToList()
            N      = grid.cells.velocity:Size()
            veclen = grid.cells.velocity:Type().N
            for j=1,veclen do
             s = ''
            for i=1,N do
            local t = tostring(values[i][j]):gsub('ULL',' ')
            s = s .. ' ' .. t .. ''
              if i % 5 == 0 then
                s = s .. '\n'
io.write("", s, "\n")
s = ''
              end
            end
            io.write("", s, "\n")
            end

            values = grid.cells.pressure:DumpToList()
            N      = grid.cells.pressure:Size()
            --for j=1,veclen do
            s = ''
            for i=1,N do
            local t = tostring(values[i]):gsub('ULL',' ')
            s = s .. ' ' .. t .. ''
            if i % 5 == 0 then
            s = s .. '\n'
io.write("", s, "\n")
s = ''
            end
            end
            -- i-1 to return to 0 indexing
            io.write("", s, "\n")
            --end

            values = grid.cells.temperature:DumpToList()
            N      = grid.cells.temperature:Size()
            --for j=1,veclen do
            s = ''
            for i=1,N do
            local t = tostring(values[i]):gsub('ULL',' ')
            s = s .. ' ' .. t .. ''
            if i % 5 == 0 then
            s = s .. '\n'
io.write("", s, "\n")
s = ''
            end
            end
            -- i-1 to return to 0 indexing
            io.write("", s, "\n")
            --end

            io.close()

            -- Write the Tecplot header
            --Flow.WriteTecplotHeader(outputFileName)

        else
            print("Output format not defined. No output written to disk.")
        end

    end
end

----------------
-- VISUALIZATION
----------------

function Visualization.Draw()
    --vdb.vbegin()
    --vdb.frame()
    --Flow.DrawKernel(grid.cells)
    --Particles.DrawKernel(particles)
    --vdb.vend()
end

-----------------------------------------------------------------------------
--[[                            MAIN EXECUTION                           ]]--
-----------------------------------------------------------------------------

TimeIntegrator.InitializeVariables()

Flow.WriteField(IO.outputFileNamePrefix,
                grid:xSize(), grid:ySize(), grid:zSize(),
                grid.cells.centerCoordinates)
Particles.WriteField(IO.outputFileNamePrefix .. "_particles",
                     particles.diameter)

IO.WriteOutput(TimeIntegrator.timeStep:get())

-- Start an external iteration counter
local iter = 0

-- Main iteration loop

while ((TimeIntegrator.simTime:get() < TimeIntegrator.final_time) and
       (iter < TimeIntegrator.max_iter)) do

    TimeIntegrator.CalculateDeltaTime()
    TimeIntegrator.AdvanceTimeStep()
    Statistics.ComputeSpatialAverages()
    IO.WriteOutput(TimeIntegrator.timeStep:get())
    Visualization.Draw()

    -- Increment iteration counter
    iter = iter + 1

end
