-- This is a Lua config file for the Soleil code.

-- This case defines the 64x64 lid-driven cavity problem
return {
  
  -- Flow Initialization  Options --
  initCase     = 'Uniform', -- Uniform, Restart, TaylorGreen2DVortex, TaylorGreen3DVortex
  initParams = {1.0,84151.05269999999,0.0,0.0,0.0}, -- necessary input conditions
  bodyForce = {0,0.0,0}, -- body force in x, y, z
  restartIter = 0,
  
  -- Grid Options --
  xnum = 64, -- number of cells in the x-direction
  ynum = 64, -- number of cells in the y-direction
  znum = 1,  -- number of cells in the z-direction
  origin = {0.0, 0.0, 0.0}, -- spatial origin of the computational domain
  xWidth = 1.0,
  yWidth = 1.0,
  zWidth = 1.0,
  -- BCs on each boundary: 'periodic,' 'symmetry,' or 'wall'
  xBCLeft  = 'wall',
  xBCLeftVel = {0.0, 0.0, 0.0},
  xBCRight = 'wall',
  xBCRightVel = {0.0, 0.0, 0.0},
  yBCLeft  = 'wall',
  yBCLeftVel = {0.0, 0.0, 0.0},
  yBCRight = 'wall',
  yBCRightVel = {0.0, 0.0, 0.0},
  zBCLeft  = 'wall',
  zBCLeftVel = {0.0, 0.0, 0.0},
  zBCRight = 'wall',
  zBCRightVel = {0.0, 0.0, 0.0},
  
  -- Spatial Integration Options --
  spatialOrder = 2, -- 2 or 6
  
  --Time Integration Options --
  final_time            = 2000.00001,
  max_iter              = 2500,
  cfl                   = -1.0, -- Negative CFL implies that we will used fixed delta T
  delta_time            = 1e-2,
  
  --- File Output Options --
  wrtRestart = 'ON',
  wrtVolumeSolution = 'ON',
  wrt1DSlice = 'ON',
  outputEveryTimeSteps  = 100,
  restartEveryTimeSteps = 100,
  headerFrequency       = 20,
  outputFormat = 'Tecplot', --Tecplot or Python
  outputDirectory = '../soleilOutput/', -- relative to the liszt-in-terra home directory
  
  -- Fluid Options --
  gasConstant = 287.058,
  gamma = 1.4,
  viscosity_model = 'Constant', -- Constant, PowerLaw, Sutherland
  dynamic_viscosity_ref = 1.0e-3,
  dynamic_viscosity_temp_ref = 293.15,
  prandtl = 36.8,

  -- Particle Options --
  initParticles = 'Restart', -- 'Random' or 'Restart'
  restartParticleIter = 0,
  particleType = 'Free', -- Fixed or Free
  twoWayCoupling = 'OFF', -- ON or OFF
  num = 20.0,
  restitutionCoefficient = 1.0,
  convectiveCoefficient = 275.0, -- W m^-2 K^-1
  heatCapacity = 4e2, -- J Kg^-1 K^-1
  initialTemperature = 293.15, -- K
  density = 1e4, --1000, --8900,
  diameter_mean = 2e-4, --1e-5, -- 1.2e-5, --0.03,
  diameter_maxDeviation = 0.0, --0.02,
  bodyForceParticles = {0,-10,0}, -- {0,-1.1,0}
  emissivity = 0.5, --0.4
  absorptivity = 0.5, -- Equal to emissivity in thermal equilibrium
  -- (Kirchhoff law of thermal radiation)
  
  -- Radiation Options --
  radiationType = 'OFF', -- ON or OFF
  radiationIntensity = 0.0,
  
  -- vdb visualization --
  visualize = 'OFF', -- ON or OFF
}
