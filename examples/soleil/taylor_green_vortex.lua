-- This is a Lua config file for the Soleil code.

-- This case defines the 32x32 lid-driven cavity problem
return {
  
  -- Flow Initialization  Options --
  initCase     = 'TaylorGreen3DVortex', -- Uniform, Restart, TaylorGreen2DVortex, TaylorGreen3DVortex
  initParams = {1,100,2,0.0,0.0}, -- for TGV: first three are density, pressure, velocity
  bodyForce = {0,0.0,0}, -- body force in x, y, z
  restartIter = 10000,
  
  -- Grid Options -- PERIODICITY
  xnum = 64, -- number of cells in the x-direction
  ynum = 64, -- number of cells in the y-direction
  znum = 64,  -- number of cells in the z-direction
  origin = {0.0, 0.0, 0.0}, -- spatial origin of the computational domain
  xWidth = 6.283185307179586,
  yWidth = 6.283185307179586,
  zWidth = 6.283185307179586,
  -- BCs on each boundary: 'periodic,' 'symmetry,' or 'wall'
  xBCLeft  = 'periodic',
  xBCLeftVel = {0.0, 0.0, 0.0},
  xBCRight = 'periodic',
  xBCRightVel = {0.0, 0.0, 0.0},
  yBCLeft  = 'periodic',
  yBCLeftVel = {0.0, 0.0, 0.0},
  yBCRight = 'periodic',
  yBCRightVel = {0.0, 0.0, 0.0},
  zBCLeft  = 'periodic',
  zBCLeftVel = {0.0, 0.0, 0.0},
  zBCRight = 'periodic',
  zBCRightVel = {0.0, 0.0, 0.0},
  
  -- Spatial Integration Options --
  spatialOrder = 2, -- 2 or 6
  
  --Time Integration Options --
  final_time            = 20.00001,
  max_iter              = 20000,
  cfl                   = 1.0, -- Negative CFL implies that we will used fixed delta T
  delta_time            = 1e-5,
  
  --- File Output Options --
  wrtRestart = 'ON',
  wrtVolumeSolution = 'ON',
  wrt1DSlice = 'ON',
  outputEveryTimeSteps  = 50,
  restartEveryTimeSteps = 50,
  headerFrequency       = 20,
  outputFormat = 'Tecplot', --Tecplot or Python
  outputDirectory = '../soleilOutput/', -- relative to the liszt-in-terra home directory
  
  -- Fluid Options --
  gasConstant = 20.4128,
  gamma = 1.4,
  viscosity_model = 'PowerLaw', -- Constant, PowerLaw, Sutherland
  dynamic_viscosity_ref = 0.00044,
  dynamic_viscosity_temp_ref = 1.0,
  prandtl = 0.7,
  
  -- Particle Options --
  particleType = 'Free', -- Fixed or Free
  twoWayCoupling = 'OFF', -- ON or OFF
  num = 10000.0,
  convectiveCoefficient = 0.7, -- W m^-2 K^-1
  heatCapacity = 0.7, -- J Kg^-1 K^-1
  initialTemperature = 20, -- K
  density = 8900, -- kg/m^3
  diameter_mean = 5e-3, -- m
  diameter_maxDeviation = 1e-3, -- m, for statistical distribution
  bodyForceParticles = {0.0,0.0,0.0},
  emissivity = 0.5,
  absorptivity = 0.5, -- Equal to emissivity in thermal equilibrium
  -- (Kirchhoff law of thermal radiation)
  
  -- Radiation Options --
  radiationType = 'OFF', -- ON or OFF
  radiationIntensity = 1e3,
  
  -- vdb visualization --
  visualize = 'OFF', -- ON or OFF
  
}
