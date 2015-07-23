-- This is a Lua config file for the Soleil code.

-- This case defines the Poiseuille Flow problem
return {
  
  -- Flow Initialization  Options --
  initCase     = 'Uniform', -- Uniform, Restart, TaylorGreen2DVortex, TaylorGreen3DVortex
  initParams = {1.0,100000.0,1.0,0.0,0.0}, -- necessary input conditions
  bodyForce = {1.2,0.0,0}, -- body force in x, y, z
  restartIter = 0,
  
  -- Grid Options --
  xnum = 32, -- number of cells in the x-direction
  ynum = 32, -- number of cells in the y-direction
  znum = 1,  -- number of cells in the z-direction
  origin = {0.0, 0.0, 0.0}, -- spatial origin of the computational domain
  xWidth = 1.0,
  yWidth = 1.0,
  zWidth = 1.0,
  -- BCs: 'periodic,' 'symmetry,' 'adiabatic_wall,' or 'isothermal_wall'
  xBCLeft  = 'periodic',
  xBCLeftVel = {0.0, 0.0, 0.0},
  xBCLeftTemp = 0.0,
  xBCRight = 'periodic',
  xBCRightVel = {0.0, 0.0, 0.0},
  xBCRightTemp = 0.0,
  yBCLeft  = 'adiabatic_wall',
  yBCLeftVel = {0.0, 0.0, 0.0},
  yBCLeftTemp = 0.0,
  yBCRight = 'adiabatic_wall',
  yBCRightVel = {0.0, 0.0, 0.0},
  yBCRightTemp = 0.0,
  zBCLeft  = 'symmetry',
  zBCLeftVel = {0.0, 0.0, 0.0},
  zBCLeftTemp = 0.0,
  zBCRight = 'symmetry',
  zBCRightVel = {0.0, 0.0, 0.0},
  zBCRightTemp = 0.0,
  
  --Time Integration Options --
  final_time            = 2000.00001,
  max_iter              = 5000,
  cfl                   = 2.5, -- Negative CFL implies that we will used fixed delta T
  delta_time            = 1e-4,
  
  --- File Output Options --
  wrtRestart = 'ON',
  wrtVolumeSolution = 'ON',
  wrt1DSlice = 'ON',
  wrtParticleEvolution = 'OFF',
  particleEvolutionIndex = 0,
  outputEveryTimeSteps  = 1000,
  restartEveryTimeSteps = 1000,
  headerFrequency       = 20,
  outputFormat = 'Tecplot', --Tecplot or Python
  outputDirectory = '../soleilOutput/', -- relative to the liszt-in-terra home directory
  
  -- Fluid Options --
  gasConstant = 200.0,
  gamma = 1.25,
  viscosity_model = 'Constant', -- Constant, PowerLaw, Sutherland
  dynamic_viscosity_ref = 0.1, -- constant value
  dynamic_viscosity_temp_ref = 273.15, --Sutherland's
  prandtl = 1.0,

  -- Particle Options --
  initParticles = 'Random', -- 'Random' or 'Restart'
  restartParticleIter = 0,
  particleType = 'Fixed', -- Fixed or Free
  twoWayCoupling = 'OFF',
  num = 1000.0,
  restitutionCoefficient = 1.0,
  convectiveCoefficient = 20000.0, -- W m^-2 K^-1
  heatCapacity = 1000.0, -- J Kg^-1 K^-1
  initialTemperature = 500.0, -- K
  density = 9e3,
  diameter_mean = 1e-2,
  diameter_maxDeviation = 0.0,
  bodyForceParticles = {0.0,-1.0,0},
  absorptivity = 1.0, -- Equal to emissivity in thermal equilibrium
  -- (Kirchhoff law of thermal radiation)

  -- Radiation Options --
  radiationType = 'OFF',
  radiationIntensity = 3e6,
  zeroAvgHeatSource = 'OFF',

  -- vdb visualization --
  visualize = 'OFF', -- ON or OFF
}
