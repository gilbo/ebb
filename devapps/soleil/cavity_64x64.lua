-- This is a Lua config file for the Soleil code.

-- This case defines the 64x64 lid-driven cavity problem
return {
  
  -- Flow Initialization  Options --
  initCase     = 'Uniform', -- Uniform, Restart, TaylorGreen2DVortex, TaylorGreen3DVortex
  initParams = {1.0,71.42857142857143,0.0,0.0,0.0}, -- necessary input conditions
  bodyForce = {0,0.0,0}, -- body force in x, y, z
  turbForceCoeff = 0.0, -- Turbulent linear forcing coefficient (f = A*rho*u)
  restartIter = 20000,
  
  -- Grid Options --
  xnum = 64, -- number of cells in the x-direction
  ynum = 64, -- number of cells in the y-direction
  znum = 1,  -- number of cells in the z-direction
  origin = {0.0, 0.0, 0.0}, -- spatial origin of the computational domain
  xWidth = 1.0,
  yWidth = 1.0,
  zWidth = 0.1,
  -- BCs: 'periodic,' 'symmetry,' 'adiabatic_wall,' or 'isothermal_wall'
  xBCLeft  = 'adiabatic_wall',
  xBCLeftVel = {0.0, 0.0, 0.0},
  xBCLeftTemp = 0.0,
  xBCRight = 'adiabatic_wall',
  xBCRightVel = {0.0, 0.0, 0.0},
  xBCRightTemp = 0.0,
  yBCLeft  = 'adiabatic_wall',
  yBCLeftVel = {0.0, 0.0, 0.0},
  yBCLeftTemp = 0.0,
  yBCRight = 'adiabatic_wall',
  yBCRightVel = {1.0, 0.0, 0.0},
  yBCRightTemp = 0.0,
  zBCLeft  = 'symmetry',
  zBCLeftVel = {0.0, 0.0, 0.0},
  zBCLeftTemp = 0.0,
  zBCRight = 'symmetry',
  zBCRightVel = {0.0, 0.0, 0.0},
  zBCRightTemp = 0.0,
  
  --Time Integration Options --
  final_time            = 2000.00001,
  max_iter              = 50000,
  cfl                   = -2.0, -- Negative CFL implies that we will used fixed delta T
  delta_time            = 1e-3,
  
  --- File Output Options --
  wrtRestart = 'ON',
  wrtVolumeSolution = 'ON',
  wrt1DSlice = 'ON',
  wrtParticleEvolution = 'OFF',
  particleEvolutionIndex = 0,
  outputEveryTimeSteps  = 1000,
  restartEveryTimeSteps = 1000,
  headerFrequency       = 20,
  outputFormat = 'Tecplot', -- Only 'Tecplot' is currently available
  outputDirectory = '../soleilOutput/', -- relative to the liszt-ebb home directory
  
  -- Fluid Options --
  gasConstant = 287.058,
  gamma = 1.4,
  prandtl = 0.72,
  viscosity_model = 'Constant', -- 'Constant', 'PowerLaw', or 'Sutherland'
  constant_visc = 1.0e-3,          -- Value for a constant viscosity [kg/m/s]
  powerlaw_visc_ref = 0.001,    -- Power-law reference viscosity [kg/m/s]
  powerlaw_temp_ref = 273.0,    -- Power-law reference temperature [K]
  suth_visc_ref = 1.716E-5,     -- Sutherland's Law reference viscosity [kg/m/s]
  suth_temp_ref = 273.15,       -- Sutherland's Law referene temperature [K]
  suth_s_ref = 110.4,           -- Sutherland's Law S constant [K]
  
  -- Particle Options --
  initParticles = 'Random', -- 'Random' or 'Restart'
  restartParticleIter = 0,
  particleType = 'Free', -- Fixed or Free
  twoWayCoupling = 'OFF', -- ON or OFF
  num = 1000.0,
  restitutionCoefficient = 1.0,
  convectiveCoefficient = 0.7, -- W m^-2 K^-1
  heatCapacity = 0.7, -- J Kg^-1 K^-1
  initialTemperature = 0.248830, -- K
  density = 8900, --1000, --8900,
  diameter_mean = 1e-3, --1e-5, -- 1.2e-5, --0.03,
  diameter_maxDeviation = 5e-4, --0.02,
  bodyForceParticles = {0,0,0}, -- {0,-1.1,0}
  absorptivity = 1.0, -- Equal to emissivity in thermal equilibrium
  -- (Kirchhoff law of thermal radiation)
  
  -- Radiation Options --
  radiationType = 'OFF', -- ON or OFF
  radiationIntensity = 1e1,
  zeroAvgHeatSource = 'OFF'
  
}
