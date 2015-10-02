-- This is a Lua config file for the Soleil code.

-- This case defines the 32x32 lid-driven cavity problem
return {
  
  -- Flow Initialization  Options --
  initCase    = 'Uniform', -- Uniform, Restart, TaylorGreen2DVortex, TaylorGreen3DVortex
  initParams  = {0.000525805,43.4923,0.0,0.0,0.0}, -- necessary input conditions
  bodyForce   = {0,0.0,0}, -- body force in x, y, z
  turbForceCoeff = 0.0,    -- Turbulent linear forcing coefficient (f = A*rho*u)

  restartIter = 110000,
  
  -- Grid Options --
  xnum = 32, -- number of cells in the x-direction
  ynum = 32, -- number of cells in the y-direction
  znum = 1,  -- number of cells in the z-direction
  origin = {0.0, 0.0, 0.0}, -- spatial origin of the computational domain
  xWidth = 1.0,
  yWidth = 1.0,
  zWidth = 1.0/32.0,
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
  yBCRightVel = {34.03, 0.0, 0.0},
  yBCRightTemp = 0.0,
  zBCLeft  = 'symmetry',
  zBCLeftVel = {0.0, 0.0, 0.0},
  zBCLeftTemp = 0.0,
  zBCRight = 'symmetry',
  zBCRightVel = {0.0, 0.0, 0.0},
  zBCRightTemp = 0.0,
  
  --Time Integration Options --
  final_time            = 20.00001,
  max_iter              = 50000,
  cfl                   = 2.0,
  
  --- File Output Options --
  wrtRestart = 'ON',
  wrtVolumeSolution = 'ON',
  wrt1DSlice = 'ON',
  wrtParticleEvolution = 'OFF',
  particleEvolutionIndex = 0,
  outputEveryTimeSteps  = 100,
  restartEveryTimeSteps = 100,
  headerFrequency       = 20,
  outputFormat = 'Tecplot', -- Only 'Tecplot' is currently available
  outputDirectory = '../soleilOutput/', -- relative to the liszt-ebb home directory
  
  -- Fluid Options --
  gasConstant = 287.058,
  gamma = 1.4,
  prandtl = 0.72,
  viscosity_model = 'Sutherland', -- 'Constant', 'PowerLaw', or 'Sutherland'
  constant_visc = 0.1,          -- Value for a constant viscosity [kg/m/s]
  powerlaw_visc_ref = 0.001,    -- Power-law reference viscosity [kg/m/s]
  powerlaw_temp_ref = 273.0,    -- Power-law reference temperature [K]
  suth_visc_ref = 1.716E-5,     -- Sutherland's Law reference viscosity [kg/m/s]
  suth_temp_ref = 273.15,       -- Sutherland's Law referene temperature [K]
  suth_s_ref = 110.4,           -- Sutherland's Law S constant [K]
  
  -- Particle Options --
  initParticles = 'Random', -- 'Random' or 'Restart'
  restartParticleIter = 0,
  particleType = 'Free', -- Fixed or Free
  twoWayCoupling = 'OFF',
  num = 100.0,
  restitutionCoefficient = 1.0,
  convectiveCoefficient = 0.7, -- W m^-2 K^-1
  heatCapacity = 0.7, -- J Kg^-1 K^-1
  initialTemperature = 250, -- K
  density = 8900, --1000, --8900,
  diameter_mean = 1e-5, -- 1.2e-5, --0.03,
  diameter_maxDeviation = 0.0, --0.02,
  bodyForceParticles = {0,-0.0,0}, -- {0,-1.1,0}
  absorptivity = 1.0, -- Equal to emissivity in thermal equilibrium
  -- (Kirchhoff law of thermal radiation)
  
  -- Radiation Options --
  radiationType = 'ON',
  radiationIntensity = 10.0,
  zeroAvgHeatSource = 'OFF'
  
}
