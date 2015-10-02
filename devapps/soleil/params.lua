
-- This is a Lua config file for the Soleil-X code. The options (and comments)
-- here are intended to be a template configuration file that can be reused
-- for new cases by changing the desired options. This 'master' config will
-- always be kept up to date with the latest version of the code.

return {
  
  -----------------------------------------------------------------------------
  --[[                            GRID OPTIONS                             ]]--
  -----------------------------------------------------------------------------
  xnum = 32,                -- Number of internal cells in the x-direction
  ynum = 32,                -- Number of internal cells in the y-direction
  znum = 32,                 -- Number of internal cells in the z-direction
  origin = {0.0, 0.0, 0.0}, -- Spatial origin of the computational domain
  xWidth = 6.283185307179586,             -- Physical length of the domain in the x-dir. [m]
  yWidth = 6.283185307179586,             -- Physical length of the domain in the y-dir. [m]
  zWidth = 6.283185307179586,             -- Physical length of the domain in the z-dir. [m]
  xBCLeft  = 'periodic',         -- Boundary conditions on each boundary face.
  xBCLeftVel = {0.0, 0.0, 0.0},  -- Opposite faces must match. Options are
  xBCLeftTemp = 0.0,             -- 'periodic', 'symmetry', 'adiabatic_wall', or
  xBCRight = 'periodic',         -- 'isothermal_wall'. Each BC can also have a
  xBCRightVel = {0.0, 0.0, 0.0}, -- prescribed wall velocity {u,v,w}, which is
  xBCRightTemp = 0.0,            -- always applied, and a wall temperature that
  yBCLeft  = 'periodic',   -- is ignored unless if is an isothermal wall.
  yBCLeftVel = {0.0, 0.0, 0.0},
  yBCLeftTemp = 0.0,
  yBCRight = 'periodic',
  yBCRightVel = {0.0, 0.0, 0.0},
  yBCRightTemp = 0.0,
  zBCLeft  = 'periodic',
  zBCLeftVel = {0.0, 0.0, 0.0},
  zBCLeftTemp = 0.0,
  zBCRight = 'periodic',
  zBCRightVel = {0.0, 0.0, 0.0},
  zBCRightTemp = 0.0,
  
  -----------------------------------------------------------------------------
  --[[                        FLUID PHASE OPTIONS                          ]]--
  -----------------------------------------------------------------------------
  initCase = 'Perturbed',         -- 'Uniform', 'Restart', 'TaylorGreen2DVortex',
                                --  'TaylorGreen3DVortex' or 'Perturbed'
  restartIter = 0,              -- Starting iteration number for flow restart
  initParams = {1.0,101325.0,0.0,0.0,0.0}, -- Input flow conditions.
                                -- Uniform: {density, pressure, u, v, w}
                                -- Restart: unused
                                -- TGV 2D: {density, pressure, vel, null, null}
                                -- TGV 3D: {density, pressure, vel, null, null}
                                -- Perturbed {mean density, pressure, u, v, w}
  bodyForce = {0.0,0.0,0},      -- Body force (acceleration) in x, y, z
  turbForceCoeff = 0.2,         -- Turbulent linear forcing coefficient (f = A*rho*u)
  gasConstant = 287.058,          -- Ideal gas constant, R = cp - cv [J/kg/K]
  gamma = 1.4,                 -- Ratio of specific heats, gamma = cp/cv
  viscosity_model = 'Constant', -- 'Constant', 'PowerLaw', or 'Sutherland'
  constant_visc = 0.004491,          -- Value for a constant viscosity [kg/m/s]
  powerlaw_visc_ref = 0.001,    -- Power-law reference viscosity [kg/m/s]
  powerlaw_temp_ref = 273.0,    -- Power-law reference temperature [K]
  suth_visc_ref = 1.68e-5,      -- Sutherland's Law reference viscosity [kg/m/s]
  suth_temp_ref = 273.0,        -- Sutherland's Law reference temperature [K]
  suth_s_ref = 110.5,           -- Sutherland's Law S constant [K]
  prandtl = 0.72,                -- Prandtl number, Pr
                                -- Note: thermal conductivity, k = cp*visc/Pr
  
  -----------------------------------------------------------------------------
  --[[                       PARTICLE PHASE OPTIONS                        ]]--
  -----------------------------------------------------------------------------
  initParticles = 'Random',        -- Particle init: 'Random' or 'Restart'
  restartParticleIter = 0,         -- Starting iteration for particle restart
  particleType = 'Free',          -- Particle can be 'Fixed' or 'Free' to move
  twoWayCoupling = 'OFF',          -- Enable two-way coupling with fluid.
                                   -- 'ON' is two-way, 'OFF' is fluid->particle
  num = 10000.0,                    -- Prescribe the total number of particles
  restitutionCoefficient = 1.0,    -- Restitution coeff. for wall collisions
  convectiveCoefficient = 20000.0, -- Convective heat transfer coeff. [W/m^2/K]
  heatCapacity = 1000.0,           -- Particle heat capacity,h [J/kg/K]
                                   -- Note: Nusselt = h*Dp/k
  absorptivity = 1.0,              -- Radiation absorption coeff., 0.0 <-> 1.0
  initialTemperature = 400.0,      -- Initial temperature [K]
  density = 9e3,                   -- Particle density [kg/m^3]
  diameter_mean = 1e-2,            -- Mean value for stochastic diameter, Dp [m]
  diameter_maxDeviation = 0.0,     -- Maximum deviation for stochastic diameter
  bodyForceParticles = {0.0,0.0,0.0}, -- Constant body force (acceleration)
                                   -- on particles in the {x,y,z} directions
  
  -----------------------------------------------------------------------------
  --[[                         RADIATION OPTIONS                           ]]--
  -----------------------------------------------------------------------------
  radiationType = 'OFF',     -- Enable algebraic radiation model, 'ON' or 'OFF'
  radiationIntensity = 3e6,  -- Radiation intensity as a heat flux [W/m^2]
                             -- Note:
  zeroAvgHeatSource = 'OFF', -- Subtract the average heat addition due to
                             -- from all cells to enable a steady problem
                             -- when radiation is applied.
  
  -----------------------------------------------------------------------------
  --[[                      TIME INTEGRATION OPTIONS                       ]]--
  -----------------------------------------------------------------------------
  final_time = 2000.00001, -- Maximum physical time for the simulation [s]
  max_iter = 30000,         -- Maximum number of iterations
  cfl = 0.3,               -- CFL condition. Setting this to a negative value
                           -- imposes a fixed time step that is given by
                           -- the 'delta_time' config option.
  delta_time = 1e-4,       -- Fixed time step [s], ignored if CFL > 0.0
  
  -----------------------------------------------------------------------------
  --[[                          FILE I/O OPTIONS                           ]]--
  -----------------------------------------------------------------------------
  wrtRestart = 'ON',            -- Enable restart file output, 'ON' or 'OFF'
  wrtVolumeSolution = 'ON',     -- Enable volume solution output
  outputFormat = 'Tecplot',     -- Volume solution format, 'Tecplot' only
  wrt1DSlice = 'ON',            -- Enable CSV slices at centerlines
  wrtParticleEvolution = 'ON', -- Enable tracking of a single particle
  particleEvolutionIndex = 0,   -- Index of particle to be tracked
  outputEveryTimeSteps  = 10000, -- Iterations between writing solutions
  restartEveryTimeSteps = 10000, -- Iterations between writing restarts
  headerFrequency       = 20,   -- Iterations between console output headers
  outputDirectory = '../forced_turbulence/' -- Relative to the ebb root dir
  
}
