-- This is a Lua config file for the pressure-based Ebb cavity code.

return {

-- Physical length of the domain in the X- and Y-directions
lx = 1.0,  
ly = 1.0,

-- Number of interior cells in the X- and Y-directions
nx = 129,
ny = 129,

-- Grid type: either 'uniform' or 'chebyshev' spacing
gridType = 'uniform', 

-- Maximum number of inner iterations per physical time step
nsteps = 100,

-- Density, Viscosity, and Prandtl number
densit = 1.0, 
visc = 0.001,
prm = 0.71,

-- Gravity source in the X- and Y-directions, beta parameter
gravx = 0.0,
gravy = 0.0,
beta = 25.0,

-- Physical time step
dt = 0.1, 

-- Final physical time (simulation terminates at this time)
finaltime = 200.0,

-- Convergence tol. for each outer iteration (residual sum for all equations)
converged = 1.0e-8, 

-- linear solver, either 'jacobi' or 'gauss_seidel'
linearSolver = 'gauss_seidel', 

-- Output location relative to the liszt-ebb home directory
outputDirectory = '../cavityOutput/',

-- Output frequency (number of iterations between writing files)
itimeprint = 10,

-- Reference temperature (initialization and in momentum source)
Tref = 400.0,

-- Flag for whether the energy equation is integrated: 'yes' or 'no'
solveEnergy = 'no',

-- BC Options: 'adiabatic' or 'isothermal' and specified velocities and
-- temperatures. Note that the velocities can only be set parallel to the wall.
-- Velocities are always used, but the temperatures are ignored for adiabatic.
northBC   = 'adiabatic',
northVel  = 1.0,
northTemp = 0.0,

southBC   = 'adiabatic',
southVel  = 0.0,
southTemp = 0.0,

eastBC    = 'adiabatic',
eastVel   = 0.0,
eastTemp  = 0.0,

westBC    = 'adiabatic',
westVel   = 0.0,
westTemp  = 0.0,

}
