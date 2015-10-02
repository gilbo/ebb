import "ebb"

-- Load the grid library for structured grids.

local Grid  = require 'ebb.domains.grid'

-- We can import the C math library through terra.

local cmath = terralib.includecstring [[
#include <math.h>
]]

-- Load the pathname library, which just provides a couple of
-- convenience functions for manipulating filesystem paths.

local PN = require 'ebb.lib.pathname'


--------------------------------------------------------------------------------
--[[                         LOAD THE CONFIG FILE                           ]]--
--------------------------------------------------------------------------------

-- This location is hard-coded at the moment. The idea is that you will
-- have multiple config files available in other locations, and copy them
-- to this location with the name params.lua before running.

local filename = './devapps/cavity/params.lua'
local config = loadfile(filename)()

-- Immediately check that the output directory exists. Throw an error if not.

local Pathname  = PN.Pathname
local outputdir = Pathname.new(config.outputDirectory)
if not outputdir:exists() then
  outputdir:mkdir()
  print("\nWARNING: The requested output directory does not exist: "..
        config.outputDirectory .. ", \n"..
        "so it has been created. Please check your intended behavior.\n")
end

--------------------------------------------------------------------------------
--[[                            CONTROL VARIABLES                           ]]--
--------------------------------------------------------------------------------

local diverged = 1.0E6
local pi = 2.0*L.acos(0)
local gamt  = 0.0  -- implicit time-integration scheme
local gds   = 0.9  -- convective flux discretization
local time  = 0.0
local itime = 0
local itimeprint = config.itimeprint
local grav = cmath.sqrt(config.gravx*config.gravx+config.gravy*config.gravy)

local nsw_mom = 5    -- max linear solver iterations for the momentum eqns.
local tol_mom = 0.2  -- tolerance for the momentum linear solve
local urf_mom = 0.8  -- under-relaxation factor for momentum

local nsw_p = 200    -- max linear solver iterations for the pressure eqn.
local tol_p = 0.02   -- tolerance for the pressure linear solve
local urf_p = 0.4    -- under-relaxation factor for pressure

local nsw_temp = 2   -- max linear solver iterations for the energy eqns.
local tol_temp = 0.2 -- tolerance for the energy linear solve
local urf_temp = 0.9 -- under-relaxation factor for temperature

-- Set up an enumerated type-like variable for defining the linear solver.

Jacobi       = L.Global(L.int, 0)
Gauss_Seidel = L.Global(L.int, 1)
local linear_solver = Gauss_Seidel

-- Set up an enumerated type-like variable for defining the grid spacing.

Uniform   = L.Global(L.int, 0)
Chebyshev = L.Global(L.int, 1)
local grid_type = Chebyshev

-- Create some booleans to control isothermal wall BCs

isothermal_westBC  = L.Global(L.bool, false)
isothermal_eastBC  = L.Global(L.bool, false)
isothermal_southBC = L.Global(L.bool, false)
isothermal_northBC = L.Global(L.bool, false)

--------------------------------------------------------------------------------
--[[                   CONSOLE OUTPUT AFTER PREPROCESSING                   ]]--
--------------------------------------------------------------------------------

-- Check the boundary conditions in the config file for consistency.

if config.northBC == "adiabatic" then
  config.northTemp = -1.0
elseif config.northBC  == "isothermal" then
  isothermal_northBC:set(true)
else
  error("North BC improperly specified. Must be 'adiabatic' or 'isothermal'.")
end
if config.southBC == "adiabatic" then
  config.southTemp  = -1.0
elseif config.southBC  == "isothermal" then
  isothermal_southBC:set(true)
else
  error("South BC improperly specified. Must be 'adiabatic' or 'isothermal'.")
end
if config.eastBC == "adiabatic" then
  config.eastTemp  = -1.0
elseif config.eastBC  == "isothermal" then
  isothermal_eastBC:set(true)
else
  error("East BC improperly specified. Must be 'adiabatic' or 'isothermal'.")
end
if config.westBC == "adiabatic" then
  config.westTemp  = -1.0
elseif config.westBC  == "isothermal" then
  isothermal_westBC:set(true)
else
  error("West BC improperly specified. Must be 'adiabatic' or 'isothermal'.")
end

-- Get the grid type from the config file.

if config.gridType == "uniform" then
  grid_type = Uniform
elseif config.gridType  == "chebyshev" then
  grid_type = Chebyshev
else
  error("Grid type improperly specified. Must be 'uniform' or 'chebyshev'.")
end

-- Get the linear solver type from the config file.

if config.linearSolver == "jacobi" then
  linear_solver = Jacobi
elseif config.linearSolver  == "gauss_seidel" then
  linear_solver = Gauss_Seidel
else
  error("Solver type improperly specified. Must be 'jacobi' or 'gauss_seidel'.")
end

-- Check for whether we need to integrate the Energy equation.

local solve_energy = true
if config.solveEnergy == "yes" then
  solve_energy = true
elseif config.solveEnergy  == "no" then
  solve_energy = false
else
  error("Solve Energy eqn. flag improperly specified. Must be 'yes' or 'no'.")
end

print("\n\n=== INPUT FILE ==================================================")
io.stdout:write(" domain size, number of inner steps >> ",
                string.format("%1.3f",config.lx), ", ",
                string.format("%1.3f",config.ly), ", ",
                string.format("%1.3f",config.finaltime),"\n")
io.stdout:write(" density, viscosity, Pr  >> ",
                string.format("%1.3f",config.densit), ", ",
                string.format("%1.3f",config.visc), ", ",
                string.format("%1.3f",config.prm),"\n")
io.stdout:write(" buoyancy force >> ",
                string.format("%1.3f",config.gravx), ", ",
                string.format("%1.3f",config.gravy), ", ",
                string.format("%1.3f",config.beta),"\n")
io.stdout:write(" reference Temp. >> ",string.format("%1.5f",config.Tref),"\n")
io.stdout:write(" north wall BC >> ",
                string.format("%s",config.northBC), ", ",
                string.format("%1.3f",config.northVel), ", ",
                string.format("%1.3f",config.northTemp),"\n")
io.stdout:write(" east wall BC >> ",
                string.format("%s",config.eastBC), ", ",
                string.format("%1.3f",config.eastVel), ", ",
                string.format("%1.3f",config.eastTemp),"\n")
io.stdout:write(" south wall BC >> ",
                string.format("%s",config.southBC), ", ",
                string.format("%1.3f",config.southVel), ", ",
                string.format("%1.3f",config.southTemp),"\n")
io.stdout:write(" west wall BC >> ",
                string.format("%s",config.westBC), ", ",
                string.format("%1.3f",config.westVel), ", ",
                string.format("%1.3f",config.westTemp),"\n")
io.stdout:write(" discretization >> ",
                string.format("%d",   config.nx), ", ",
                string.format("%d",   config.ny), ", ",
                string.format("%1.3f",config.dt), ", ",
                string.format("%d",   config.nsteps), ", ",
                string.format("%1.3f",config.converged),"\n")
print("=====================================================================")
io.stdout:write(" Prandtl >> ",string.format("%1.5f",config.prm),"\n")
print("=====================================================================")


--------------------------------------------------------------------------------
--[[                          GRID INITIALIZATION                           ]]--
--------------------------------------------------------------------------------

-- Define the boundary size. We have a single halo layer around the 2D domain.

local xBnum = 1
local yBnum = 1

-- Compute the width of the boundary cells. Note that we have uniform grids
-- at the moment, so the width of the halos matches the interior.

local xBw = config.lx/config.nx * xBnum
local yBw = config.ly/config.ny * yBnum

-- Set the origin of the domain using the input length. Note that we place
-- the origin such that (0.0,0.0) is in the ceneter of the domain.

local origin = {-config.lx/2.0, -config.ly/2.0}

-- For now, we disable periodic boundaries, as all four boundaries are walls.

local PERIODIC = false
local period = {false,false}
if PERIODIC then
  period = {true,true}
end

-- Create the Ebb grid object from the above inputs. Once the grid is
-- initialized, we will have a number of relations available to us,
-- including cells, dual cells, and vertices.

local grid = Grid.NewGrid2d {
  size           = {config.nx + 2*xBnum, config.ny + 2*yBnum},
  origin         = {origin[1] - xBnum * xBw, origin[2] - yBnum * yBw},
  width          = {config.lx + 2*xBw, config.ly + 2*yBw},
  boundary_depth = {xBnum, yBnum},
  periodic_boundary = period
}

-- Define origin as Ebb globals for use in kernels.

local grid_originX = L.Global(L.double, grid:xOrigin())
local grid_originY = L.Global(L.double, grid:yOrigin())

-- Create a field for the cell vertices for uniform/chebyshev. This is
-- the first instance of creating a field on the grid.cells relation.
-- We also define our first Ebb kernel (to be called below) that will
-- populate our fields for the x and y coordinates for the vertices.
-- Note that we are using the cell id here, which may be unsafe.

grid.cells:NewField('x', L.double):Load(0.0)
grid.cells:NewField('y', L.double):Load(0.0)
local ebb init_vertices (c : grid.cells)
  var xid = L.double(L.xid(c))
  var yid = L.double(L.yid(c))
  var nx  = L.double(config.nx)
  var ny  = L.double(config.ny)
  if (grid_type == Uniform) then
    c.x = -config.lx/2.0+config.lx*xid/nx
    c.y = -config.ly/2.0+config.ly*yid/ny
  elseif (grid_type == Chebyshev) then
    c.x = -(config.lx/2.0)*L.cos(pi/(nx)*(xid))
    c.y = -(config.ly/2.0)*L.cos(pi/(ny)*(yid))
  end
end

-- Create a field for the center coords based on the vertex locations.
-- First, we initialize the interior cell centers, then have a separate
-- kernel to initialize the halo cell centers.

grid.cells:NewField('xc',L.double):Load(0.0)
grid.cells:NewField('yc',L.double):Load(0.0)
grid.cells:NewField('xc_temp',L.double):Load(0.0)
grid.cells:NewField('yc_temp',L.double):Load(0.0)
local ebb init_interior_centroids (c : grid.cells)
  if c.in_interior then
    c.xc = 0.5*(c.x+c(-1,0).x)
    c.yc = 0.5*(c.y+c(0,-1).y)
  end
end
local ebb init_boundary_centroids (c : grid.cells)
  if c.xneg_depth > 0 then
    c.xc_temp = c.x - (c(1,0).xc - c.x)
    c.yc_temp = c(1,0).yc
  end
  if c.xpos_depth > 0 then
    c.xc_temp = c(-1,0).x + (c(-1,0).x - c(-1,0).xc)
    c.yc_temp = c(-1,0).yc
  end
  if c.yneg_depth > 0 then
    c.yc_temp = c.y - (c(0,1).yc - c.y)
    c.xc_temp = c(0,1).xc
  end
  if c.ypos_depth > 0 then
    c.yc_temp = c(0,-1).y + (c(0,-1).y - c(0,-1).yc)
    c.xc_temp = c(0,-1).xc
  end
end
local ebb update_boundary_centroids (c : grid.cells)
  if c.xneg_depth > 0 or c.xpos_depth > 0 or
     c.yneg_depth > 0 or c.ypos_depth > 0 then
    c.xc = c.xc_temp
    c.yc = c.yc_temp
  end
  if c.xneg_depth > 0 and c.ypos_depth > 0 then -- top left corner
    c.xc = c(0,-1).xc_temp
    c.yc = c(1, 0).yc_temp
    end
  if c.xneg_depth > 0 and c.yneg_depth > 0 then -- bottom left corner
    c.xc = c(0,1).xc_temp
    c.yc = c(1,0).yc_temp
  end
  if c.xpos_depth > 0 and c.ypos_depth > 0 then -- top right corner
    c.xc = c(0,-1).xc_temp
    c.yc = c(-1,0).yc_temp
  end
  if c.xpos_depth > 0 and c.yneg_depth > 0 then -- bottom right corner
    c.xc = c( 0,1).xc_temp
    c.yc = c(-1,0).yc_temp
  end
end

-- Create a field to carry the stretching factor (weights).

grid.cells:NewField('fx',L.double):Load(0.0)
grid.cells:NewField('fy',L.double):Load(0.0)
local ebb init_weights (c : grid.cells)
  c.fx  = (c.x - c.xc)/(c(1,0).xc - c.xc)
  c.fy  = (c.y - c.yc)/(c(0,1).yc - c.yc)
end

-- Create a field to flag the halo layer so it is not written in the output.
-- This is purely for visualization purposes.

grid.cells:NewField('halo_layer', L.int):Load(1)
local ebb init_halo_layer(c : grid.cells)
  if c.in_interior then
    c.halo_layer = 0
  end
end


--------------------------------------------------------------------------------
--[[                           FIELD INITIALIZATION                         ]]--
--------------------------------------------------------------------------------

-- Primitive variables (velocities, temperature, pressure).
-- '0' and '00' are time levels n and n-1 for dual time stepping.
-- The scratch spaces are used to store residuals/matrix entries on the 
-- left and right of cell faces in a second kernel after their calculation,
-- or often to hold temporary values for updating halos.

-- X-velocity

grid.cells:NewField('u',      L.double):Load(0.0)
grid.cells:NewField('u0',     L.double):Load(0.0)
grid.cells:NewField('u00',    L.double):Load(0.0)
grid.cells:NewField('u_temp', L.double):Load(0.0) -- scratch space

-- Y-velocity

grid.cells:NewField('v',      L.double):Load(0.0)
grid.cells:NewField('v0',     L.double):Load(0.0)
grid.cells:NewField('v00',    L.double):Load(0.0)
grid.cells:NewField('v_temp', L.double):Load(0.0) -- scratch space

-- Temperature

grid.cells:NewField('T',      L.double):Load(0.0)
grid.cells:NewField('T0',     L.double):Load(0.0)
grid.cells:NewField('T00',    L.double):Load(0.0)
grid.cells:NewField('T_temp', L.double):Load(0.0) -- scratch space

-- Pressure

grid.cells:NewField('p',      L.double):Load(0.0)
grid.cells:NewField('p_temp', L.double):Load(0.0) -- scratch space

-- Delta Pressure (used in pressure poisson solve)

grid.cells:NewField('pp',      L.double):Load(0.0)
grid.cells:NewField('pp_temp', L.double):Load(0.0) -- scratch space

-- Implicit data arrays for linear solves stored as matrix diagonals (a*).
-- First, main diagonal contribution used in multiple equation solves.

grid.cells:NewField('ap',      L.double):Load(0.0)
grid.cells:NewField('ap_temp', L.double):Load(0.0)

-- Main diagonal entries for momentum

grid.cells:NewField('apu', L.double):Load(0.0)
grid.cells:NewField('apv', L.double):Load(0.0)

-- Entries for surrounding cells stored as east, west, north, south.

grid.cells:NewField('ae', L.double):Load(0.0)
grid.cells:NewField('aw', L.double):Load(0.0)
grid.cells:NewField('an', L.double):Load(0.0)
grid.cells:NewField('as', L.double):Load(0.0)
grid.cells:NewField('al', L.double):Load(0.0) -- scratch space
grid.cells:NewField('ar', L.double):Load(0.0) -- scratch space

-- Residual data arrays (s*), i.e., right-hand sides for linear solves.
-- The fields 'su' and 'sv' will be used for the x- and y-momentum
-- equations, and the field 'su' will be reused as the r.h.s for the
-- pressure solve and energy equation.

grid.cells:NewField('su',  L.double):Load(0.0)
grid.cells:NewField('sv',  L.double):Load(0.0)
grid.cells:NewField('sul', L.double):Load(0.0) -- scratch space
grid.cells:NewField('svl', L.double):Load(0.0) -- scratch space
grid.cells:NewField('sur', L.double):Load(0.0) -- scratch space
grid.cells:NewField('svr', L.double):Load(0.0) -- scratch space

-- Additional data arrays (mass fluxes and geometry).

grid.cells:NewField('f1',     L.double):Load(0.0)
grid.cells:NewField('f2',     L.double):Load(0.0)
grid.cells:NewField('f_temp', L.double):Load(0.0) -- scratch space
grid.cells:NewField('dpx',    L.double):Load(0.0)
grid.cells:NewField('dpy',    L.double):Load(0.0)

-- Several auxiliary Ebb globals that we need for tracking quantities,
-- such as heat flux through walls. By initializing them
-- as Ebb globals, they will be accessible within Ebb kernels.

qwall_n = L.Global(L.double,0.0)
qwall_s = L.Global(L.double,0.0)
qwall_w = L.Global(L.double,0.0)
qwall_e = L.Global(L.double,0.0)

-- Residuals after the solve of each equation (inner iteration).

res  = L.Global(L.double,0.0)
resu = L.Global(L.double,0.0)
resv = L.Global(L.double,0.0)
resp = L.Global(L.double,0.0)
resT = L.Global(L.double,0.0)

-- Sum of mass in and out of domain.

sum = L.Global(L.double,0.0)

-- Variable used to store a reference pressure for correction.

ppo = L.Global(L.double,0.0)

-- Field for the velocity magnitude in each cell and a global for the max.

grid.cells:NewField('vmag', L.double):Load(0.0)
vmagmax = L.Global(L.double,-1.0)

-- Global variable to store the currentime for access within kernels.

currenttime = L.Global(L.double,0.0)

-- Epsilon value for use throughout the code

eps = L.Global(L.double,0.0)


--------------------------------------------------------------------------------
--[[                    INITIALIZATION & UPDATE KERNELS                     ]]--
--------------------------------------------------------------------------------

-- Set pressure and temp. All other fields have been initialized to zero.

local ebb set_initial_conditions(c : grid.cells)
  c.f1  = L.double(0.0)
  c.f2  = L.double(0.0)
  c.u   = L.double(0.0)
  c.u0  = L.double(0.0)
  c.u00 = L.double(0.0)
  c.v   = L.double(0.0)
  c.v0  = L.double(0.0)
  c.v00 = L.double(0.0)
  c.p   = L.double(1.0)
  c.pp  = L.double(1.0)
  c.T   = config.Tref
  c.T0  = config.Tref
  c.T00 = config.Tref
end

-- Update the solution for dual time stepping in all cells by pushing
-- the solution back to time levels n-1 and n.

local ebb update_level00(c: grid.cells)
  c.T00 = c.T0
  c.u00 = c.u0
  c.v00 = c.v0
end
local ebb update_level0(c: grid.cells)
  c.T0 = c.T
  c.u0 = c.u
  c.v0 = c.v
end
local function solution_update(cells)
  cells:foreach(update_level00)
  cells:foreach(update_level0)
end

-- A simple print kernel that is useful for debugging.

local ebb output_p (c : grid.cells)
  L.print(L.xid(c), L.yid(c), c.x, c.y, c.xc, c.yc, c.fx,
          c.fy, c.pp, c.su, c.sv, c.ap, c.ae, c.apu, c.apv,
          c.aw, c.an, c.as, c.f1, c.f2)
end


--------------------------------------------------------------------------------
--[[                            LINEAR SOLVERS                              ]]--
--------------------------------------------------------------------------------

-- Jacobi iterative linear solver. Any solution vector field (x) and r.h.s.
-- (b) can be used as input. The number of sweeps and convergence tolerence
-- are defined for each system at the top of the file.

local ebb jacobi_update (c, x, x_temp)
  c[x] = c[x_temp]
end
local ebb jacobi_step (c, x, x_temp, b)
  var sum = c.aw*c(-1,0)[x] + c.ae*c(1,0)[x] + c.as*c(0,-1)[x] + c.an*c(0,1)[x]
  res += cmath.fabs(c[b] - sum - c.ap*c[x])
  c[x_temp] = (c[b] - sum) / (c.ap)
end
local function jacobi_solve (cells, x, x_temp, b, nsw, tol, resx)
  local res0 = 0.0
  for i=1,nsw do
    res:set(0.0)
    cells.interior:foreach(jacobi_step, x, x_temp, b)
    cells.interior:foreach(jacobi_update, x, x_temp)
    if i == 1 then res0 = res:get() end
    if res0 == 0.0 or res:get()/(res0+eps:get()) < tol then break end
  end
  resx:set(res:get())
end

-- Red-black Gauss-Seidel iterative linear solver. Any solution vector 
-- field (x) and r.h.s. (b) can be used as input. The number of sweeps 
-- and convergence tolerence are defined for each system at the top of 
-- the file. Due to the data access restrictions in Ebb, the red-black 
-- coloring is required to break the dependency. In that way, we can 
-- perform the step and update in separate passes.

local ebb red_update (c, x, x_temp)
  if ((0+L.xid(c))%2 == 0 and (1+L.yid(c))%2 == 0) or
     ((1+L.xid(c))%2 == 0 and (0+L.yid(c))%2 == 0) then
    c[x] = c[x_temp]
  end
end
local ebb red_step (c, x, x_temp, b)
  if ((0+L.xid(c))%2 == 0 and (1+L.yid(c))%2 == 0) or
     ((1+L.xid(c))%2 == 0 and (0+L.yid(c))%2 == 0) then
    var sum = c.aw*c(-1,0)[x]+c.ae*c(1,0)[x]+c.as*c(0,-1)[x]+c.an*c(0,1)[x]
    res += cmath.fabs(c[b] - sum - c.ap*c[x])
    c[x_temp] = (c[b] - sum) / (c.ap)
  end
end
local ebb black_update (c, x, x_temp)
  if ((0+L.xid(c))%2 == 0 and (0+L.yid(c))%2 == 0) or
     ((1+L.xid(c))%2 == 0 and (1+L.yid(c))%2 == 0) then
    c[x] = c[x_temp]
  end
end
local ebb black_step (c, x, x_temp, b)
  if ((0+L.xid(c))%2 == 0 and (0+L.yid(c))%2 == 0) or
     ((1+L.xid(c))%2 == 0 and (1+L.yid(c))%2 == 0) then
    var sum = c.aw*c(-1,0)[x]+c.ae*c(1,0)[x]+c.as*c(0,-1)[x]+c.an*c(0,1)[x]
    res += cmath.fabs(c[b] - sum - c.ap*c[x])
    c[x_temp] = (c[b] - sum) / (c.ap)
  end
end
local function gauss_seidel_solve (cells, x, x_temp, b, nsw, tol, resx)
  local res0 = 0.0
  for i=1,nsw do
    res:set(0.0)
    cells.interior:foreach(red_step, x, x_temp, b)
    cells.interior:foreach(red_update, x, x_temp)
    cells.interior:foreach(black_step, x, x_temp, b)
    cells.interior:foreach(black_update, x, x_temp)
    if i == 1 then res0 = res:get() end
    if res0 == 0.0 or res:get()/(res0+eps:get()) < tol then break end
  end
  resx:set(res:get())
end


--------------------------------------------------------------------------------
--[[                                MOMENTUM                                ]]--
--------------------------------------------------------------------------------

-- These fields are zeroed at the start of each momentum inner iteration.

local ebb init_momentum_fields (c : grid.cells)
  c.su  = L.double(0.0)
  c.sv  = L.double(0.0)
  c.apu = L.double(0.0)
  c.apv = L.double(0.0)
end

-- Update the pressure in the halo cells by extrapolation. This kernel is
-- called at the beginning of the momentum iteration, typically after a 
-- previous iteration has computed and applied the pressure correction to
-- the interior nodes. It is also called after the pressure poisson solve
-- to extrapolate the delta pressure to the boundaries before correction.
-- Therefore, either the 'p' or the 'pp' field will be input as 'phi'.
-- This is also the first time that we see the typical 'update' and
-- 'copy' kernels using scratch space to obey Ebb's data access rules.

local ebb phi_halo_update (c, phi, phi_temp)
  if c.xneg_depth > 0 then
    c[phi_temp] = c(1,0)[phi]  + ((c(1,0)[phi] - c(2,0)[phi])*
                                  ((c(1,0).xc-c.xc)/(c(2,0).xc-c(1,0).xc)))
  elseif c.xpos_depth > 0 then
    c[phi_temp] = c(-1,0)[phi] + ((c(-1,0)[phi] - c(-2,0)[phi])*
                                  ((c.xc-c(-1,0).xc)/(c(-1,0).xc-c(-2,0).xc)))
  elseif c.yneg_depth > 0 then
    c[phi_temp] = c(0,1)[phi]  + ((c(0,1)[phi] - c(0,2)[phi])*
                                  ((c(0,1).yc-c.yc)/(c(0,2).yc-c(0,1).yc)))
  elseif c.ypos_depth > 0 then
    c[phi_temp] = c(0,-1)[phi] + ((c(0,-1)[phi] - c(0,-2)[phi])*
                                  ((c.yc-c(0,-1).yc)/(c(0,-1).yc-c(0,-2).yc)))
  end
end
local ebb phi_copy_update (c, phi, phi_temp)
  c[phi] = c[phi_temp]
end
local function phibc (cells, phi, phi_temp)
  cells.boundary:foreach(phi_halo_update, phi, phi_temp)
  cells.boundary:foreach(phi_copy_update, phi, phi_temp)
end

-- Apply boundary conditions for the momentum equations. Here, we
-- reflect the velocities parallel to the wall and simply extrapolate
-- the velocities perpendicular to the wall. We also impost any
-- wall velocity that is specified in the config file.
-- West / East boundary:   wall, shear force in y-dir, du/dx=0
-- South / North boundary: wall, shear force in x-dir, dv/dy=0

local ebb uv_halo_update (c : grid.cells)
  if c.xneg_depth > 0 then     -- west boundary
    c.u_temp = -c( 1,0).u
    --c.v_temp = -c( 1,0).v + config.westVel
    c.v_temp = -c( 1,0).v + 2.0*config.westVel
  elseif c.xpos_depth > 0 then -- east boundary
    c.u_temp = -c(-1,0).u
    --c.v_temp = -c(-1,0).v + config.eastVel
    c.v_temp = -c(-1,0).v + 2.0*config.eastVel
  elseif c.yneg_depth > 0 then -- south boundary
    --c.u_temp = -c(0, 1).u + config.southVel
    c.u_temp = -c(0, 1).u + 2.0*config.southVel
    c.v_temp = -c(0, 1).v
  elseif c.ypos_depth > 0 then -- north boundary
    --c.u_temp = -c(0,-1).u + config.northVel
    c.u_temp = -c(0,-1).u + 2.0*config.northVel
    c.v_temp = -c(0,-1).v
  end
end
local ebb uv_copy_update (c : grid.cells) -- copy from temporary field
  c.u = c.u_temp
  c.v = c.v_temp
end

-- This is a kernel that directly adds a contribution to the fluxes
-- and Jacobian for the momentum equations (as in the baseline code).
-- In the future, we hope to remove the need for this by extending 
-- the main flux loops to include the halos.

local ebb uv_weak_bc (c : grid.cells)
  if c.in_interior then
    var d = L.double(0.0)
    if c(0,-1).yneg_depth > 0 then -- south boundary
      d = config.visc*(c.x - c(-1,0).x)/(c.yc - c(0,-1).yc)
      c.su  += d*(c(0,-1).u)
      c.apu += d
    end
    if c(0,1).ypos_depth > 0 then  -- north boundary
      d = config.visc*(c.x - c(-1,0).x)/(c(0,1).yc - c.yc)
      c.su  += d*(c(0,1).u)
      c.apu += d
    end
    if c(-1,0).xneg_depth > 0 then -- west boundary
      d = config.visc*(c.y - c(0,-1).y)/(c.xc - c(-1,0).xc)
      c.sv  += d*(c(-1,0).v)
      c.apv += d
    end
    if c(1,0).xpos_depth > 0 then  -- east boundary
      d = config.visc*(c.y - c(0,-1).y)/(c(1,0).xc - c.xc)
      c.sv  += d*(c(1,0).v)
      c.apv += d
    end
  end
end

-- Apply the boundary conditions to the momentum equations.

local function uvbc (cells)
  cells.boundary:foreach(uv_halo_update)
  cells.boundary:foreach(uv_copy_update)
  cells.interior:foreach(uv_weak_bc)
end

-- Compute the momentum flux in the x-direction. Here, we compute the flux
-- through the east face of each control volume and store the results to be
-- scattered to the current cell on the left of the face (sul, svl, al) and
-- to the eastern cell (sur, svr, ar). These fluxes are then scattered in a
-- second kernel over interior control volumes and stored in the proper data
-- arrays that will be used for the linear solve.

local ebb momentum_flux_x (c : grid.cells)
  if c.in_interior and c(1,0).xpos_depth ~= 1 then
    
    var dxpe = c(1,0).xc - c.xc
    var s    = c.y - c(0,-1).y
    var d    = config.visc*s/dxpe
    
    var fxe = c.fx
    var fxp = 1.0 - fxe

    var ce = cmath.fmin(c.f1,0.0)
    var cp = cmath.fmax(c.f1,0.0)
    
    var fuuds = cp*c.u + ce*c(1,0).u
    var fvuds = cp*c.v + ce*c(1,0).v
    
    var fucds = c.f1 * (c(1,0).u*fxe + c.u*fxp)
    var fvcds = c.f1 * (c(1,0).v*fxe + c.v*fxp)
    
    -- The east fluxes for the current cell
    
    c.al  = ce-d
    c.sul = gds*(fuuds-fucds)
    c.svl = gds*(fvuds-fvcds)
    
    -- The west fluxes for c(1,0) (negative here, so add below)
    
    c.ar  = -cp-d
    c.sur = -gds*(fuuds-fucds)
    c.svr = -gds*(fvuds-fvcds)
    
  end
end
local ebb add_momentum_flux_x (c : grid.cells)
  if c.in_interior then
    if c(1, 0).xpos_depth ~= 1 then
      c.su += c.sul
      c.sv += c.svl
      c.ae  = c.al
    end
    if c(-1,0).xneg_depth ~= 1 then
      c.su += c(-1,0).sur
      c.sv += c(-1,0).svr
      c.aw  = c(-1,0).ar
    end
  end
end

-- Compute the momentum flux in the y-direction. Same for the x-direction
-- above, except now we treat the northern face.

local ebb momentum_flux_y (c : grid.cells)
  if c.in_interior and c(0,1).ypos_depth ~= 1 then
    
    var dypn = c(0,1).yc - c.yc
    var s    = c.x - c(-1,0).x
    var d    = config.visc*s/dypn
    
    var fyn = c.fy
    var fyp = 1.0 - fyn

    var cn = cmath.fmin(c.f2,0.0)
    var cp = cmath.fmax(c.f2,0.0)
    
    var fuuds = cp*c.u + cn*c(0,1).u
    var fvuds = cp*c.v + cn*c(0,1).v
    
    var fucds = c.f2 * (c(0,1).u*fyn + c.u*fyp)
    var fvcds = c.f2 * (c(0,1).v*fyn + c.v*fyp)

    -- The north fluxes for the current cell.
    
    c.al  = cn-d
    c.sul = gds*(fuuds-fucds)
    c.svl = gds*(fvuds-fvcds)
    
    -- The south fluxes for c(0,1) (negative here, so add below)
    
    c.ar  = -cp-d
    c.sur = -gds*(fuuds-fucds)
    c.svr = -gds*(fvuds-fvcds)
    
  end
end
local ebb add_momentum_flux_y (c : grid.cells)
  if c.in_interior then
    if c(0, 1).ypos_depth ~= 1 then
      c.su += c.sul
      c.sv += c.svl
      c.an  = c.al
    end
    if c(0,-1).yneg_depth ~= 1 then
      c.su += c(0,-1).sur
      c.sv += c(0,-1).svr
      c.as  = c(0,-1).ar
    end
  end
end

-- Implicit flux discretization for the momentum equations. This local
-- function calls the kernels above to compute x & y contributions.

local function uvlhs (cells)
  cells:foreach(momentum_flux_x)     -- compute x fluxes (east/west)
  cells:foreach(add_momentum_flux_x) -- combine and store fluxes for x
  cells:foreach(momentum_flux_y)     -- compute y fluxes (north/south)
  cells:foreach(add_momentum_flux_y) -- combine and store fluxes for y
end

-- Source terms for the momentum equations. This includes the gravity and
-- buoyancy terms.

local ebb uvrhs (c : grid.cells)

  var dx  = c.x - c(-1,0).x
  var dy  = c.y - c(0,-1).y
  var vol = dx*dy

  var pe = c(1,0).p*c.fx + c.p*(1.0-c.fx)
  var pw = c.p*c(-1,0).fx + c(-1,0).p*(1.0-c(-1,0).fx)
  var pn = c(0,1).p*c.fy + c.p*(1.0-c.fy)
  var ps = c.p*c(0,-1).fy + c(0,-1).p*(1.0-c(0,-1).fy)

  var sb = -config.beta*config.densit*vol*(c.T - config.Tref)

  c.dpx = (pe-pw)/dx
  c.dpy = (pn-ps)/dy

  c.su += config.gravx*sb - ((pe-pw)/dx)*vol
  c.sv += config.gravy*sb - ((pn-ps)/dy)*vol

end

-- Additional source terms due to the discretization of the time derivative.
-- If gamt = 0: backward implicit. If gamt = non-zero: 3-level scheme.

local ebb time_discretization_uv (c : grid.cells)

  var dx  = c.x - c(-1,0).x
  var dy  = c.y - c(0,-1).y
  var vol = dx*dy
  var apt = config.densit*vol/config.dt

  c.su += (1.0+gamt)*apt*c.u0 - 0.5*gamt*apt*c.u00
  c.sv += (1.0+gamt)*apt*c.v0 - 0.5*gamt*apt*c.v00

  c.apu += (1.0 + 0.5*gamt)*apt
  c.apv += (1.0 + 0.5*gamt)*apt

end

-- Apply an under-relaxation for the x-momentum equation. Here, we modify
-- the residual (r.h.s.) and Jacobian before smoothing the system.

local ebb relax_update_u (c : grid.cells)
  c.su  += (1.0 - urf_mom)*c.ap*c.u
  c.apu  = 1.0 / c.ap
end
local ebb relax_factor_u (c : grid.cells)
  c.ap  = (c.apu - c.ae - c.aw - c.an - c.as)/urf_mom
end
local function under_relaxation_u (cells)
  cells.interior:foreach(relax_factor_u)
  cells.interior:foreach(relax_update_u)
end

-- Apply an under-relaxation for the y-momentum equation. Here, we modify
-- the residual (r.h.s.) and Jacobian before smoothing the system.

local ebb relax_update_v (c : grid.cells)
  c.sv  += (1.0 - urf_mom)*c.ap*c.v
  c.apv  = 1.0 / c.ap
end
local ebb relax_factor_v (c : grid.cells)
  c.ap = (c.apv - c.ae - c.aw - c.an - c.as)/urf_mom
end
local function under_relaxation_v (cells)
  cells.interior:foreach(relax_factor_v)
  cells.interior:foreach(relax_update_v)
end

-- Relax one inner iteration of the momentum equations implicitly.

local function calcuv(cells)
  
  cells:foreach(init_momentum_fields) -- Zero the r.h.s and main diagonals.
  
  phibc (cells, 'p', 'p_temp') -- Extrapolate the pressure at the boundaries.
  
  uvlhs(cells) -- Computing x & y momentum fluxes (convective and viscous).
  
  cells.interior:foreach(uvrhs) -- Compute momentum source terms on the r.h.s.
  
  cells.interior:foreach(time_discretization_uv) -- Add sources for time discr.
  
  uvbc(cells)  -- Set the velocity boundary condition (reflection in halos).
  
  under_relaxation_u(cells) -- Apply under-relaxation for u before solve.
  
  -- Smooth the linear system to update u.
  if (linear_solver == Jacobi) then
    jacobi_solve(cells, 'u', 'u_temp', 'su', nsw_mom, tol_mom, resu)
  elseif (linear_solver == Gauss_Seidel) then
    gauss_seidel_solve(cells, 'u', 'u_temp', 'su', nsw_mom, tol_mom, resu)
  end
  
  under_relaxation_v(cells) -- Apply under-relaxation for v before solve.
  
  -- Smooth the linear system to update v.
  if (linear_solver == Jacobi) then
    jacobi_solve(cells, 'v', 'v_temp', 'sv', nsw_mom, tol_mom, resv)
  elseif (linear_solver == Gauss_Seidel) then
    gauss_seidel_solve(cells, 'v', 'v_temp', 'sv', nsw_mom, tol_mom, resv)
  end
  
end

--------------------------------------------------------------------------------
--[[                                PRESSURE                                ]]--
--------------------------------------------------------------------------------

-- Compute the pressure flux in the x-direction. Here, we compute the mass
-- flux through the east face of each control volume and matrix entries.
-- We store the results to be scattered to the current cell on the left 
-- of the face (f_temp, ae) and to the eastern cell (f_temp, aw). These 
-- fluxes are then scattered/stored in a second kernel.

local ebb pressure_flux_x (c : grid.cells)
  if c.in_interior and c.xpos_depth ~= 1 then

    var dxpe = c(1,0).xc - c.xc
    var s    = c.y - c(0,-1).y

    var fxe = c.fx
    var fxp = 1.0 - fxe
    
    var vole = dxpe*s
    var d = config.densit*s
    
    var dpxel = 0.5*(c(1,0).dpx + c.dpx)
    var uel = c(1,0).u*fxe + c.u*fxp
    
    var apue = c(1,0).apu*fxe + c.apu*fxp
    
    var dpxe = (c(1,0).p - c.p)/dxpe
    
    var ue = uel - apue*vole*(dpxe-dpxel)
    
    c.f_temp = d*ue
    
    c.al = -d*apue*s
    c.ar = -d*apue*s
    
  end
end
local ebb add_pressure_flux_x (c : grid.cells)
  if c.in_interior then
     if c(1, 0).xpos_depth ~= 1 and c.xneg_depth ~= 1 then c.f1 = c.f_temp end
     if c( 1,0).xpos_depth ~= 1 then c.ae = c.al       end
     if c(-1,0).xneg_depth ~= 1 then c.aw = c(-1,0).ar end
  end
end

-- Compute the pressure flux in the y-direction. Same as for the x-direction
-- above, except now we treat the northern face.

local ebb pressure_flux_y (c : grid.cells)
  if c.in_interior and c.ypos_depth ~= 1 then
    
    var dypn = c(0,1).yc - c.yc
    var s    = c.x - c(-1,0).x
    
    var fyn = c.fy
    var fyp = 1.0 - fyn
    
    var voln = dypn*s
    var d = config.densit*s
    
    var dpynl = 0.5*(c(0,1).dpy + c.dpy)
    var vnl = c(0,1).v*fyn + c.v*fyp
    
    var apvn = c(0,1).apv*fyn + c.apv*fyp

    var dpyn = (c(0,1).p - c.p)/dypn
    
    var vn = vnl - apvn*voln*(dpyn-dpynl)
    
    c.f_temp = d*vn
    
    c.al = -d*apvn*s
    c.ar = -d*apvn*s
    
  end
end
local ebb add_pressure_flux_y (c : grid.cells)
  if c.in_interior then
    if c(0, 1).ypos_depth ~= 1 and c.yneg_depth ~= 1 then c.f2 = c.f_temp end
    if c(0, 1).ypos_depth ~= 1 then c.an = c.al end
    if c(0,-1).yneg_depth ~= 1 then c.as = c(0,-1).ar end
  end
end

-- Set up the linear system for the pressure solve. This local
-- function calls the kernels above to compute x & y contributions.

local function plhs(cells)
  cells:foreach(pressure_flux_x)     -- Compute x fluxes (east/west)
  cells:foreach(add_pressure_flux_x) -- Combine and store fluxes for x
  cells:foreach(pressure_flux_y)     -- Compute y fluxes (north/south)
  cells:foreach(add_pressure_flux_y) -- combine and store fluxes for y
end

-- Initial the pressure system variables and set the delta pressure to zero
-- as our initial guess for the system solve.

local ebb init_pressure_system (c : grid.cells)
  var rhs = c(-1,0).f1 - c.f1 + c(0,-1).f2 - c.f2
  c.su = rhs
  sum += rhs
  c.pp = 0.0
  c.ap = -(c.ae + c.aw + c.an + c.as)
end

-- Solve the pressure poisson equation.

local function calcp(cells)

  plhs(cells) -- Compute the x & y pressure fluxes and matrix entries.
  
  sum:set(0.0) -- Zero out the total mass in/out of all cells.
  
  cells.interior:foreach(init_pressure_system) -- Set up r.h.s./initial guess.
  
  -- Smooth the linear system to solve for pp.
  if (linear_solver == Jacobi) then
    jacobi_solve(cells, 'pp', 'pp_temp', 'su', nsw_p, tol_p, resp)
  elseif (linear_solver == Gauss_Seidel) then
    gauss_seidel_solve(cells, 'pp', 'pp_temp', 'su', nsw_p, tol_p, resp)
  end
  
  phibc (cells, 'pp', 'pp_temp') -- Extrap. the delta pressure to boundaries.

end


--------------------------------------------------------------------------------
--[[                              CORRECTION                                ]]--
--------------------------------------------------------------------------------

-- We are only concerned with the gradient of the pressure, so we grab
-- a reference pressure value from a "random" internal cell.

local ebb set_reference_pp (c : grid.cells)
  if L.xid(c) == 2 and L.yid(c) == 2 then
    ppo += c.pp
  end
end

-- Correct the mass fluxes using the results of the pressure poisson solve.

local ebb correct_mass_fluxes (c : grid.cells)
  if c.in_interior and c(1,0).xpos_depth ~= 1 then
    c.f1 += c.ae*(c(1,0).pp - c.pp)
  end
  if c.in_interior and c(0,1).ypos_depth ~= 1 then
    c.f2 += c.an*(c(0,1).pp - c.pp)
  end
end

-- Correct the cell center velocity and pressure using the results of 
-- the pressure poisson solve.

local ebb correct_vel_p (c : grid.cells)
  if c.in_interior then
    
    var dx  = c.x - c(-1,0).x
    var dy  = c.y - c(0,-1).y

    var ppe = c(1,0).pp*c.fx + c.pp*(1.0 - c.fx)
    var ppw = c.pp*c(-1,0).fx + c(-1,0).pp*(1.0 - c(-1,0).fx)
    var ppn = c(0,1).pp*c.fy + c.pp*(1.0 - c.fy)
    var pps = c.pp*c(0,-1).fy + c(0,-1).pp*(1.0 - c(0,-1).fy)

    c.u += -(ppe-ppw)*dy*c.apu
    c.v += -(ppn-pps)*dx*c.apv
    c.p +=  urf_p*(c.pp-ppo)

  end
end

-- Correct the mass flux, velocity, and pressure fields.

local function correct(cells)
  ppo:set(0.0)                       -- zero the reference pressure
  cells:foreach(set_reference_pp)    -- choose a "random" reference pressure
  cells:foreach(correct_mass_fluxes) -- correct mass fluxes
  cells:foreach(correct_vel_p)       -- correct cell center vel. and pressure
end


--------------------------------------------------------------------------------
--[[                                ENERGY                                  ]]--
--------------------------------------------------------------------------------

-- Zero out the r.h.s. and main Jacobian diagonal for the energy eqn.

local ebb init_temperature_fields (c : grid.cells)
  c.su  = L.double(0.0)
  c.ap  = L.double(0.0)
end

-- Apply boundary conditions for the energy equation.
-- West / East boundary: isothermal wall, non-zero diffusive flux
-- South / North boundary: adiabatic wall, dt/dy=0, zero flux

-- Helper function for updating the temp fields to minimize repeated code.
local ebb temperature_helper (c_bnd, c_int, BCTemp)

  -- Temporary variables for computing new halo state
  var temp_wall   = L.double(0.0)
  var temperature = L.double(0.0)

  -- Compute the temperature for the halo cell (possibly adiabatic/isothermal)
  temp_wall = c_int.T
  if BCTemp > 0.0 then
    temp_wall = BCTemp
  end
  temperature = 2.0*temp_wall - c_int.T

  -- Update the boundary cell temporary field.
  c_bnd.T_temp = temperature

end
local ebb temperature_halo_update (c : grid.cells)
  if c.xneg_depth > 0 then     -- west boundary
    temperature_helper(c, c( 1,0), config.westTemp)
  elseif c.xpos_depth > 0 then -- east boundary
    temperature_helper(c, c(-1,0), config.eastTemp)
  elseif c.yneg_depth > 0 then -- south boundary
    temperature_helper(c, c(0, 1), config.southTemp)
  elseif c.ypos_depth > 0 then -- north boundary
    temperature_helper(c, c(0,-1), config.northTemp)
  end
end
local ebb temperature_copy_update (c : grid.cells)
  c.T = c.T_temp
end

-- This is an update to the r.h.s./Jacobian in the first interior cell
-- along the east and west walls where a non-zero diffusive flux occurs.

local ebb temperature_weak_bc (c : grid.cells)
  var d = L.double(0.0)
  if c(-1,0).xneg_depth > 0 and isothermal_westBC then
    d = config.visc/config.prm*(c.y - c(0,-1).y)/(c.xc - c(-1,0).xc)
    c.su += d*(c(-1,0).T)
    c.ap += d
  end
  if c(1,0).xpos_depth > 0 and isothermal_eastBC then
    d = config.visc/config.prm*(c.y - c(0,-1).y)/(c(1,0).xc - c.xc)
    c.su += d*(c(1,0).T)
    c.ap += d
  end
  if c(0,-1).yneg_depth > 0 and isothermal_southBC then
    d = config.visc/config.prm*(c.x - c(-1,0).x)/(c.yc - c(0,-1).yc)
    c.su += d*(c(0,-1).T)
    c.ap += d
  end
  if c(0,1).ypos_depth > 0 and isothermal_northBC then
    d = config.visc/config.prm*(c.x - c(-1,0).x)/(c(0,1).yc - c.yc)
    c.su += d*(c(0,1).T)
    c.ap += d
  end
end

 -- Apply temperature boundary conditions.

local function tbc (cells)
  cells.boundary:foreach(temperature_halo_update)
  cells.boundary:foreach(temperature_copy_update)
  cells:foreach(temperature_weak_bc)
end

-- Compute the energy flux in the x-direction. Here, we compute the flux
-- through the east face of each control volume and store the results to be
-- scattered to the current cell on the left of the face (sul, al) and
-- to the eastern cell (sur, ar). These fluxes are then scattered in a
-- second kernel over interior control volumes and stored in the proper data
-- arrays that will be used for the linear solve.

local ebb energy_flux_x (c : grid.cells)
  if c.in_interior or c.xpos_depth ~= 1 then
    
    var dxpe = c(1,0).xc - c.xc
    var s    = c.y - c(0,-1).y
    var d    = config.visc/config.prm*s/dxpe
    
    var fxe = c.fx
    var fxp = 1.0 - fxe
    
    var ce = cmath.fmin(c.f1,0.0)
    var cp = cmath.fmax(c.f1,0.0)
    
    var fuds = cp*c.T + ce*c(1,0).T
    
    var fcds = c.f1 * (c(1,0).T*fxe + c.T*fxp)
    
    -- The left fluxes for the current cell.
    
    c.al  = ce-d
    c.sul = gds*(fuds-fcds)
    
    -- The right fluxes for c(1,0) (negative here, so add below).
    
    c.ar  = -cp-d
    c.sur = -gds*(fuds-fcds)
    
  end
end
local ebb add_energy_flux_x (c : grid.cells)
  if c.in_interior then
    if c(1, 0).xpos_depth ~= 1 then
      c.su += c.sul
      c.ae  = c.al
    end
    if c(-1,0).xneg_depth ~= 1 then
      c.su += c(-1,0).sur
      c.aw  = c(-1,0).ar
    end
  end
end

-- Compute the energy flux in the y-direction. Same as for the x-direction
-- above, except now we treat the northern face.

local ebb energy_flux_y (c : grid.cells)
  if c.in_interior or c.ypos_depth ~= 1 then
    
    var dypn = c(0,1).yc - c.yc
    var s    = c.x - c(-1,0).x
    var d = config.visc/config.prm*s/dypn
    
    var fyn = c.fy
    var fyp = 1.0 - fyn
    
    var cn = cmath.fmin(c.f2,0.0)
    var cp = cmath.fmax(c.f2,0.0)
    
    var fuds = cp*c.T + cn*c(0,1).T
    
    var fcds = c.f2 * (c(0,1).T*fyn + c.T*fyp)
    
    -- The left fluxes for the current cell.
    
    c.al  = cn-d
    c.sul = gds*(fuds-fcds)
    
    -- The right fluxes for c(1,0) (negative here, so add below).
    
    c.ar  = -cp-d
    c.sur = -gds*(fuds-fcds)
    
  end
end
local ebb add_energy_flux_y (c : grid.cells)
  if c.in_interior then
    if c(0, 1).ypos_depth ~= 1 then
      c.su += c.sul
      c.an  = c.al
    end
    if c(0,-1).yneg_depth ~= 1 then
      c.su += c(0,-1).sur
      c.as  = c(0,-1).ar
    end
  end
end

-- Implicit flux discretization for the energy equation. This local
-- function calls the kernels above to compute x & y contributions.

local function tlhs (cells)
  cells:foreach(energy_flux_x)     -- east/west fluxes
  cells:foreach(add_energy_flux_x) -- combine and store fluxes for x
  cells:foreach(energy_flux_y)     -- south/north fluxes
  cells:foreach(add_energy_flux_y) -- combine and store fluxes for y
end

-- Additional source terms due to the discretization of the time derivative.
-- If gamt = 0: backward implicit. If gamt = non-zero: 3-level scheme.

local ebb time_discretization_T (c : grid.cells)
  if c.in_interior then
    
    var dx  = c.x - c(-1,0).x
    var dy  = c.y - c(0,-1).y
    var vol = dx*dy
    var apt = config.densit*vol/config.dt
    
    c.su += (1.0+gamt)*apt*c.T0 - 0.5*gamt*apt*c.T00
    c.ap += (1.0 + 0.5*gamt)*apt
    
  end
end

-- Apply an under-relaxation for the energy equation. Here, we modify
-- the residual (r.h.s.) and Jacobian before smoothing the system.

local ebb relax_update_T (c : grid.cells)
  if c.in_interior then
    c.su  += (1.0 - urf_temp)*c.ap_temp*c.T
    c.ap   = c.ap_temp
  end
end
local ebb relax_factor_T (c : grid.cells)
  if c.in_interior then
    c.ap_temp = (c.ap - c.ae - c.aw - c.an - c.as)/urf_temp
  end
end
local function under_relaxation_T (cells)
  cells:foreach(relax_factor_T)
  cells:foreach(relax_update_T)
end

-- Relax one inner iteration of the energy equation implicitly.

local function calct(cells)
  
  cells:foreach(init_temperature_fields) -- Zero the r.h.s and main diagonal.
  
  tlhs(cells) -- Computing x & y energy fluxes (convective and viscous).
  
  cells:foreach(time_discretization_T) -- Add sources for time discretization.
  
  tbc(cells) -- Set the velocity boundary condition (adiabatic/isothermal).
  
  under_relaxation_T(cells) -- Apply under-relaxation for T before solve.
  
  -- Smooth the linear system to update T.
  if (linear_solver == Jacobi) then
    jacobi_solve(cells, 'T', 'T_temp', 'su', nsw_temp, tol_temp, resT)
  elseif (linear_solver == Gauss_Seidel) then
    gauss_seidel_solve(cells, 'T', 'T_temp', 'su', nsw_temp, tol_temp, resT)
  end
  
end


-----------------------------------------------------------------------------
--[[                               OUTPUT                                ]]--
-----------------------------------------------------------------------------

-- Kernels to compute the heat flux through four walls.

local ebb heat_flux_north(c : grid.cells)
  var d = L.double(0.0)
  if c.in_interior and c(0,1).ypos_depth > 0 then
    d = config.visc/config.prm*(c.x - c(-1,0).x)/(c(0,1).yc - c.yc)
    qwall_n += d*(c(0,1).T - c.T)
  end
end
local ebb heat_flux_south(c : grid.cells)
  var d = L.double(0.0)
  if c.in_interior and c(0,-1).yneg_depth > 0 then
    d = config.visc/config.prm*(c.x - c(-1,0).x)/(c.yc - c(0,-1).yc)
    qwall_s += d*(c.T - c(0,-1).T)
  end
end
local ebb heat_flux_east(c : grid.cells)
  var d = L.double(0.0)
  if c.in_interior and c(1,0).xpos_depth > 0 then
    d = config.visc/config.prm*(c.y - c(0,-1).y)/(c(1,0).xc - c.xc)
    qwall_e += d*(c(1,0).T - c.T)
  end
end
local ebb heat_flux_west(c : grid.cells)
  var d = L.double(0.0)
  if c.in_interior and c(-1,0).xneg_depth > 0 then
    d = config.visc/config.prm*(c.y - c(0,-1).y)/(c.xc - c(-1,0).xc)
    qwall_w += d*(c.T - c(-1,0).T)
  end
end

-- Compute and store the maximum velocity magnitude in the interior.

local ebb velocity_magnitude_max(c : grid.cells)
  if c.in_interior then
    c.vmag = c.u * c.u + c.v * c.v
    vmagmax max= c.vmag
  end
end

 -- Compute output quantities of interest and print to the console.

local function output(cells)
  
  -- Initialize our integrated/max values.
  
  qwall_n:set(0.0)
  qwall_s:set(0.0)
  qwall_e:set(0.0)
  qwall_w:set(0.0)
  vmagmax:set(-1.0)
  
  -- Call kernels wall heat flux and velocity magnitude.
  
  cells:foreach(heat_flux_north)
  cells:foreach(heat_flux_south)
  cells:foreach(heat_flux_east)
  cells:foreach(heat_flux_west)
  cells:foreach(velocity_magnitude_max)
  
  -- Print information to console.
  
  print("\n")
  print("=== GLOBAL OUTPUTS ==================================================")
  io.stdout:write(" Total Heat flux through north wall: ",
                  string.format("%f",qwall_n:get()),"\n")
  io.stdout:write(" Total Heat flux through south wall: ",
                  string.format("%f",qwall_s:get()),"\n")
  io.stdout:write(" Total Heat flux through east wall: ",
                  string.format("%f",qwall_e:get()),"\n")
  io.stdout:write(" Total Heat flux through west wall: ",
                  string.format("%f",qwall_w:get()),"\n")
  io.stdout:write(" Maximum velocity magnitude: ",
                  string.format("%f",cmath.sqrt(vmagmax:get())),"\n")
  
end

-- Local function(s) for writing flow solutions in Tecplot ASCII format

local function value_tostring(val)
  if type(val) == 'table' then
    local s = tostring(val[1])
    for i=2,#val do s = s..' '..tostring(val[i]) end
    return s
  end
  return tostring(val)
end

local function dump_with_cell_rind(field_name)
  local s = ''
  local k = 1
  grid.cells:DumpJoint({'halo_layer', field_name},
                       function(ids, halo_layer, field_val)
                       if halo_layer == 0 then
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

local function write_tecplot(grid, timestep)
  
  local outputFileName = config.outputDirectory .. "flow_" ..
  tostring(timestep) .. ".dat"
  
  -- Open the file.
  
  local outputFile = io.output(outputFileName)
  
  -- Get the bool fields for the rind layer so we can avoid writing it.
  
  local halo_cell = grid.cells.halo_layer:DumpToList()
  
  -- Write Tecplot header. Note that we are writing in cell-centered mode,
  -- so we're declaring that variables 3-6 are given at the centers while
  -- the X and Y coordinates are given at the vertices.
  
  io.write('TITLE = "Data"\n')
  io.write('VARIABLES = "X", "Y", "U", "V", "T", "P"\n')
  io.write('ZONE STRANDID=', timestep+1, ' SOLUTIONTIME=',
           timestep*config.dt, ' I=', config.nx+1, ' J=', config.ny+1,
           ' DATAPACKING=BLOCK VARLOCATION=([3-6]=CELLCENTERED)\n')
   local s = ''
   local k = 0 -- Counter to remove extra white space from file (hack)
   
   -- Here, we will recompute the coordinates just for output.
   -- This is being done as a workaround for stretched grids.
   
   local xField = grid.cells.x:DumpToList()
   local yField = grid.cells.y:DumpToList()
   local xCoord = {}
   local yCoord = {}
   local iVertex = 1
   for i=1,config.nx+1 do
     for j=1,config.ny+1 do
       xCoord[iVertex] = xField[i][j]
       yCoord[iVertex] = yField[i][j]
       iVertex = iVertex+1
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
   
   -- Now write density, velocity, temperature, and pressure.
   
   dump_with_cell_rind('u')
   dump_with_cell_rind('v')
   dump_with_cell_rind('T')
   dump_with_cell_rind('p')
   
   -- Close the Tecplot file
   
   io.close()
 
end

local function write_csv_x0 (grid, field, filename)

  -- Open file
  local outputFile = io.output(config.outputDirectory .. filename)

  -- CSV header
  io.write('y, ' .. field .. '\n')

  -- Check for the vertical center of the domain and write the field.
  grid.cells:DumpJoint({ 'xc', 'yc', field },
                       function(ids, xc, yc, field)
                       local s = ''
                       if    (cmath.fabs(xc) <= 1e-4) then
                       s = tostring(yc) .. ', ' .. tostring(field) .. '\n'
                       io.write(s)
                       end
                       end)

  -- Close the file
  io.close()

end

local function write_csv_y0 (grid, field, filename)
  
  -- Open file
  local outputFile = io.output(config.outputDirectory .. filename)
  
  -- CSV header
  io.write('x, ' .. field .. '\n')
  
  -- Check for the vertical center of the domain and write the field.
  grid.cells:DumpJoint({ 'xc', 'yc', field },
                       function(ids, xc, yc, field)
                       local s = ''
                       if    (cmath.fabs(yc) <= 1e-4) then
                       s = tostring(xc) .. ', ' .. tostring(field) .. '\n'
                       io.write(s)
                       end
                       end)
                       
                       -- Close the file
                       io.close()
                       
end

--------------------------------------------------------------------------------
--[[                            MAIN EXECUTION                              ]]--
--------------------------------------------------------------------------------

-- Initialize grid related variables and set the initial conditions.

grid.cells:foreach(init_vertices)
grid.cells:foreach(init_interior_centroids)
grid.cells:foreach(init_boundary_centroids)
grid.cells:foreach(update_boundary_centroids)
grid.cells:foreach(init_weights)
grid.cells:foreach(init_halo_layer)
grid.cells:foreach(set_initial_conditions)

-- Write out the original mesh with the initial conditions loaded.

write_tecplot(grid, itime)

-- Main iteration loop

while (time <= config.finaltime) do
  
  -- Set the current time and iteration number.
  
  currenttime:set(time)
  itime = itime + 1
  
  -- Update the solution containers in time.
  
  solution_update(grid.cells)
  
  -- Prepare header for console output for the current inner iteration.
  
  io.stdout:write("\n\n Current time step: ",
                  string.format(" %2.6d",itime), ". Time: ",
                  string.format(" %2.6e",time)," s.\n")
  print("=== CONVERGENCE HISTORY =============================================")
  io.stdout:write(string.format("%8s",'    Iter'),
                  string.format("%12s",'   Res(U)'),
                  string.format("%12s",'Res(V)'),
                  string.format("%12s",'Res(p)'),
                  string.format("%12s",'Res(T)'),
                  string.format("%12s",'Mas I/O'),'\n')
  
  -- Begin inner iteration loop.
  
  for istep= 1,config.nsteps do
    
    calcuv(grid.cells)  -- Solve the momentum equations.
    
    calcp(grid.cells)   -- Solve pressure equation.
    
    correct(grid.cells) -- Correct the velocity and pressure fields.
    
    if (solve_energy) then
      calct(grid.cells)   -- Solve the energy equation.
    end
    
    -- Output residuals after this inner iteration.
    
    io.stdout:write(string.format("%8d",istep),
                    string.format("  %7.4E",cmath.fabs(resu:get())),
                    string.format("  %7.4E",cmath.fabs(resv:get())),
                    string.format("  %7.4E",cmath.fabs(resp:get())),
                    string.format("  %7.4E",cmath.fabs(resT:get())),
                    string.format("  %+7.4E",sum:get()),"\n")
    
    -- Detect divergence and exit if found.
    
    local global_tol=cmath.fmax(cmath.fmax(cmath.fmax(resu:get(),resv:get()),
                                           resp:get()),resT:get())
    if (global_tol > diverged) then
      print(" Divergence detected...\n")
      time  = 2*config.finaltime
      break
    end
    
    -- Exit if we have converged the inner iteration loop early.
    
    if (global_tol < config.converged) then break end
    
  end
  
  -- If requested on this iteration, write output files.
  
  if (itime % itimeprint == 0) then
    write_tecplot (grid, itime)
    write_csv_x0 (grid, "u", "u_x0.csv")
    write_csv_y0 (grid, "v", "v_y0.csv")
    write_csv_x0 (grid, "T", "T_x0.csv")
    write_csv_y0 (grid, "T", "T_y0.csv")
    write_csv_x0 (grid, "p", "p_x0.csv")
    write_csv_y0 (grid, "p", "p_y0.csv")
  end
  
  -- Output global quantities of interest to the console.
  
  output(grid.cells)
  
  -- Increment our physical time before moving to the next outer iteration.
  
  time = time + config.dt
  
end

-- Write solution files once more upon exit.

write_tecplot (grid, itime)
write_csv_x0 (grid, "u", "u_x0.csv")
write_csv_y0 (grid, "v", "v_y0.csv")
write_csv_x0 (grid, "T", "T_x0.csv")
write_csv_y0 (grid, "T", "T_y0.csv")
write_csv_x0 (grid, "p", "p_x0.csv")
write_csv_y0 (grid, "p", "p_y0.csv")

print("\n\n")
print("=== EXIT SUCCESS ====================================================")
print("\n\n")
