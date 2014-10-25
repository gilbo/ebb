import 'compiler.liszt'


local Tetmesh = L.require 'examples.fem.tetmesh'
local VEGFileIO = L.require 'examples.fem.vegfileio'

local turtle = VEGFileIO.LoadTetmesh
  'examples/fem/turtle-volumetric-homogeneous.veg'






function initConfigurations()
  local options = {
    volumetricMeshFilename = 'examples/fem/turtle-volumetric-homogeneous.veg',
    timestep                    = 0.01,
    dampingMassCoef             = 1.0, -- alt. 10.0
    dampingStiffnessCoef        = 0.01, -- alt 0.001
    deformableObjectCompliance  = 30.0,
    frequencyScaling            = 1.0,

    maxIterations               = 1,
    epsilon                     = 1e-6,
    numInternalForceThreads     = 4,
    numTimesteps                = 10,
  }
  return options
end


-- TODO: How is the result stored?
function computeMassMatrix()

end

-- TODO: How is the result stored?
function precomputeStVKIntegrals(options)
  local use_low_memory = options.use_low_memory

end

-- TODO: How is the result stored
function computeInternalForces()

end
-- TODO: How is the result stored
function computeStiffnessMatrix()

end

-- TODO: How is the result stored
function computeForceModel()

end

-- TODO: Again Data Model and skeleton needs to be filled in more here...
function buildImplicitBackwardsEulerIntegrator(opts)

  return {} -- should be an actual integrator object
end


function main()
  local options = initConfigurations()

  local volumetric_mesh = turtle

  local nvertices = volumetric_mesh:nVerts()
  -- No fixed vertices for now
  local numFixedVertices = 0
  local numFixedDOFs     = 3*numFixedVertices
  local fixedDOFs        = nil

  computeMassMatrix()

  precomputeStVKIntegrals{
    use_low_memory = true,
  } -- called StVKElementABCDLoader:load() in ref

  computeInternalForces()
  computeStiffnessMatrix()

  computeForceModel()

  local integrator = buildImplicitBackwardsEulerIntegrator{
    n_vars                = 3*nvertices,
    timestep              = options.timestep,
    positiveDefinite      = 0,
    nFixedDOFs            = 0,
    dampingMassCoef       = options.dampingMassCoef,
    dampingStiffnessCoef  = options.dampingStiffnessCoef,
    maxIterations         = options.maxIterations,
    epsilon               = options.epsilon,
    numSolverThreads      = options.numSolverThreads,
  }

  local setInitialConditions = function() end -- should be liszt kernel

  for i=1,options.numTimesteps do
    integrator:setExternalForcesToZero()
    if i == 1 then
      setInitialConditions()
    end

    integrator:doTimestep()
  end

  -- read out the state here somehow?
end






