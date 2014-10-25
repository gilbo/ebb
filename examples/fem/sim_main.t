import 'compiler.liszt'


local Tetmesh = L.require 'examples.fem.tetmesh'
local VEGFileIO = L.require 'examples.fem.vegfileio'

local turtle = VEGFileIO.LoadTetmesh
  'examples/fem/turtle-volumetric-homogeneous.veg'

local mesh = turtle





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


------------------------------------------------------------------------------
-- For corresponding VEGA code, see
--    libraries/volumetricMesh/generateMassMatrix.cpp (computeMassMatrix)
--    libraries/volumetricMesh/tetMesh.cpp (computeElementMassMatrix)

-- TODO: How is the result stored?
-- The main issue is how compressed the storage should be.
-- The general form represents the vertex-vertex mass relationship as a
-- 3x3 matrix, but the actual computation used only requires
-- a scalar for each vertex-vertex relationship...
-- TODO: Make sure that the reference and our code use the same strategy

-- ELEMENT == TET
-- probably should become a liszt function...
function computeElementMassMatrix()
  -- compute the "4x4" mass matrix for a tetrahedral element
  -- where each entry is a 3x3 diagonal matrix
  -- but we don't really need to use the 3x3??????
end
function computeMassMatrix()
  -- Q: Is inflate3Dim flag on?
  -- A: Yes.  This means we want the full mass matrix,
  --    not just a uniform scalar per-vertex

  -- LOOP OVER THE TETRAHEDRA (liszt kernel probably)
    -- BUFFER = COMPUTE TETRAHEDRAL MASS MATRIX
    -- LOOP OVER VERTICES OF THE TETRAHEDRON
      -- LOOP OVER VERTICES AGAIN
        -- dump a diagonal matrix down for each vertex pair
        -- but these diagonal matrices are uniform... ugh?????
  -- CLOSE LOOP

  -- any cleanup code we need goes here
end

------------------------------------------------------------------------------
-- For corresponding VEGA code, see
--    libraries/stvk/StVKTetABCD.cpp (most of the file)

-- STORAGE for output from this part of setup
mesh.tetrahedra:NewField('volume', L.double)
mesh.tetrahedra:NewField('Phig0', L.vec3d)
mesh.tetrahedra:NewField('Phig1', L.vec3d)
mesh.tetrahedra:NewField('Phig2', L.vec3d)
mesh.tetrahedra:NewField('Phig3', L.vec3d)

function precomputeStVKIntegrals(options)
  local use_low_memory = options.use_low_memory
  -- THIS DOES use low memory in the reference code...

  -- note elementData == { volume, Phig[4]:vec3d }

  -- (StVKTetABCD::StVKTetABCD)
  -- LOOP OVER THE TETRAHEDRA (liszt kernel probably)
    -- GET THE 4 VERTICES
    -- (StVKTetABCDStVKSingleTetABCD)
    -- COMPUTE THE TET VOLUME AND STORE
    -- Loop i 0,1,2,3 (OVER VERTICES)
      -- Loop j 0,1,2 (OVER XYZ COORDINATES)
        -- IN HERE WE SET Phig[i][j] which is a per-tetrahedron value
    -- END LOOP
  -- END LOOP
end

------------------------------------------------------------------------------

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




------------------------------------------------------------------------------

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






