import 'compiler.liszt'


local Tetmesh = L.require 'examples.fem.tetmesh'
local VEGFileIO = L.require 'examples.fem.vegfileio'

local turtle = VEGFileIO.LoadTetmesh
  'examples/fem/turtle-volumetric-homogeneous.veg'

local mesh = turtle
-- TODO: parse density and fill in mesh.density while loading mesh
mesh.density = 1000



local gravity = 9.81

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
-- Helper functions and kernels

-- Compute absolute value for a given variable
local fabs = liszt function(num)
  var result = num
  if num < 0 then result = -num end
  return result
end

-- Compute volume, given positions for 4 vertices
local tetVolume = liszt function(a, b, c, d)
  return (fabs(L.dot(a - d, L.cross(b - d, c - d))) / 6)
end

-- Get element density for a mesh element (tetreahedron)
local getElementDensity = liszt function(a)
  return mesh.density
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

-- The following implementation combines computing element matrix and updating
-- global mass matrix, for convenience.
-- Also, it corresponds to inflate3Dim=False.
-- Inflate3Dim adds the same entry for each dimension, in the implementation
-- at libraries/volumetricMesh/generateMassMatrix.cpp. This is redundant,
-- unless the mass matrix is modified in a different way for each dimension
-- sometime later. What should we do??
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

  mesh.edges:NewField('mass', L.double):Load(0)
  local buildMassMatrix = liszt kernel(t : mesh.tetrahedra)
    var tet_vol = tetVolume(t.v0.pos, t.v1.pos, t.v2.pos, t.v3.pos)
    var factor = tet_vol * getElementDensity(t) / 20
    t.e00.mass += factor * 2
    t.e01.mass += factor
    t.e02.mass += factor
    t.e03.mass += factor
    t.e10.mass += factor
    t.e11.mass += factor * 2
    t.e12.mass += factor
    t.e13.mass += factor
    t.e20.mass += factor
    t.e21.mass += factor
    t.e22.mass += factor * 2
    t.e23.mass += factor
    t.e30.mass += factor
    t.e31.mass += factor
    t.e32.mass += factor
    t.e33.mass += factor * 2
  end
  buildMassMatrix(mesh.tetrahedra)

  -- any cleanup code we need goes here
end

------------------------------------------------------------------------------
-- For corresponding VEGA code, see
--    libraries/stvk/StVKTetABCD.cpp (most of the file)

-- STORAGE for output from this part of setup
mesh.tetrahedra:NewField('volume', L.double)
--mesh.tetrahedra:NewField('Phig0', L.vec3d)
--mesh.tetrahedra:NewField('Phig1', L.vec3d)
--mesh.tetrahedra:NewField('Phig2', L.vec3d)
--mesh.tetrahedra:NewField('Phig3', L.vec3d)
-- access as Phig[3*i + j] i == vertex, j == coord (i.e. x,y,z)
mesh.tetrahedra:NewField('Phig', L.vector(L.double, 12))

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

-- need to define dots and
-- A, B, C, D here

-- The VEGA Code seems to compute the dots matrix once, and then
-- cache it for the duration of a per-tet computation rather than
-- allocate disk space
local tetDots = liszt function(tet)
  var dots : L.vector(L.double, 16)
  for i=0,4 do
    var Phigi : L.vec3d =
      { tet.Phig[3*i + 0], tet.Phig[3*i + 1], tet.Phig[3*i + 2] }
    for j=0,4 do
      var Phigj : L.vec3d =
        { tet.Phig[3*j + 0], tet.Phig[3*j + 1], tet.Phig[3*j + 2] }

      dots[4*i+j] = L.dot(Phigi, Phigj)
    end
  end
  return dots
end

local tetCoeffA = liszt function(tet, i, j)
  -- Volume * tensor product of Phig i and Phig j
  -- results in a matrix
end

local tetCoeffB = liszt function(tet, i, j)
  -- Volume * dots[i][j]
  -- Scalar
end

local tetCoeffC = liszt function(tet, i, j, k)
  -- Volume * dots[j][k] * Phig i
  -- Vector
end

local tetCoeffD = liszt function(tet, i, j, k, l)
  -- Volume * dots[i][j] * dots[k][l]
  -- Scalar
end

------------------------------------------------------------------------------
-- For corresponding VEGA code, see
--    libraries/stvk/StVKInternalForces.cpp (most of the file)

mesh.tetrahedra:NewField('lambdaLame', L.double)
mesh.tetrahedra:NewField('muLame', L.double)

-- TODO: How is the result stored
function computeInternalForces()
  -- add Gravity = 0
  -- gravity is defined above
  -- I guess we're not using gravity right now though...

  -- LOOP OVER TETRAHEDRA
    -- mat = Get Element Material
    -- Populate lambdaLame and muLame
  -- END LOOP
end

-- extra functions supplied by this module

function addIFLinearTermsContribution()
end
function addIFQuadraticTermsContribution()
end
function addIFCubicTermsContribution()
end
function computeEnergyContribution()
end

function computeEnergy()
  computeEnergyContribution()
end

function computeForces()
  -- RESET FORCES VECTOR HERE

  addIFLinearTermsContribution()
  addIFQuadraticTermsContribution()
  addIFCubicTermsContribution()
end

------------------------------------------------------------------------------
-- For corresponding VEGA code, see
--    libraries/stvk/StVKStiffnessMatrix.cpp (most of the file)

-- TODO:
--    I have no idea how we're storing the stiffness matrix
--    But it definitely needs to be frequently recomputed

-- May not need this
function allocateStiffnessMatrixTopology()

  -- LOOP OVER TETRAHEDRA
    -- LOOP OVER VERTICES x VERTICES (i,j)
      -- Create a 4x4x3x3 tensor object's 3x3 entry here
  -- END LOOP
end

-- May not need this
function allocateStiffnessMatrix()
  -- Don't Need to Recompute/Store Per-Tet Lame Constants

  local stiffnessMatrixTopology
  computeStiffnessMatrixTopology()

  -- Acceleration only (do we need this?  I don't think so)
  -- row[n_tet][4] : int
  -- col[n_tet][4*4] : int

  -- build acceleration indices ???
  -- LOOP OVER TETRAHEDRA
    -- LOOP OVER VERTICES of TET
      -- ROW[TET][VERTEX] = VERTEX INDEX
    -- END LOOP

    -- LOOP OVER VERTICES x VERTICES of TET
      -- COLUMN[TET][4*i + j] = DENSE LOCATION OF ROW[TET][i],ROW[TET][j]
    -- END LOOP
  -- END LOOP
end

function addStiffLinearTermsContribution()
end
function addStiffQuadraticTermsContribution()
end
function addStiffCubicTermsContribution()
end

function computeStiffnessMatrix()
  -- RESET STIFFNESS MATRIX TO ZERO
  addStiffLinearTermsContribution()
  addStiffQuadraticTermsContribution()
  addStiffCubicTermsContribution()
end

------------------------------------------------------------------------------

-- TODO: How is the result stored
function computeForceModel()

end

------------------------------------------------------------------------------

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
    -- integrator:setExternalForcesToZero()
    if i == 1 then
      setInitialConditions()
    end

    -- integrator:doTimestep()
  end

  -- read out the state here somehow?
end

main()
