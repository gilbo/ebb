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
-- Helper functions, kernels, variables etc

-- Compute absolute value for a given variable
local fabs = liszt function(num)
  var result = num
  if num < 0 then result = -num end
  return result
end

-- Compute volume for a tetrahedral element
local getElementVolume = liszt function(t)
  var a = t.v0.pos
  var b = t.v1.pos
  var c = t.v2.pos
  var d = t.v3.pos
  return (fabs(L.dot(a - d, L.cross(b - d, c - d))) / 6)
end

-- Get element density for a mesh element (tetreahedron)
local getElementDensity = liszt function(a)
  return L.double(mesh.density)
end

-- Identity matrix
local getId3 = liszt function()
  return { { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 } }
end

-- Matrix with all entries equal to value v
local constantMatrix3 = liszt function(v)
  return { { v, v, v }, { v, v, v }, { v, v, v } }
end

-- Tensor product of 2 vectors
local tensor3 = liszt function(a, b)
  var result = { { a[0] * b[0], a[0] * b[1], a[0] * b[2] },
                 { a[1] * b[0], a[1] * b[1], a[1] * b[2] },
                 { a[2] * b[0], a[2] * b[1], a[2] * b[2] } }
  return result
end

------------------------------------------------------------------------------
-- Allocate common fields

mesh.tetrahedra:NewField('lambdaLame', L.double)
mesh.tetrahedra:NewField('muLame', L.double)
mesh.vertices:NewField('forces', L.vec3d)
mesh.vertices:NewField('displacement', L.vec3d)

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
function computeMassMatrix(mesh)
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
    var tet_vol = getElementVolume(t)
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
  var AStub : L.mat3d = getId3()
  return AStub
end

local tetCoeffB = liszt function(tet, i, j)
  -- Volume * dots[i][j]
  -- Scalar
  var BStub : L.double = 1
  return BStub
end

local tetCoeffC = liszt function(tet, i, j, k)
  -- Volume * dots[j][k] * Phig i
  -- Vector
  var CStub : L.vec3d = {1, 1, 1}
  return CStub
end

local tetCoeffD = liszt function(tet, i, j, k, l)
  -- Volume * dots[i][j] * dots[k][l]
  -- Scalar
  var DStub : L.double = 1
  return DStub
end

------------------------------------------------------------------------------
-- For corresponding VEGA code, see
--    libraries/stvk/StVKInternalForces.cpp (most of the file)

-- TODO: How is the result stored
-- I think this corresponds to initialization.
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
-- Outer loop is generally over all elements (tetrahedra).
-- Result is stored as a 3D vector field over all the vertices.

function addIFLinearTermsContribution()
end
function addIFQuadraticTermsContribution()
end
function addIFCubicTermsContribution()
end

--[[
-- We probably do not need energy contribution.
function computeEnergyContribution()
end
function computeEnergy()
  computeEnergyContribution()
end
]]

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

-- NOTES:
-- row_ and col_ are only acceleration structures, used to index into the
-- sparse stiffness matrix. Stiffness matrix is stored as a sparse 3n x 3n
-- matrix, where n is the number of vertices. I think the reason why it is
-- 3n X 3n is because they need 3X3 over nXn points.
-- GetStiffnessMatrixTopology is similar to the topology matrix constructed for
-- mass. There is an entry in the nXn (or equivalently 3n X 3n) matrix if the
-- two vertices belong to the same element. For tetrahedra, this means diagonal
-- entries and positions corresponding to an undirected edge between two
-- vertices are set.
-- row_ stores the indices into the 3nx3n stiffness matrix for each vertex of
-- a tetrahedral element (what row does each vertex of an element map to).
-- column_ stores the column position corresponding to the entry for (i, j)
-- vertices of an element. That is, for the first row of column_, the first
-- entry tells us which column in the stiffnexx matrix xorresponds to the first
-- vertex of the element, the second one says which one corresponds to the
-- second vertex of the element and so on.
-- BASICALLY, the entries from the ith row of a column_ should be used as column
-- indices for the row given by ith entry of row_, and correspond to the
-- relation between ith vertex of an element, and the remaining vertices of
-- that element.

-- At a very high level, Add***Terms loop over all the elements, and add a 3X3
-- block at (i, j) (technically (3*i, 3*j)) position corresponding to each
-- (x, y) vertex pair for the element. i is the row, and j is the column,
-- in the nXn (3nX3n) stiffness matrix, corresponding to vertices (x,y) which
-- go from (0,0) to (3,3). The rest of the code performs further loops to
-- calculate the 3X3 matrix, using precomputed integrals and vertex
-- displacements.

-- We should probably store the stiffness matrix as a field of 3X3 matrices
-- over edges??

mesh.edges:NewField('stiffness', L.mat3d)

-- PERFORMANCE NOTE:
-- All operations are written as scatter operations. The code may perform
-- better if we rewrite the operations as gather operations over edges, as
-- against scatter from tetrahedra.

-- Linear contributions to stiffness matrix.
local addStiffLinearHelper = liszt function(t, e)
  var mat = t.muLame * tetCoeffB(t, e.head, e.tail) * getId3()
  mat += (t.lambdaLame * tetCoeffA(t, e.tail, e.head) +
         (t.muLame * tetCoeffA(t, e.head, e.tail)))
  e.stiffness += mat
end
local addStiffLinearTerms = liszt function(t)
  -- allocate and prepare precomputed integral element
  addStiffLinearHelper(t, t.e00)
  addStiffLinearHelper(t, t.e01)
  addStiffLinearHelper(t, t.e02)
  addStiffLinearHelper(t, t.e03)
  addStiffLinearHelper(t, t.e10)
  addStiffLinearHelper(t, t.e11)
  addStiffLinearHelper(t, t.e12)
  addStiffLinearHelper(t, t.e13)
  addStiffLinearHelper(t, t.e20)
  addStiffLinearHelper(t, t.e21)
  addStiffLinearHelper(t, t.e22)
  addStiffLinearHelper(t, t.e23)
  addStiffLinearHelper(t, t.e30)
  addStiffLinearHelper(t, t.e31)
  addStiffLinearHelper(t, t.e32)
  addStiffLinearHelper(t, t.e33)
  --  release precomputed integral element
end

-- Quadratic contributions to stiffness matrix.
local addStiffQuadraticHelper2 = liszt function(t, e, a, mat)
  var qa = a.displacement
  var c0v = t.lambdaLame * tetCoeffC(t, e.tail, a, e.head) +
            t.muLame * tetCoeffC(t, e.head, a, e.tail) +
            tetCoeffC(t, a, e.head, e.tail)
  mat += tensor3(qa, c0v)
  var c1v = t.lambdaLame * tetCoeffC(t, e.head, a, e.tail) +
            t.muLame * tetCoeffC(t, e.tail, e.head, a) +
            tetCoeffC(t, a, e.head, e.tail)
  mat += tensor3(qa, c1v)
  var c2v = t.lambdaLame * tetCoeffC(t, a, e.head, e.tail) +
            t.muLame * tetCoeffC(t, e.tail, a, e.head) +
            tetCoeffC(t, e.head, a, e.tail)
  var dotp = L.dot(qa, c2v)
  mat += dotp * getId3()
end
local addStiffQuadraticHelper1 = liszt function(t, e)
  var mat : L.mat3d = constantMatrix3(0)
  addStiffQuadraticHelper2(t, e, t.v0, mat)
  addStiffQuadraticHelper2(t, e, t.v1, mat)
  addStiffQuadraticHelper2(t, e, t.v2, mat)
  addStiffQuadraticHelper2(t, e, t.v3, mat)
  e.stiffness += mat
end
local addStiffQuadraticTerms = liszt function(t)
  -- allocate and prepare precomputed integral element
  addStiffQuadraticHelper1(t, t.e00)
  addStiffQuadraticHelper1(t, t.e01)
  addStiffQuadraticHelper1(t, t.e02)
  addStiffQuadraticHelper1(t, t.e03)
  addStiffQuadraticHelper1(t, t.e10)
  addStiffQuadraticHelper1(t, t.e11)
  addStiffQuadraticHelper1(t, t.e12)
  addStiffQuadraticHelper1(t, t.e13)
  addStiffQuadraticHelper1(t, t.e20)
  addStiffQuadraticHelper1(t, t.e21)
  addStiffQuadraticHelper1(t, t.e22)
  addStiffQuadraticHelper1(t, t.e23)
  addStiffQuadraticHelper1(t, t.e30)
  addStiffQuadraticHelper1(t, t.e31)
  addStiffQuadraticHelper1(t, t.e32)
  addStiffQuadraticHelper1(t, t.e33)
end

-- may have to reverse head, tail below
local addStiffCubicTerms = liszt function(t)
  -- allocate and prepare precomputed integral element
  var lambda = t.lambdaLame
  var mu = t.muLame
  -- loop over vertex-vertex pairs in the element, that is, all the 16 edge
  -- fields over a tetrahedron (e00 to e33) e
  --   var mat = { { 0, 0, 0 }, { 0, 0, 0 }, { 0, 0, 0 } }
  --   loop over all element vertices a
  --     var qa = a.displacement
  --     loop over all element vertices b
  --       var qb = b.displacement
  --       var d0 = lambda * precomputedD(t, a, e.tail, b, e.head) +
  --                mu * precomputedD(t, a, e.head, b, e.tail) +
  --                precomputedD(t, a, b, e.tail, e.head)
  --       mat += d0 * (qa tensor qb)
  --       var d1 = 0.5 * lambda * precomputedD(t, a, b, e.tail, e.head) +
  --                mu * precomputedD(t, a, e.tail, b, e.head)
  --       var dotpd = d1 * dot(qa, qb)
  --       mat += dotpd * { { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 } }
  --  e.stiffness += mat
  --  release precomputed integral element
end

local resetStiffnessMatrixKernel = liszt kernel(e : mesh.edges)
  e.stiffness = constantMatrix3(0)
end

local computeStiffnessMatrixKernel = liszt kernel(t : mesh.tetrahedra)
  addStiffLinearTerms(t)
  addStiffQuadraticTerms(t)
  addStiffCubicTerms(t)
end

function computeStiffnessMatrix(mesh)
  computeStiffnessMatrixKernel(mesh.tetrahedra)
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

  computeMassMatrix(volumetric_mesh)

  precomputeStVKIntegrals{
    use_low_memory = true,
  } -- called StVKElementABCDLoader:load() in ref

  computeInternalForces()
  computeStiffnessMatrix(volumetric_mesh)

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
