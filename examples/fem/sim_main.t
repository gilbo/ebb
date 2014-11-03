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

-- Compute determinant for matrix formed by vertex positions
local getElementDet = liszt function(t)
  var a = t.v[0].pos
  var b = t.v[1].pos
  var c = t.v[2].pos
  var d = t.v[3].pos
  return (L.dot(a - d, L.cross(b - d, c - d)))
end

-- Compute volume for a tetrahedral element
local getElementVolume = liszt function(t)
  return (fabs(getElementDet(t)/6))
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

-- Matrix vector product
local multiplyMatVec3 = liszt function(M, x)
  var y = { M[0, 0]*x[0] + M[0, 1]*x[1] + M[0, 2]*x[2],
            M[1, 0]*x[0] + M[1, 1]*x[1] + M[1, 2]*x[2],
            M[2, 0]*x[0] + M[2, 1]*x[1] + M[2, 2]*x[2] }
  return y
end

------------------------------------------------------------------------------
-- Allocate common fields

mesh.tetrahedra:NewField('lambdaLame', L.double)
mesh.tetrahedra:NewField('muLame', L.double)
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
    for i = 0,3 do
      for j = 0,3 do
        var mult_const = 1
        if i == j then
          var mult_const = 2
        end
        t.e[i, j].mass += factor * mult_const
      end
    end
  end
  buildMassMatrix(mesh.tetrahedra)

  -- any cleanup code we need goes here
end

------------------------------------------------------------------------------
-- For corresponding VEGA code, see
--    libraries/stvk/StVKTetABCD.cpp (most of the file)

-- STORAGE for output from this part of setup
mesh.tetrahedra:NewField('volume', L.double)
mesh.tetrahedra:NewField('Phig', L.mat3x4d)

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

local precomputeStVKIntegrals = liszt kernel(t : mesh.tetrahedra)
  var det = getElementDet(t)
  for i = 0,3 do
    for j = 0,2 do
      var column0 : L.vec3d
      var column1 : L.vec3d
      var countI = 0
      for ii = 0,3 do
        if ii ~= i then
          var countJ = 0
          for jj = 0,2 do
            if jj ~= j then
              if countJ == 0 then
                column0[countI] = t.v[ii].pos[jj]
              else
                column1[countI] = t.v[ii].pos[jj]
              end
              countJ += 1
            end
          countI += 1
        end
      end
      var sign = 0
      if (i + j) % 2 == 0 then
        sign = 1
      else
        sign = -1
      end
      t.Phig[i, j] = sign * L.dot( L.vec3d( { 1, 1, 1 } ),
                                   L.cross(column0, column1) ) / det
      end
    end
  end
end

-- need to define dots and
-- A, B, C, D here

-- The VEGA Code seems to compute the dots matrix once, and then
-- cache it for the duration of a per-tet computation rather than
-- allocate disk space
local tetDots = liszt function(tet)
  var dots : L.mat3d
  for i=0,4 do
    var Phigi : L.vec3d =
      { tet.Phig[i, 0], tet.Phig[i, 1], tet.Phig[i, 2] }
    for j=0,4 do
      var Phigj : L.vec3d =
        { tet.Phig[j, 0], tet.Phig[j, 1], tet.Phig[j, 2] }
      dots[i, j] = L.dot(Phigi, Phigj)
    end
  end
  return dots
end

local tetCoeffA = liszt function(tet, dots, volume, i, j)
  var phi = tet.Phig
  return ( volume * tensor3( { phi[i, 0], phi[i, 1], phi[i, 2] },
                             { phi[j, 0], phi[j, 1], phi[j, 2] } ) )
end

local tetCoeffB = liszt function(tet, dots, volume, i, j)
  return volume * dots[i, j]
end

local tetCoeffC = liszt function(tet, dots, volume, i, j, k)
  var phi = tet.Phig
  var res : L.vec3d = volume * dots[j, k] * {1, 1, 1}
                      { phi[i, 0], phi[i, 1], phi[i, 2] }
  return res
end

local tetCoeffD = liszt function(tet, dots, volume, i, j, k, l)
  return ( volume * dots[i, j] * dots[k, l] )
end

------------------------------------------------------------------------------
-- For corresponding VEGA code, see
--    libraries/stvk/StVKInternalForces.cpp (most of the file)

mesh.vertices:NewField('forces', L.vec3d):Load({0, 0, 0})

-- extra functions supplied by this module
-- Outer loop is generally over all elements (tetrahedra).
-- Result is stored as a 3D vector field over all the vertices.

-- Linear contributions to internal forces
local addIFLinearTerms = liszt kernel(t : mesh.tetrahedra)
  var dots = tetDots(t)
  var volume = getElementVolume(t)
  for ci = 0,3 do
    var c = t.v[ci]
    for ai = 0,3 do
      var qa = t.v[ai].displacement
      var force = t.lambdaLame *
                  multiplyMatVec3(tetCoeffA(t, dots, volume, ci, ai), qa) +
                  (t.muLame * tetCoeffB(t, dots, volume, ai, ci)) * qa +
                  t.muLame *
                  multiplyMatVec3(tetCoeffA(t, dots, volume, ai, ci), qa)
      c.forces += force
    end
  end
end

-- Quadratic contributions to internal forces
local addIFQuadraticTerms = liszt kernel(t : mesh.tetrahedra)
  var dots = tetDots(t)
  var volume = getElementVolume(t)
  for ci = 0,3 do
    var c = t.v[ci]
    for ai = 0,3 do
      var qa = t.v[ai].displacement
      for bi = 0,3 do
        var qb = t.v[bi].displacement
        var dotp = L.dot(qa, qb)
        var forceTerm1 = 0.5 * t.lambdaLame * dotp *
                         tetCoeffC(t, dots, volume, ci, ai, bi) +
                         t.muLame * dotp *
                         tetCoeffC(t, dots, volume, ai, bi, ci)
        var C = t.lambdaLame * tetCoeffC(t, dots, volume, ai, bi, ci) +
                t.muLame * ( tetCoeffC(t, dots, volume, ci, ai, bi) +
                tetCoeffC(t, dots, volume, bi, ai, ci) )
        var dotCqa = L.dot(C, qa)
        c.forces += forceTerm1 + dotCqa * qb
      end
    end
  end
end

-- Cubic contributions to internal forces
local addIFCubicTerms = liszt kernel(t : mesh.tetrahedra)
  var dots = tetDots(t)
  var volume = getElementVolume(t)
  for ci = 0,3 do
    var c = t.v[ci]
    for ai = 0,3 do
      var qa = t.v[ai].displacement
      for bi = 0,3 do
        var qb = t.v[bi].displacement
        for di = 0,3 do
          var d = t.v[di]
          var qd = d.displacement
          var dotp = L.dot(qa, qb)
          var scalar = dotp * ( 0.5 * t.lambdaLame *
                                tetCoeffD(t, dots, volume, ai, bi, ci, di) +
                                t.muLame *
                                tetCoeffD(t, dots, volume, ai, ci, bi, di) )
          c.forces += scalar * qd
        end
      end
    end
  end
end

local resetForces = liszt kernel(v : mesh.vertices)
  v.forces = {0, 0, 0}
end

--[[
-- We probably do not need energy contribution.
function computeEnergyContribution()
end
function computeEnergy()
  computeEnergyContribution()
end
]]

local computeForcesHelper = function(tetrahedra)
  print("Computing linear contributions to forces ...")
  addIFLinearTerms(tetrahedra)
  print("Computing quadratic contributions to forces ...")
  addIFQuadraticTerms(tetrahedra)
  print("Computing cubic contributions to forces ...")
  addIFCubicTerms(tetrahedra)
end

function computeForces(mesh)
  print("Computing forces ...")
  resetForces(mesh.vertices)
  computeForcesHelper(mesh.tetrahedra)
  print("Completed computing forces.")
end

------------------------------------------------------------------------------
-- For corresponding VEGA code, see
--    libraries/stvk/StVKStiffnessMatrix.cpp (most of the file)

-- At a very high level, Add___Terms loop over all the elements, and add a 3X3
-- block at (i, j) (technically (3*i, 3*j)) position corresponding to each
-- (x, y) vertex pair for the element. i is the row, and j is the column,
-- in the nXn (3nX3n) stiffness matrix, corresponding to vertices (x,y) which
-- go from (0,0) to (3,3). The rest of the code performs further loops to
-- calculate the 3X3 matrix, using precomputed integrals and vertex
-- displacements.

mesh.edges:NewField('stiffness', L.mat3d)

-- PERFORMANCE NOTE:
-- All operations are written as scatter operations. The code may perform
-- better if we rewrite the operations as gather operations over edges, as
-- against scatter from tetrahedra.

-- Linear contributions to stiffness matrix
local addStiffLinearTerms = liszt kernel(t : mesh.tetrahedra)
  var dots = tetDots(t)
  var volume = getElementVolume(t)
  for ci = 0,3 do
    for ai = 0,3 do
      var mat = t.muLame * tetCoeffB(t, dots, volume, ai, ci) * getId3()
      mat += (t.lambdaLame * tetCoeffA(t, dots, volume, ci, ai) +
             (t.muLame * tetCoeffA(t, dots, volume, ai, ci)))
      t.e[ci, ai].stiffness += mat
    end
  end
end

-- Quadratic contributions to stiffness matrix
local addStiffQuadraticTerms = liszt kernel(t : mesh.tetrahedra)
  var dots = tetDots(t)
  var volume = getElementVolume(t)
  for ci = 0,3 do
    for ai = 0,3 do
      var qa = t.v[ai].displacement
      var mat : L.mat3d = constantMatrix3(0)
      for ei = 0,3 do
        var c0v = t.lambdaLame * tetCoeffC(t, dots, volume, ci, ai, ei) +
                  t.muLame * ( tetCoeffC(t, dots, volume, ei, ai, ci) +
                               tetCoeffC(t, dots, volume, ai, ei, ci) )
        mat += tensor3(qa, c0v)
        var c1v = t.lambdaLame * tetCoeffC(t, dots, volume, ei, ai, ci) +
                  t.muLame * ( tetCoeffC(t, dots, volume, ci, ei, ai) +
                               tetCoeffC(t, dots, volume, ai, ei, ci) )
        mat += tensor3(qa, c1v)
        var c2v = t.lambdaLame * tetCoeffC(t, dots, volume, ai, ei, ci) +
                  t.muLame * ( tetCoeffC(t, dots, volume, ci, ai, ei) +
                               tetCoeffC(t, dots, volume, ei, ai, ci) )
        var dotp = L.dot(qa, c2v)
        mat += dotp * getId3()
      end
      t.e[ci, ai].stiffness += mat
    end
  end
end

-- Cubic contributions to stiffness matrix
local addStiffCubicTerms = liszt kernel(t : mesh.tetrahedra)
  var dots = tetDots(t)
  var volume = getElementVolume(t)
  for ci = 0,3 do
    for ei = 0,3 do
      var mat : L.mat3d = constantMatrix3(0)
      for ai = 0,3 do
        var qa = t.v[ai].displacement
        for bi = 0,3 do
          var qb = t.v[bi].displacement
          var d0 = t.lambdaLame * tetCoeffD(t, dots, volume, ai, ci, bi, ei) +
                   t.muLame * ( tetCoeffD(t, dots, volume, ai, ei, bi, ci) +
                                tetCoeffD(t, dots, volume, ai, bi, ci, ei) )
          mat += d0 * (tensor3(qa, qb))
          var d1 = 0.5 * t.lambdaLame *
                   tetCoeffD(t, dots, volume, ai, bi, ci, ei) +
                   t.muLame * tetCoeffD(t, dots, volume, ai, ci, bi, ei)
          var dotpd = d1 * L.dot(qa, qb)
          mat += dotpd * getId3()
        end
      end
      t.e[ci, ei].stiffness += mat
    end
  end
end

local resetStiffnessMatrix = liszt kernel(e : mesh.edges)
  e.stiffness = constantMatrix3(0)
end

local computeStiffnessMatrixHelper = function(tetrahedra)
  print("Computing linear contributions to stiffness matrix ...")
  addStiffLinearTerms(tetrahedra)
  print("Computing quadratic contributions to stiffness matrix ...")
  addStiffQuadraticTerms(tetrahedra)
  print("Computing cubic contributions to stiffness matrix ...")
  addStiffCubicTerms(tetrahedra)
end

function computeStiffnessMatrix(mesh)
  print("Computing stiffness matrix ...")
  resetStiffnessMatrix(mesh.edges)
  computeStiffnessMatrixHelper(mesh.tetrahedra)
  print("Completed computing stiffness matrix.")
end

------------------------------------------------------------------------------

-- TODO: Again Data Model and skeleton needs to be filled in more here...

--[[
local ImplicitBackwardEulerIntegrator = {}
ImplicitBackwardEulerIntegrator.__index = ImplicitBackwardEulerIntegrator

function ImplicitBackwardEulerIntegrator.New(opts)
  local stepper = setmetatable({
    internalForceScalingFactor = ???, -- Where does this get set?

    epsilon                     = opts.epsilon,
    timestep                    = opts.timestep,
    dampingMassCoef             = opts.dampingMassCoef,
    dampingStiffnessCoef        = opts.dampingStiffnessCoef,
  }, ImplicitBackwardEulerIntegrator)

  return stepper
end

function ImplicitBackwardEulerIntegrator:DoTimestep()
  local err0 = 0 -- L.Global?
  local errQuotient

  -- store current amplitudes and set initial gues for qaccel, qvel
  -- AHHHHH, THE BELOW CAN BE IMPLEMENTED USING COPIES
  -- LOOP OVER THE STATE VECTOR (vertices for now)
    -- SET qaccel_1 = qaccel = 0
    -- SET q_1 = q
    -- SET qvel_1 = qvel
  -- end

  -- Limit our total number of iterations allowed per timestep
  for numIter = 1, self.maxIterations do
    -- ASSEMBLY
    -- TIMING START
    INTERNAL_FORCES = computeForces() -- PASS ANYTHING?
    STIFFNESS = computeStiffnessMatrix() -- PASS ANYTHING?
    -- TIMING END
    -- RECORD ASSEMBLY TIME

    -- NOTE: NOTE: We can implement these scalings as separate kernels to be
    -- "FAIR" to VEGA or we can fold it into other kernels.
    -- Kinda unclear which to do
    INTERNAL_FORCES = self.internalForceScalingFactor * INTERNAL_FORCES
    STIFFNESS = self.internalForceScalingFactor * STIFFNESS

    -- ZERO out the residual field
    mesh.vertices.qresidual:Load(0)

    -- NOTE: useStaticSolver == FALSE
    --    We just assume this everywhere
    RAYLEIGH_DAMP_MATRIX  = self.dampingStiffnessCoef * STIFFNESS
    RAYLEIGH_DAMP_MATRIX += self.dampingMassCoef * MASS_MATRIX

    -- Build effective stiffness:
    --    Keff = M + h D + h^2 * K
    -- compute force residual, store it into aux variable qresidual
    -- Semi-Implicit Euler
    --    qresidual = h * (-D qdot - fint + fext - h * K * qdot)
    -- Fully-Implicit Euler
    --    qresidual = M (qvel_1-qvel) +
    --                h * (-D qdot - fint + fext - K * (q_1 - q + h * qdot))

    -- superfluous on iteration 1, but safe to run
    if numIter ~= 1 then
      qresidual += STIFFNESS * (q_1-q)
    end

    -- some magic incantations corresponding to the above
    STIFFNESS *= self.timestep
    STIFFNESS += RAYLEIGH_DAMP_MATRIX
    STIFFNESS += dampingMatrix -- DON'T KNOW WHERE THIS IS FROM (constant?)
    qresidual += STIFFNESS * qvel
    STIFFNESS *= self.timestep
    STIFFNESS += MASS_MATRIX

    -- ADD EXTERNAL/INTERNAL FORCES
    qresidual += INTERNAL_FORCES - EXTERNAL_FORCES
    qresidual = self.timestep * qresidual

    -- superfluous on iteration 1, but safe to run
    if numIter ~= 1 then
      qresidual += MASS_MATRIX * (qvel_1-qvel)
    end

    qdelta = qresidual

    -- TODO: This code doesn't have any way of handling fixed vertices
    -- at the moment.  Should enforce that here somehow

    local err = L.global(L.double, 0) -- HUH? L.Global?
    err = REDUCE(qdelta*qdelta) -- only over unconstrained vertices

    -- compute initial error on the 1st iteration
    if numIter ~= 1 then
      err0 = err
      errQuotient = 1
    else
      errQuotient = err / err0
    end

    if errQuotient < self.epsilon*self.epsilon then
      break
    end

    SYSTEM_MATRIX = STIFFNESS -- HERE b/c of REMOVE ROWS...
    RHS = qdelta

    -- SYSTEM SOLVE: SYSTEM_MATRIX * BUFFER = BUFFER_CONSTRAINED
    -- START PERFORMANCE TIMING
    -- ASSUMING PCG FOR NOW
    local solverEpsilon = 1e-6
    local solverMaxIter = 10000
    local err_code = DO_JACOBI_PRECONDITIONED_CG_SOLVE(RESULT, RHS,
                                      solverEpsilon, solverMaxIter)
    if err_code ~= 0 then
      error("ERROR: PCG sparse solver "..
            "returned non-zero exit status "..err_code)
    end
    -- STOP PERFORMANCE TIMING
    -- RECORD SYSTEM SOLVE TIME

    -- Reinsert the rows?

    qvel += qdelta
    q += q_1-q + self.timestep * qvel

    -- for the subset of constrained vertices
      q=0
      qvel=0
      qaccel=0
  end
end

function buildImplicitBackwardsEulerIntegrator(opts)



  return {} -- should be an actual integrator object
end

]]

------------------------------------------------------------------------------

function setInitialConditions()
  -- FROM THE REFERENCE FOR NOW
--  implicitBackwardEulerSparse->SetExternalForcesToZero();
--  // set some force at the first timestep
--  if (i == 0) {
--    for(int j=0; j<r; j++)
--        f[j] = 0; // clear to 0
--    f[0] = -500;
--    f[1] = -500;
--    f[2] = -500;
--    implicitBackwardEulerSparse->SetExternalForces(f);
--  }
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

  print("Computing mass matrix ...")
  computeMassMatrix(volumetric_mesh)

  print("Computing integrals ...")
  precomputeStVKIntegrals(mesh.tetrahedra) -- called StVKElementABCDLoader:load() in ref

  -- TODO: Move these to do time step finally. There is no need for any
  -- initialization. Just testing these calls right now. 
  computeForces(volumetric_mesh)
  computeStiffnessMatrix(volumetric_mesh)

  computeForceModel()

  local integrator = ImplicitBackwardEulerIntegrator.New{
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

  for i=1,options.numTimesteps do
    -- integrator:setExternalForcesToZero()
    if i == 1 then
      setInitialConditions()
    end

    integrator:DoTimestep()
  end

  -- read out the state here somehow?
end

main()
