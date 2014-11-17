import 'compiler.liszt'
local vdb = L.require 'lib.vdb'

local Tetmesh = L.require 'examples.fem.tetmesh'
local VEGFileIO = L.require 'examples.fem.vegfileio'
local PN = L.require 'lib.pathname'

-- print("Loading mesh ...")
local turtle = VEGFileIO.LoadTetmesh
  'examples/fem/turtle-volumetric-homogeneous.veg'

local mesh = turtle
local gravity = 9.81

function initConfigurations()
  local options = {
    volumetricMeshFilename = 'examples/fem/turtle-volumetric-homogeneous.veg',
    timestep                    = 0.1,
    dampingMassCoef             = 1.0, -- alt. 10.0
    dampingStiffnessCoef        = 0.01, -- alt 0.001
    deformableObjectCompliance  = 1.0,

    maxIterations               = 1,
    epsilon                     = 1e-6,
    numTimesteps                = 10,

    cgEpsilon                   = 1e-6,
    cgMaxIterations             = 10000
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
local multiplyVectors = liszt function(x, y)
  return { x[0]*y[0], x[1]*y[1], x[2]*y[2]  }
end

-- Add constant to matrix

------------------------------------------------------------------------------
-- Allocate common fields

mesh.tetrahedra:NewField('lambdaLame', L.double):Load(0)
mesh.tetrahedra:NewField('muLame', L.double):Load(0)
mesh.vertices:NewField('q', L.vec3d):Load({ 0, 0, 0})
mesh.vertices:NewField('qvel', L.vec3d):Load({ 0, 0, 0 })
mesh.vertices:NewField('qaccel', L.vec3d):Load({ 0, 0, 0 })
mesh.vertices:NewField('external_forces', L.vec3d):Load({ 0, 0, 0 })

------------------------------------------------------------------------------
-- Print out fields over edges (things like stiffnexx matrix or mass), to
-- compare side by side with vega output (using sparse_matrix.Save(filename))'

function fieldToString(x)
  if type(x) == 'table' then
    local str = "{ "
    for k,v in ipairs(x) do
      str = str..fieldToString(v).." "
    end
    return (str.."}")
  else
    return tostring(x)
  end
end

function DumpEdgeFieldToFile(mesh, field, file_name)
  local field_list = mesh.edges[field]:DumpToList()
  local tail_list = mesh.edges.tail:DumpToList()
  local head_list = mesh.edges.head:DumpToList()
  local field_liszt = PN.scriptdir() .. file_name
  local out = io.open(tostring(field_liszt), 'w')
  for i = 1, #field_list do
    out:write( tostring(tail_list[i]) .. "  " ..
               tostring(head_list[i]) .. "  " ..
               fieldToString(field_list[i]) .. "\n" )
  end
  out:close()
end

function DumpVertFieldToFile(mesh, field, file_name)
  local field_list = mesh.vertices[field]:DumpToList()
  local field_liszt = PN.scriptdir() .. file_name
  local out = io.open(tostring(field_liszt), 'w')
  for i = 1, #field_list do
    out:write(fieldToString(field_list[i]) .. "\n" )
  end
  out:close()
end

function DumpDeformationToFile(mesh, file_name)
  local pos = mesh.vertices.pos:DumpToList()
  local d = mesh.vertices.q:DumpToList()
  local field_liszt = PN.scriptdir() .. file_name
  local out = io.open(tostring(field_liszt), 'w')
  for i = 1, #pos do
    out:write(tostring(pos[i][1] + d[i][1]) .. ", " ..
              tostring(pos[i][2] + d[i][2]) .. ", " ..
              tostring(pos[i][3] + d[i][3]) .. "\n" )
  end
  out:close()
end

------------------------------------------------------------------------------
-- Visualize vertex displacements.

local sqrt3 = math.sqrt(3)

local trinorm = liszt function(p0,p1,p2)
  var d1 = p1-p0
  var d2 = p2-p0
  var n  = L.cross(d1,d2)
  var len = L.length(n)
  if len < 1e-10 then len = L.float(1e-10) end
  return n/len
end
local dot_to_color = liszt function(d)
  var val = d * 0.5 + 0.5
  var col = L.vec3f({val,val,val})
  return col
end

local lightdir = L.NewVector(L.float,{sqrt3,sqrt3,sqrt3})

local visualizeDeformation = liszt kernel ( t : mesh.tetrahedra )
  var p0 = t.v[0].pos + t.v[0].q
  var p1 = t.v[1].pos + t.v[1].q
  var p2 = t.v[2].pos + t.v[2].q
  var p3 = t.v[3].pos + t.v[3].q

  var flipped : L.double = 1.0
  if getElementDet(t) < 0 then flipped = -1.0 end

  var d0 = flipped * L.dot(lightdir, trinorm(p1,p2,p3))
  var d1 = flipped * L.dot(lightdir, trinorm(p0,p3,p2))
  var d2 = flipped * L.dot(lightdir, trinorm(p0,p1,p3))
  var d3 = flipped * L.dot(lightdir, trinorm(p1,p0,p2))

  vdb.color(dot_to_color(d0))
  vdb.triangle(p1, p2, p3)
  vdb.color(dot_to_color(d1))
  vdb.triangle(p0, p3, p2)
  vdb.color(dot_to_color(d2))
  vdb.triangle(p0, p1, p3)
  vdb.color(dot_to_color(d3))
  vdb.triangle(p1, p0, p2)
end

function visualize(mesh)
  vdb.vbegin()
  vdb.frame()
  visualizeDeformation(mesh.tetrahedra)
  vdb.vend()
  -- print('Hit enter for next frame')
  io.read()
  -- os.execute("sleep 1")
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

  mesh.edges:NewField('mass', L.double):Load(0)
  local buildMassMatrix = liszt kernel(t : mesh.tetrahedra)
    var tet_vol = fabs(getElementDet(t))/6
    var factor = tet_vol * getElementDensity(t) / 20
    for i = 0,4 do
      for j = 0,4 do
        var mult_const = 1
        if i == j then
          mult_const = 2
        end
        t.e[i, j].mass += factor * mult_const
      end
    end
  end
  buildMassMatrix(mesh.tetrahedra)

  -- any cleanup code we need goes here
end

------------------------------------------------------------------------------
-- Initialize Lame constants. This includes lambda and mu. See
--     libraries/volumetricMesh/volumetricMeshENuMaterial.h
--     (getLambda and getMu)
-- This code is used to initialize Lame constants when stiffnexx matrix/ internal_forces
-- are initialized.

local getLambda = liszt function(t)
  var E : L.double = mesh.E
  var Nu : L.double = mesh.Nu
  return ( (Nu * E) / ( ( 1 + Nu ) * ( 1 - 2 * Nu ) ) )
end

local getMu = liszt function(t)
  var E : L.double = mesh.E
  var Nu : L.double = mesh.Nu
  return ( ( E / ( 2 * ( 1 + Nu) ) ) )
end

local initializeLameConstants = liszt kernel(t : mesh.tetrahedra)
  t.lambdaLame = getLambda(t)
  t.muLame = getMu(t)
end

------------------------------------------------------------------------------
-- For corresponding VEGA code, see
--    libraries/stvk/StVKTetABCD.cpp (most of the file)

-- STORAGE for output from this part of setup
mesh.tetrahedra:NewField('volume', L.double)
mesh.tetrahedra:NewField('Phig', L.mat3x4d)

-- Here, we precompute PhiG which is used to compute and cache dots, and
-- compute A, b, C, and D as required, on a per element basis.
local precomputeStVKIntegrals = liszt kernel(t : mesh.tetrahedra)
  var det = getElementDet(t)
  for i = 0,4 do
    for j = 0,3 do
      var column0 : L.vec3d
      var column1 : L.vec3d
      var countI = 0
      for ii = 0,4 do
        if ii ~= i then
          var countJ = 0
          for jj = 0,3 do
            if jj ~= j then
              if countJ == 0 then
                column0[countI] = t.v[ii].pos[jj]
              else
                column1[countI] = t.v[ii].pos[jj]
              end
              countJ += 1
            end
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
      t.volume = fabs(det) / 6
    end
  end
end

-- The VEGA Code seems to compute the dots matrix once, and then
-- cache it for the duration of a per-tet computation rather than
-- allocate disk space
local tetDots = liszt function(tet)
  var dots : L.mat4d
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

local tetCoeffA = liszt function(tet, dots, i, j)
  var phi = tet.Phig
  var volume = tet.volume
  return ( volume * tensor3( { phi[i, 0], phi[i, 1], phi[i, 2] },
                             { phi[j, 0], phi[j, 1], phi[j, 2] } ) )
end

local tetCoeffB = liszt function(tet, dots, i, j)
  var volume = tet.volume
  return volume * dots[i, j]
end

local tetCoeffC = liszt function(tet, dots, i, j, k)
  var phi = tet.Phig
  var volume = tet.volume
  var res : L.vec3d = volume * dots[j, k] * 
                      { phi[i, 0], phi[i, 1], phi[i, 2] }
  return res
end

local tetCoeffD = liszt function(tet, dots, i, j, k, l)
  var volume = tet.volume
  return ( volume * dots[i, j] * dots[k, l] )
end

------------------------------------------------------------------------------
-- For corresponding VEGA code, see
--    libraries/stvk/StVKInternalinternal_forces.cpp (most of the file)

mesh.vertices:NewField('internal_forces', L.vec3d):Load({0, 0, 0})

-- extra functions supplied by this module
-- Outer loop is generally over all elements (tetrahedra).
-- Result is stored as a 3D vector field over all the vertices.

-- Linear contributions to internal internal_forces
local addIFLinearTerms = liszt kernel(t : mesh.tetrahedra)
  var dots = tetDots(t)
  for ci = 0,4 do
    var c = t.v[ci]
    for ai = 0,4 do
      var qa = t.v[ai].q
      var force = t.lambdaLame *
                  multiplyMatVec3(tetCoeffA(t, dots, ci, ai), qa) +
                  (t.muLame * tetCoeffB(t, dots, ai, ci)) * qa +
                  t.muLame *
                  multiplyMatVec3(tetCoeffA(t, dots, ai, ci), qa)
      c.internal_forces += force
    end
  end
end

-- Quadratic contributions to internal internal_forces
local addIFQuadraticTerms = liszt kernel(t : mesh.tetrahedra)
  var dots = tetDots(t)
  for ci = 0,4 do
    var c = t.v[ci]
    for ai = 0,4 do
      var qa = t.v[ai].q
      for bi = 0,4 do
        var qb = t.v[bi].q
        var dotp = L.dot(qa, qb)
        var forceTerm1 = 0.5 * t.lambdaLame * dotp *
                         tetCoeffC(t, dots, ci, ai, bi) +
                         t.muLame * dotp *
                         tetCoeffC(t, dots, ai, bi, ci)
        var C = t.lambdaLame * tetCoeffC(t, dots, ai, bi, ci) +
                t.muLame * ( tetCoeffC(t, dots, ci, ai, bi) +
                tetCoeffC(t, dots, bi, ai, ci) )
        var dotCqa = L.dot(C, qa)
        c.internal_forces += forceTerm1 + dotCqa * qb
      end
    end
  end
end

-- Cubic contributions to internal internal_forces
local addIFCubicTerms = liszt kernel(t : mesh.tetrahedra)
  var dots = tetDots(t)
  for ci = 0,4 do
    var c = t.v[ci]
    for ai = 0,4 do
      var qa = t.v[ai].q
      for bi = 0,4 do
        var qb = t.v[bi].q
        for di = 0,4 do
          var d = t.v[di]
          var qd = d.q
          var dotp = L.dot(qa, qb)
          var scalar = dotp * ( 0.5 * t.lambdaLame *
                                tetCoeffD(t, dots, ai, bi, ci, di) +
                                t.muLame *
                                tetCoeffD(t, dots, ai, ci, bi, di) )
          c.internal_forces += scalar * qd
        end
      end
    end
  end
end

local resetInternalForces = liszt kernel(v : mesh.vertices)
  v.internal_forces = {0, 0, 0}
end

--[[
-- We probably do not need energy contribution.
function computeEnergyContribution()
end
function computeEnergy()
  computeEnergyContribution()
end
]]

local computeInternalForcesHelper = function(tetrahedra)
  -- print("Computing linear contributions to internal_forces ...")
  addIFLinearTerms(tetrahedra)
  -- print("Computing quadratic contributions to internal_forces ...")
  addIFQuadraticTerms(tetrahedra)
  -- print("Computing cubic contributions to internal_forces ...")
  addIFCubicTerms(tetrahedra)
end

function computeInternalForces(mesh)
  -- print("Computing internal_forces ...")
  resetInternalForces(mesh.vertices)
  computeInternalForcesHelper(mesh.tetrahedra)
  -- print("Completed computing internal_forces.")
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
  for ci = 0,4 do
    for ai = 0,4 do
      var mat = t.muLame * tetCoeffB(t, dots, ai, ci) * getId3()
      mat += (t.lambdaLame * tetCoeffA(t, dots, ci, ai) +
             (t.muLame * tetCoeffA(t, dots, ai, ci)))
      t.e[ci, ai].stiffness += mat
    end
  end
end

-- Quadratic contributions to stiffness matrix
local addStiffQuadraticTerms = liszt kernel(t : mesh.tetrahedra)
  var dots = tetDots(t)
  for ci = 0,4 do
    for ai = 0,4 do
      var qa = t.v[ai].q
      var mat : L.mat3d = constantMatrix3(0)
      for ei = 0,4 do
        var c0v = t.lambdaLame * tetCoeffC(t, dots, ci, ai, ei) +
                  t.muLame * ( tetCoeffC(t, dots, ei, ai, ci) +
                               tetCoeffC(t, dots, ai, ei, ci) )
        mat += tensor3(qa, c0v)
        var c1v = t.lambdaLame * tetCoeffC(t, dots, ei, ai, ci) +
                  t.muLame * ( tetCoeffC(t, dots, ci, ei, ai) +
                               tetCoeffC(t, dots, ai, ei, ci) )
        mat += tensor3(qa, c1v)
        var c2v = t.lambdaLame * tetCoeffC(t, dots, ai, ei, ci) +
                  t.muLame * ( tetCoeffC(t, dots, ci, ai, ei) +
                               tetCoeffC(t, dots, ei, ai, ci) )
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
  for ci = 0,4 do
    for ei = 0,4 do
      var mat : L.mat3d = constantMatrix3(0)
      for ai = 0,4 do
        var qa = t.v[ai].q
        for bi = 0,4 do
          var qb = t.v[bi].q
          var d0 = t.lambdaLame * tetCoeffD(t, dots, ai, ci, bi, ei) +
                   t.muLame * ( tetCoeffD(t, dots, ai, ei, bi, ci) +
                                tetCoeffD(t, dots, ai, bi, ci, ei) )
          mat += d0 * (tensor3(qa, qb))
          var d1 = 0.5 * t.lambdaLame *
                   tetCoeffD(t, dots, ai, bi, ci, ei) +
                   t.muLame * tetCoeffD(t, dots, ai, ci, bi, ei)
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
  -- print("Computing linear contributions to stiffness matrix ...")
  addStiffLinearTerms(tetrahedra)
  -- print("Computing quadratic contributions to stiffness matrix ...")
  addStiffQuadraticTerms(tetrahedra)
  -- print("Computing cubic contributions to stiffness matrix ...")
  addStiffCubicTerms(tetrahedra)
end

function computeStiffnessMatrix(mesh)
  -- print("Computing stiffness matrix ...")
  resetStiffnessMatrix(mesh.edges)
  computeStiffnessMatrixHelper(mesh.tetrahedra)
  -- print("Completed computing stiffness matrix.")
end

------------------------------------------------------------------------------

-- TODO: Again Data Model and skeleton needs to be filled in more here...

local ImplicitBackwardEulerIntegrator = {}
ImplicitBackwardEulerIntegrator.__index = ImplicitBackwardEulerIntegrator

function ImplicitBackwardEulerIntegrator.New(opts)
  local stepper = setmetatable({
    internalForcesScalingFactor  = opts.internalForcesScalingFactor,
    epsilon                     = opts.epsilon,
    timestep                    = opts.timestep,
    dampingMassCoef             = opts.dampingMassCoef,
    dampingStiffnessCoef        = opts.dampingStiffnessCoef,
    maxIterations               = opts.maxIterations,
    cgEpsilon                   = opts.cgEpsilon,
    cgMaxIterations             = opts.cgMaxIterations
  }, ImplicitBackwardEulerIntegrator)

  return stepper
end

function genMultiplySparseMatrixVector(gmesh, matrix, x, res)
  return liszt kernel(v : gmesh.vertices)
    for e in v.edges do
      v.[res] += e.[matrix] * v.[x]
    end
  end
end

function ImplicitBackwardEulerIntegrator:setupFieldsKernels(mesh)

  mesh.vertices:NewField('q_1', L.vec3d):Load({ 0, 0, 0 })
  mesh.vertices:NewField('qvel_1', L.vec3d):Load({ 0, 0, 0 })
  mesh.vertices:NewField('qaccel_1', L.vec3d):Load({ 0, 0, 0 })
  mesh.vertices:NewField('qresidual', L.vec3d):Load({ 0, 0, 0 })
  mesh.vertices:NewField('qdelta', L.vec3d):Load({ 0, 0, 0 })

  mesh.vertices:NewField('precond', L.vec3d):Load({ 0, 0, 0 })
  mesh.vertices:NewField('x', L.vec3d):Load({ 0, 0, 0 })
  mesh.vertices:NewField('r', L.vec3d):Load({ 0, 0, 0 })
  mesh.vertices:NewField('z', L.vec3d):Load({ 0, 0, 0 })
  mesh.vertices:NewField('p', L.vec3d):Load({ 0, 0, 0 })
  mesh.vertices:NewField('Ap', L.vec3d):Load({ 0, 0, 0 })

  mesh.edges:NewField('raydamp', L.mat3d)

  self.err = L.NewGlobal(L.double, 0)
  self.normRes = L.NewGlobal(L.double, 0)
  self.alphaDenom = L.NewGlobal(L.double, 0)
  self.alpha = L.NewGlobal(L.double, 0)
  self.beta = L.NewGlobal(L.double, 0)

  self.initializeQFields = liszt kernel(v : mesh.vertices)
    v.q_1 = v.q
    v.qvel_1 = v.qvel
    v.qaccel = { 0, 0, 0 }
    v.qaccel_1 = { 0, 0, 0 }
  end

  self.initializeqdelta = liszt kernel(v : mesh.vertices)
    v.qdelta = v.qresidual
  end

  self.scaleInternalForces = liszt kernel(v : mesh.vertices)
    v.internal_forces = self.internalForcesScalingFactor * v.internal_forces
  end

  self.scaleStiffnessMatrix = liszt kernel(e : mesh.edges)
    e.stiffness = self.internalForcesScalingFactor * e.stiffness
  end

  self.createRayleighDampMatrix = liszt kernel(e : mesh.edges)
    e.raydamp = self.dampingStiffnessCoef * e.stiffness +
                self.dampingMassCoef * e.mass * getId3()
  end

  self.updateqresidual1 = liszt kernel(v : mesh.vertices)
    for e in v.edges do
      v.qresidual += multiplyMatVec3(e.stiffness, (e.head.q_1 - e.head.q))
    end
  end

  self.updateqresidual2 = liszt kernel(v : mesh.vertices)
    for e in v.edges do
      v.qresidual += multiplyMatVec3(e.stiffness, e.head.qvel)
    end
  end

  self.updateqresidual3 = liszt kernel(v : mesh.vertices)
    v.qresidual += (v.internal_forces - v.external_forces)
    v.qresidual = - ( self.timestep * v.qresidual )
  end

  self.updateqresidual4 = liszt kernel(v : mesh.vertices)
    for e in v.edges do
      v.qresidual += e.mass * (e.head.qvel_1 - e.head.qvel)
    end
  end

  self.updateStiffness1 = liszt kernel(e : mesh.edges)
    e.stiffness = self.timestep * e.stiffness
    e.stiffness += e.raydamp
    -- TODO: This damping matrix seems to be zero unless set otherwise.
    -- e.stiffness = e.stiffness + e.dampingmatrix
  end

  self.updateStiffness11 = liszt kernel(e : mesh.edges)
    e.stiffness = self.timestep * e.stiffness
  end

  self.updateStiffness12 = liszt kernel(e : mesh.edges)
    e.stiffness += e.raydamp
  end

  self.updateStiffness2 = liszt kernel(e : mesh.edges)
    e.stiffness = self.timestep * e.stiffness
    e.stiffness += e.mass * getId3()
  end

  self.getError = liszt kernel(v : mesh.vertices)
    self.err += L.dot(v.qdelta, v.qdelta)
  end

  self.pcgCalculatePreconditioner = liszt kernel(v : mesh.vertices)
    var stiff = v.diag.stiffness
    var diag = { stiff[0,0], stiff[1,1], stiff[2,2] }
    v.precond = { 1/diag[0], 1/diag[1], 1/diag[2] }
  end

  self.pcgCalculateExactResidual = liszt kernel(v : mesh.vertices)
    v.r = { 0, 0, 0 }
    for e in v.edges do
      v.r += multiplyMatVec3(e.stiffness, e.head.x)
    end
    v.r = v.qdelta - v.r
  end

  self.pcgCalculateNormResidual = liszt kernel(v : mesh.vertices)
    self.normRes += L.dot(multiplyVectors(v.r, v.precond), v.r)
  end

  self.pcgInitialize = liszt kernel(v : mesh.vertices)
    v.p = multiplyVectors(v.r, v.precond)
  end

  self.pcgComputeAlphaDenom = liszt kernel(v : mesh.vertices)
    v.Ap = { 0, 0, 0 }
    for e in v.edges do
      v.Ap += multiplyMatVec3(e.stiffness, e.head.p)
    end
    self.alphaDenom += L.dot(v.p, v.Ap)
  end

  self.pcgUpdateX = liszt kernel(v : mesh.vertices)
    v.x += self.alpha * v.p
  end

  self.pcgUpdateResidual = liszt kernel(v : mesh.vertices)
    v.r -= self.alpha * v.Ap
  end

  self.pcgUpdateP = liszt kernel(v : mesh.vertices)
    v.p = self.beta * v.p + multiplyVectors(v.precond, v.r)
  end

  self.updateAfterSolve = liszt kernel(v : mesh.vertices)
    v.qdelta = v.x
    v.qvel += v.qdelta
    -- TODO: subtracting q from q?
    -- q += q_1-q + self.timestep * qvel
    v.q = v.q_1 + self.timestep * v.qvel
  end

end

-- The following pcg solver uses Jacobi preconditioner, as implemented in Vega.
-- It uses the same algorithm as Vega (exact residual on 30th iteration). But
-- the symbol names are kept to match the pseudo code on Wikipedia for clarity.
function ImplicitBackwardEulerIntegrator:solvePCG(mesh)
  mesh.vertices.x:Load({ 0, 0, 0 })
  local tss = terralib.currenttimeinseconds()
  self.pcgCalculatePreconditioner(mesh.vertices)
  self.pcgCalculateExactResidual(mesh.vertices)
  self.normRes:set(0)
  self.pcgCalculateNormResidual(mesh.vertices)
  self.pcgInitialize(mesh.vertices)
  local iter = 1
  local normRes = self.normRes:get()
  while normRes > self.cgEpsilon * self.cgEpsilon and
        iter <= self.cgMaxIterations do
    -- print("PCG iteration "..iter)
    self.alphaDenom:set(0)
    self.pcgComputeAlphaDenom(mesh.vertices)
    self.alpha:set( normRes / self.alphaDenom:get() )
    self.pcgUpdateX(mesh.vertices)
    if iter % 30 == 0 then
      self.pcgCalculateExactResidual(mesh.vertices)
    else
      self.pcgUpdateResidual(mesh.vertices)
    end
    local normResOld = normRes
    self.normRes:set(0)
    self.pcgCalculateNormResidual(mesh.vertices)
    normRes = self.normRes:get()
    self.beta:set( normRes / normResOld )
    self.pcgUpdateP(mesh.vertices)
    iter = iter + 1
  end
  local tse = terralib.currenttimeinseconds()
  print("Time for solver is "..((tse-tss)*1E6).." us")
end

function ImplicitBackwardEulerIntegrator:doTimestep(mesh)
  local tds = terralib.currenttimeinseconds()
  local err0 = 0 -- L.Global?
  local errQuotient

  -- store current amplitudes and set initial gues for qaccel, qvel
  -- AHHHHH, THE BELOW CAN BE IMPLEMENTED USING COPIES
  -- LOOP OVER THE STATE VECTOR (vertices for now)
  self.initializeQFields(mesh.vertices)

  -- Limit our total number of iterations allowed per timestep
  for numIter = 1, self.maxIterations do
    -- print("#dotimestep iteration = "..numIter.." ...")
    -- ASSEMBLY
    -- TIMING START
    local tfs = terralib.currenttimeinseconds()
    computeInternalForces(mesh)
    computeStiffnessMatrix(mesh)
    local tfe = terralib.currenttimeinseconds()
    print("Time to assemble force and stiffness is "..((tfe-tfs)*1E6).." us")
    -- TIMING END
    -- RECORD ASSEMBLY TIME

    -- NOTE: NOTE: We can implement these scalings as separate kernels to be
    -- "FAIR" to VEGA or we can fold it into other kernels.
    -- Kinda unclear which to do
    -- NOTE TODO: scaleinternalinternal_forces over-writes internal_forces. we may want to change
    -- this.
    self.scaleInternalForces(mesh.vertices)
    self.scaleStiffnessMatrix(mesh.edges)

    -- ZERO out the residual field
    mesh.vertices.qresidual:Load({ 0, 0, 0 })

    -- NOTE: useStaticSolver == FALSE
    --    We just assume this everywhere
    self.createRayleighDampMatrix(mesh.edges)

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
      self.updateqresidual1(mesh.vertices)
    end

    -- some magic incantations corresponding to the above
    self.updateStiffness11(mesh.edges)
    self.updateStiffness12(mesh.edges)
    self.updateqresidual2(mesh.vertices)
    self.updateStiffness2(mesh.edges)

    -- ADD EXTERNAL/INTERNAL internal_forces
    self.updateqresidual3(mesh.vertices)

    -- superfluous on iteration 1, but safe to run
    if numIter ~= 1 then
      self.updateqresidual4(mesh.vertices)
    end

    -- TODO: this should be a copy and not a separate kernel in the end
    self.initializeqdelta(mesh.vertices)

    -- TODO: This code doesn't have any way of handling fixed vertices
    -- at the moment.  Should enforce that here somehow
    self.err:set(0)
    self.getError(mesh.vertices)

    -- compute initial error on the 1st iteration
    if numIter == 1 then
      err0 = self.err:get()
      errQuotient = 1
    else
      errQuotient = self.err:get() / err0
    end

    if errQuotient < self.epsilon*self.epsilon or
      err0 < self.epsilon*self.epsilon then
      break
    end

    -- SYSTEM SOLVE: SYSTEM_MATRIX * BUFFER = BUFFER_CONSTRAINED
    -- START PERFORMANCE TIMING
    -- ASSUMING PCG FOR NOW
    --[[
    if err_code ~= 0 then
      error("ERROR: PCG sparse solver "..
            "returned non-zero exit status "..err_code)
    end
    ]]

    self:solvePCG(mesh)

    -- STOP PERFORMANCE TIMING
    -- RECORD SYSTEM SOLVE TIME

    -- Reinsert the rows?

    self.updateAfterSolve(mesh.vertices)

    -- Constrain (zero) fields for the subset of constrained vertices
  end
  local tde = terralib.currenttimeinseconds()
  print("DoTimeStep time is "..((tde-tds)*1E6).." us")
end

------------------------------------------------------------------------------

function clearExternalForces(mesh)
  mesh.vertices.external_forces:Load({ 0, 0, 0 })
end

local setExternalForces = liszt kernel(v : mesh.vertices)
  var pos = v.pos + v.q
  v.external_forces = { -1000*(pos[0]), 2000, 0 }
end

local s = mesh:nVerts()/2
function setConstantForce(mesh)
  mesh.vertices.external_forces:Load( { 1000.0, 1000.0, 0 } )
end

function setExternalConditions(mesh, iter)
  -- setExternalForces(mesh.vertices)
  if iter == 1 then
    setConstantForce(mesh)
  end
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

  -- print("Computing mass matrix ...")
  computeMassMatrix(volumetric_mesh)

  -- print("Computing integrals ...")
  precomputeStVKIntegrals(mesh.tetrahedra) -- called StVKElementABCDLoader:load() in ref
  -- print("Precomputed integrals")

  -- print("Initializing Lame constants ...")
  initializeLameConstants(mesh.tetrahedra)

  local integrator = ImplicitBackwardEulerIntegrator.New{
    n_vars                = 3*nvertices,
    timestep              = options.timestep,
    positiveDefinite      = 0,
    nFixedDOFs            = 0,
    dampingMassCoef       = options.dampingMassCoef,
    dampingStiffnessCoef  = options.dampingStiffnessCoef,
    maxIterations         = options.maxIterations,
    epsilon               = options.epsilon,
    cgEpsilon             = options.cgEpsilon,
    cgMaxIterations       = options.cgMaxIterations,
    internalForcesScalingFactor  = options.deformableObjectCompliance
  }
  integrator:setupFieldsKernels(mesh)

  -- print("Performing time steps ...")
  -- visualize(volumetric_mesh)
  DumpDeformationToFile(volumetric_mesh, "out/mesh_liszt_"..tostring(0))
  for i=1,options.numTimesteps do
    -- print("#timestep = "..i)
    local tts = terralib.currenttimeinseconds()
    setExternalConditions(volumetric_mesh, i)
    integrator:doTimestep(volumetric_mesh)
    local tte = terralib.currenttimeinseconds()
    print("Time for step "..i.." is "..((tte-tts)*1E6).." us")
    print("")
    -- visualize(volumetric_mesh)
    DumpDeformationToFile(volumetric_mesh, "out/mesh_liszt_"..tostring(i))
  end

  -- read out the state here somehow?
end

main()
