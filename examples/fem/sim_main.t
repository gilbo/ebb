import 'compiler.liszt'
local vdb = L.require 'lib.vdb'
-- L.default_processor = L.GPU

local Tetmesh = L.require 'examples.fem.tetmesh'
local VEGFileIO = L.require 'examples.fem.vegfileio'
local PN = L.require 'lib.pathname'

-- print("Loading mesh ...")
local turtle = VEGFileIO.LoadTetmesh
  'examples/fem/turtle-volumetric-homogeneous.veg'

-- local I = terralib.includecstring([[
-- #include "cuda_profiler_api.h"
-- ]])

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
-- Timer for timing execution time

local Timer = {}
Timer.__index = Timer

function Timer.New()
  local timer = { start = 0, finish = 0, total = 0 }
  setmetatable(timer, Timer)
  return timer
end

function Timer:Reset()
  self.start = 0
  self.finish = 0
  self.total = 0
end

function Timer:Start()
  self.start = terralib.currenttimeinseconds()
end

function Timer:Pause()
  self.finish = terralib.currenttimeinseconds()
  self.total = self.total + self.finish - self.start
  self.start = 0
end

function Timer:Stop()
  self.finish = terralib.currenttimeinseconds()
  local total = self.total + self.finish - self.start
  self.start = 0
  self.finish = 0
  self.total = 0
  return total
end

function Timer:GetTime()
  return self.total
end

------------------------------------------------------------------------------
-- Helper functions, kernels, variables etc

-- Compute absolute value for a given variable
local liszt function fabs(num)
  var result = num
  if num < 0 then result = -num end
  return result
end

-- Compute determinant for matrix formed by vertex positions
local liszt function getElementDet(t)
  var a = t.v[0].pos
  var b = t.v[1].pos
  var c = t.v[2].pos
  var d = t.v[3].pos
  return (L.dot(a - d, L.cross(b - d, c - d)))
end

-- Get element density for a mesh element (tetreahedron)
local liszt function getElementDensity(a)
  return L.double(mesh.density)
end

-- Identity matrix
local liszt function getId3()
  return { { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 } }
end

-- Matrix with all entries equal to value v
local liszt function constantMatrix3(v)
  return { { v, v, v }, { v, v, v }, { v, v, v } }
end

-- Tensor product of 2 vectors
local liszt function tensor3(a, b)
  var result = { { a[0] * b[0], a[0] * b[1], a[0] * b[2] },
                 { a[1] * b[0], a[1] * b[1], a[1] * b[2] },
                 { a[2] * b[0], a[2] * b[1], a[2] * b[2] } }
  return result
end

-- Matrix vector product
local liszt function multiplyMatVec3(M, x)
  var y = { M[0, 0]*x[0] + M[0, 1]*x[1] + M[0, 2]*x[2],
            M[1, 0]*x[0] + M[1, 1]*x[1] + M[1, 2]*x[2],
            M[2, 0]*x[0] + M[2, 1]*x[1] + M[2, 2]*x[2] }
  return y
end
local liszt function multiplyVectors(x, y)
  return { x[0]*y[0], x[1]*y[1], x[2]*y[2]  }
end

-- Diagonals
local liszt function diagonalMatrix(a)
    return { { a, 0, 0 }, { 0, a, 0 }, { 0, 0, a } }
end

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

local liszt function trinorm(p0,p1,p2)
  var d1 = p1-p0
  var d2 = p2-p0
  var n  = L.cross(d1,d2)
  var len = L.length(n)
  if len < 1e-10 then len = L.float(1e-10) end
  return n/len
end
local liszt function dot_to_color(d)
  var val = d * 0.5 + 0.5
  var col = L.vec3f({val,val,val})
  return col
end

local lightdir = L.NewVector(L.float,{sqrt3,sqrt3,sqrt3})

local liszt kernel visualizeDeformation ( t : mesh.tetrahedra )
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
  local liszt kernel buildMassMatrix (t : mesh.tetrahedra)
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
end

------------------------------------------------------------------------------
-- Initialize Lame constants. This includes lambda and mu. See
--     libraries/volumetricMesh/volumetricMeshENuMaterial.h
--     (getLambda and getMu)
-- This code is used to initialize Lame constants when stiffnexx matrix/ internal_forces
-- are initialized.

local liszt function getLambda(t)
  var E : L.double = mesh.E
  var Nu : L.double = mesh.Nu
  return ( (Nu * E) / ( ( 1 + Nu ) * ( 1 - 2 * Nu ) ) )
end

local liszt function getMu(t)
  var E : L.double = mesh.E
  var Nu : L.double = mesh.Nu
  return ( ( E / ( 2 * ( 1 + Nu) ) ) )
end

local liszt kernel initializeLameConstants (t : mesh.tetrahedra)
  t.lambdaLame = getLambda(t)
  t.muLame = getMu(t)
end

------------------------------------------------------------------------------
-- For corresponding VEGA code, see
--    libraries/stvk/StVKTetABCD.cpp (most of the file)

-- STORAGE for output from this part of setup
mesh.tetrahedra:NewField('volume', L.double)
mesh.tetrahedra:NewField('Phig', L.mat4x3d)

-- Here, we precompute PhiG which is used to compute and cache dots, and
-- compute A, b, C, and D as required, on a per element basis.
local liszt kernel precomputeStVKIntegrals (t : mesh.tetrahedra)
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
local liszt function tetDots(tet)
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

local liszt function tetCoeffA(tet, dots, i, j)
  var phi = tet.Phig
  var volume = tet.volume
  return ( volume * tensor3( { phi[i, 0], phi[i, 1], phi[i, 2] },
                             { phi[j, 0], phi[j, 1], phi[j, 2] } ) )
end

local liszt function tetCoeffB(tet, dots, i, j)
  var volume = tet.volume
  return volume * dots[i, j]
end

local liszt function tetCoeffC(tet, dots, i, j, k)
  var phi = tet.Phig
  var volume = tet.volume
  var res : L.vec3d = volume * dots[j, k] * 
                      { phi[i, 0], phi[i, 1], phi[i, 2] }
  return res
end

local liszt function tetCoeffD(tet, dots, i, j, k, l)
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
local liszt kernel addIFLinearTerms (t : mesh.tetrahedra)
  var dots = tetDots(t)
  var lambda = t.lambdaLame
  var mu = t.muLame
  for ci = 0,4 do
    var c = t.v[ci]
    for ai = 0,4 do
      var qa = t.v[ai].q
      var Aca = tetCoeffA(t, dots, ci, ai)
      var Aac = tetCoeffA(t, dots, ai, ci)
      var force = lambda *
                  multiplyMatVec3(Aca, qa) +
                  (mu * tetCoeffB(t, dots, ai, ci)) * qa +
                  mu *
                  multiplyMatVec3(Aac, qa)
      c.internal_forces += force
    end
  end
end

-- Quadratic contributions to internal internal_forces
local liszt kernel addIFQuadraticTerms (t : mesh.tetrahedra)
  var dots = tetDots(t)
  var lambda = t.lambdaLame
  var mu = t.muLame
  for ci = 0,4 do
    var c = t.v[ci]
    for ai = 0,4 do
      var qa = t.v[ai].q
      for bi = 0,4 do
        var qb = t.v[bi].q
        var dotp = L.dot(qa, qb)
        var forceTerm1 = 0.5 * lambda * dotp *
                         tetCoeffC(t, dots, ci, ai, bi) +
                         mu * dotp *
                         tetCoeffC(t, dots, ai, bi, ci)
        var C = lambda * tetCoeffC(t, dots, ai, bi, ci) +
                mu * ( tetCoeffC(t, dots, ci, ai, bi) +
                tetCoeffC(t, dots, bi, ai, ci) )
        var dotCqa = L.dot(C, qa)
        c.internal_forces += forceTerm1 + dotCqa * qb
      end
    end
  end
end

-- Cubic contributions to internal internal_forces
local liszt kernel addIFCubicTerms (t : mesh.tetrahedra)
  var dots = tetDots(t)
  var lambda = t.lambdaLame
  var mu = t.muLame
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
          var scalar = dotp * ( 0.5 * lambda *
                                tetCoeffD(t, dots, ai, bi, ci, di) +
                                mu *
                                tetCoeffD(t, dots, ai, ci, bi, di) )
          c.internal_forces += scalar * qd
        end
      end
    end
  end
end

local liszt kernel resetInternalForces (v : mesh.vertices)
  v.internal_forces = {0, 0, 0}
end

local computeInternalForcesHelper = function(tetrahedra)
  local timer = Timer.New()
  -- print("Computing linear contributions to internal_forces ...")
  timer:Start()
  addIFLinearTerms(tetrahedra)
  print("Time for linear terms is "..(timer:Stop()*1E6).." us")
  -- print("Computing quadratic contributions to internal_forces ...")
  timer:Start()
  addIFQuadraticTerms(tetrahedra)
  print("Time for quadratic terms is "..(timer:Stop()*1E6).." us")
  -- print("Computing cubic contributions to internal_forces ...")
  timer:Start()
  addIFCubicTerms(tetrahedra)
  print("Time for cubic terms is "..(timer:Stop()*1E6).." us")
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
local liszt kernel addStiffLinearTerms (t : mesh.tetrahedra)
  var dots = tetDots(t)
  var lambda = t.lambdaLame
  var mu = t.muLame
  for ci = 0,4 do
    for ai = 0,4 do
      var mat = diagonalMatrix(mu * tetCoeffB(t, dots, ai, ci))
      var Aca = tetCoeffA(t, dots, ci, ai)
      var Aac = tetCoeffA(t, dots, ai, ci)
      mat += (lambda * Aca + (mu * Aac))
      t.e[ci, ai].stiffness += mat
    end
  end
end

-- Quadratic contributions to stiffness matrix
local liszt kernel addStiffQuadraticTerms (t : mesh.tetrahedra)
  var dots = tetDots(t)
  var lambda = t.lambdaLame
  var mu = t.muLame
  for ci = 0,4 do
    for ai = 0,4 do
      var qa = t.v[ai].q
      var mat : L.mat3d = constantMatrix3(0)
      for ei = 0,4 do
        var c0v = lambda * tetCoeffC(t, dots, ci, ai, ei) +
                  mu * ( tetCoeffC(t, dots, ei, ai, ci) +
                               tetCoeffC(t, dots, ai, ei, ci) )
        mat += tensor3(qa, c0v)
        var c1v = lambda * tetCoeffC(t, dots, ei, ai, ci) +
                  mu * ( tetCoeffC(t, dots, ci, ei, ai) +
                               tetCoeffC(t, dots, ai, ei, ci) )
        mat += tensor3(qa, c1v)
        var c2v = lambda * tetCoeffC(t, dots, ai, ei, ci) +
                  mu * ( tetCoeffC(t, dots, ci, ai, ei) +
                               tetCoeffC(t, dots, ei, ai, ci) )
        var dotp = L.dot(qa, c2v)
        mat += diagonalMatrix(dotp)
      end
      t.e[ci, ai].stiffness += mat
    end
  end
end

-- Cubic contributions to stiffness matrix
local liszt kernel addStiffCubicTerms (t : mesh.tetrahedra)
  var dots = tetDots(t)
  var lambda = t.lambdaLame
  var mu = t.muLame
  for ci = 0,4 do
    for ei = 0,4 do
      var mat : L.mat3d = constantMatrix3(0)
      for ai = 0,4 do
        var qa = t.v[ai].q
        for bi = 0,4 do
          var qb = t.v[bi].q
          var d0 = lambda * tetCoeffD(t, dots, ai, ci, bi, ei) +
                   mu * ( tetCoeffD(t, dots, ai, ei, bi, ci) +
                                tetCoeffD(t, dots, ai, bi, ci, ei) )
          mat += d0 * (tensor3(qa, qb))
          var d1 = 0.5 * lambda *
                   tetCoeffD(t, dots, ai, bi, ci, ei) +
                   mu * tetCoeffD(t, dots, ai, ci, bi, ei)
          var dotpd = d1 * L.dot(qa, qb)
          mat += diagonalMatrix(dotpd)
        end
      end
      t.e[ci, ei].stiffness += mat
    end
  end
end

local liszt kernel resetStiffnessMatrix (e : mesh.edges)
  e.stiffness = constantMatrix3(0)
end

local computeStiffnessMatrixHelper = function(tetrahedra)
  local timer = Timer.New()
  -- print("Computing linear contributions to stiffness matrix ...")
  timer:Start()
  addStiffLinearTerms(tetrahedra)
  print("Time for linear terms is "..(timer:Stop()*1E6).." us")
  -- print("Computing quadratic contributions to stiffness matrix ...")
  timer:Start()
  addStiffQuadraticTerms(tetrahedra)
  print("Time for quadratic terms is "..(timer:Stop()*1E6).." us")
  -- print("Computing quadratic contributions to stiffness matrix ...")
  -- print("Computing cubic contributions to stiffness matrix ...")
  timer:Start()
  addStiffCubicTerms(tetrahedra)
  print("Time for cubic terms is "..(timer:Stop()*1E6).." us")
  -- print("Computing quadratic contributions to stiffness matrix ...")
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
  mesh.vertices:NewField('dummy', L.vec3d):Load({ 0, 0, 0 })

  mesh.edges:NewField('raydamp', L.mat3d)

  self.err = L.NewGlobal(L.double, 0)
  self.normRes = L.NewGlobal(L.double, 0)
  self.alphaDenom = L.NewGlobal(L.double, 0)
  self.alpha = L.NewGlobal(L.double, 0)
  self.beta = L.NewGlobal(L.double, 0)

  local liszt kernel initializeQFields (v : mesh.vertices)
    v.q_1 = v.q
    v.qvel_1 = v.qvel
    v.qaccel = { 0, 0, 0 }
    v.qaccel_1 = { 0, 0, 0 }
  end
  self.initializeQFields = initializeQFields

  local liszt kernel initializeqdelta (v : mesh.vertices)
    v.qdelta = v.qresidual
  end
  self.initializeqdelta = initializeqdelta

  local liszt kernel scaleInternalForces (v : mesh.vertices)
    v.internal_forces = self.internalForcesScalingFactor * v.internal_forces
  end
  self.scaleInternalForces = scaleInternalForces

  local liszt kernel scaleStiffnessMatrix (e : mesh.edges)
    e.stiffness = self.internalForcesScalingFactor * e.stiffness
  end
  self.scaleStiffnessMatrix = scaleStiffnessMatrix

  local liszt kernel createRayleighDampMatrix (e : mesh.edges)
    e.raydamp = self.dampingStiffnessCoef * e.stiffness +
                diagonalMatrix(self.dampingMassCoef * e.mass)
  end
  self.createRayleighDampMatrix = createRayleighDampMatrix

  local liszt kernel updateqresidual1 (v : mesh.vertices)
    for e in v.edges do
      v.qresidual += multiplyMatVec3(e.stiffness, (e.head.q_1 - e.head.q))
    end
  end
  self.updateqresidual1 = updateqresidual1

  local liszt kernel updateqresidual2 (v : mesh.vertices)
    for e in v.edges do
      v.qresidual += multiplyMatVec3(e.stiffness, e.head.qvel)
    end
  end
  self.updateqresidual2 = updateqresidual2

  local liszt kernel updateqresidual3 (v : mesh.vertices)
    v.qresidual += (v.internal_forces - v.external_forces)
    v.qresidual = - ( self.timestep * v.qresidual )
  end
  self.updateqresidual3 = updateqresidual3

  local liszt kernel updateqresidual4 (v : mesh.vertices)
    for e in v.edges do
      v.qresidual += e.mass * (e.head.qvel_1 - e.head.qvel)
    end
  end
  self.updateqresidual4 = updateqresidual4

  local liszt kernel updateStiffness1 (e : mesh.edges)
    e.stiffness = self.timestep * e.stiffness
    e.stiffness += e.raydamp
    -- TODO: This damping matrix seems to be zero unless set otherwise.
    -- e.stiffness = e.stiffness + e.dampingmatrix
  end
  self.updateStiffness1 = updateStiffness1

  local liszt kernel updateStiffness11 (e : mesh.edges)
    e.stiffness = self.timestep * e.stiffness
  end
  self.updateStiffness11 = updateStiffness11

  local liszt kernel updateStiffness12 (e : mesh.edges)
    e.stiffness += e.raydamp
  end
  self.updateStiffness12 = updateStiffness12

  local liszt kernel updateStiffness2 (e : mesh.edges)
    e.stiffness = self.timestep * e.stiffness
    e.stiffness += diagonalMatrix(e.mass)
  end
  self.updateStiffness2 = updateStiffness2

  local liszt kernel getError (v : mesh.vertices)
    self.err += L.dot(v.qdelta, v.qdelta)
  end
  self.getError = getError

  local liszt kernel pcgCalculatePreconditioner (v : mesh.vertices)
    var stiff = v.diag.stiffness
    var diag = { stiff[0,0], stiff[1,1], stiff[2,2] }
    v.precond = { 1.0/diag[0], 1.0/diag[1], 1.0/diag[2] }
  end
  self.pcgCalculatePreconditioner = pcgCalculatePreconditioner

  local liszt kernel pcgCalculateExactResidual (v : mesh.vertices)
    v.r = { 0, 0, 0 }
    for e in v.edges do
      v.r += multiplyMatVec3(e.stiffness, e.head.x)
    end
    v.r = v.qdelta - v.r
  end
  self.pcgCalculateExactResidual = pcgCalculateExactResidual

  local liszt kernel pcgCalculateNormResidual (v : mesh.vertices)
    self.normRes += L.dot(multiplyVectors(v.r, v.precond), v.r)
  end
  self.pcgCalculateNormResidual = pcgCalculateNormResidual

  local liszt kernel pcgInitialize (v : mesh.vertices)
    v.p = multiplyVectors(v.r, v.precond)
  end
  self.pcgInitialize = pcgInitialize

  local liszt kernel pcgComputeAp (v : mesh.vertices)
    v.Ap = { 0, 0, 0 }
    for e in v.edges do
      v.Ap += multiplyMatVec3(e.stiffness, e.head.p)
    end
  end
  self.pcgComputeAp = pcgComputeAp

  local liszt kernel pcgComputeDummy (v : mesh.vertices)
    v.dummy = { 0, 0, 0 }
    for e in v.edges do
      v.dummy += e.head.p
    end
  end
  self.pcgComputeDummy = pcgComputeDummy

  local liszt kernel pcgComputeAlphaDenom (v : mesh.vertices)
    self.alphaDenom += L.dot(v.p, v.Ap)
  end
  self.pcgComputeAlphaDenom = pcgComputeAlphaDenom

  local liszt kernel pcgUpdateX (v : mesh.vertices)
    v.x += self.alpha * v.p
  end
  self.pcgUpdateX = pcgUpdateX

  local liszt kernel pcgUpdateResidual (v : mesh.vertices)
    v.r -= self.alpha * v.Ap
  end
  self.pcgUpdateResidual = pcgUpdateResidual

  local liszt kernel pcgUpdateP (v : mesh.vertices)
    v.p = self.beta * v.p + multiplyVectors(v.precond, v.r)
  end
  self.pcgUpdateP = pcgUpdateP

  local liszt kernel updateAfterSolve (v : mesh.vertices)
    v.qdelta = v.x
    v.qvel += v.qdelta
    -- TODO: subtracting q from q?
    -- q += q_1-q + self.timestep * qvel
    v.q = v.q_1 + self.timestep * v.qvel
  end
  self.updateAfterSolve = updateAfterSolve

end

-- The following pcg solver uses Jacobi preconditioner, as implemented in Vega.
-- It uses the same algorithm as Vega (exact residual on 30th iteration). But
-- the symbol names are kept to match the pseudo code on Wikipedia for clarity.
function ImplicitBackwardEulerIntegrator:solvePCG(mesh)
--  I.cudaProfilerStart()
  local timer_total = Timer.New()
  timer_total:Start()
  mesh.vertices.x:Load({ 0, 0, 0 })
  self.pcgCalculatePreconditioner(mesh.vertices)
  local iter = 1
  self.pcgCalculateExactResidual(mesh.vertices)
  self.normRes:set(0)
  self.pcgInitialize(mesh.vertices)
  self.pcgCalculateNormResidual(mesh.vertices)
  local normRes = self.normRes:get()
  local thresh = self.cgEpsilon * self.cgEpsilon * normRes
  while normRes > thresh and
        iter <= self.cgMaxIterations do
    -- print("PCG iteration "..iter)
    self.pcgComputeAp(mesh.vertices)
    self.pcgComputeDummy(mesh.vertices)
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
  print("Time for solver is "..(timer_total:Stop()*1E6).." us")
--  I.cudaProfilerStop()
end

function ImplicitBackwardEulerIntegrator:doTimestep(mesh)
  local timer_total = Timer.New()
  local timer_inner = Timer.New()
  timer_total:Start()

  local err0 = 0 -- L.Global?
  local errQuotient

  -- store current amplitudes and set initial gues for qaccel, qvel
  -- AHHHHH, THE BELOW CAN BE IMPLEMENTED USING COPIES
  -- LOOP OVER THE STATE VECTOR (vertices for now)
  self.initializeQFields(mesh.vertices)

  -- Limit our total number of iterations allowed per timestep
  for numIter = 1, self.maxIterations do

    -- print("#dotimestep iteration = "..numIter.." ...")

    timer_inner:Start()
    computeInternalForces(mesh)
    print("Time to assemble force is "..(timer_inner:Stop()*1E6).." us")
    timer_inner:Start()
    computeStiffnessMatrix(mesh)
    print("Time to assemble stiffness matrix is "..(timer_inner:Stop()*1E6).." us")

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

    -- Add external/ internal internal_forces
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

    self:solvePCG(mesh)

    -- Reinsert the rows?

    self.updateAfterSolve(mesh.vertices)

    -- Constrain (zero) fields for the subset of constrained vertices
  end
  print("DoTimeStep time is "..(timer_total:Stop()*1E6).." us")

end

------------------------------------------------------------------------------

function clearExternalForces(mesh)
  mesh.vertices.external_forces:Load({ 0, 0, 0 })
end

local liszt kernel setExternalForces (v : mesh.vertices)
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

  local t1, t2
  -- print("Performing time steps ...")
  -- visualize(volumetric_mesh)
  DumpDeformationToFile(volumetric_mesh, "out/mesh_liszt_"..tostring(0))

  local timer = Timer.New()
  for i=1,options.numTimesteps do
    -- print("#timestep = "..i)
    timer:Start()
    setExternalConditions(volumetric_mesh, i)
    integrator:doTimestep(volumetric_mesh)
    print("Time for step "..i.." is "..(timer:Stop()*1E6).." us")
    print("")
    -- visualize(volumetric_mesh)
    DumpDeformationToFile(volumetric_mesh, "out/mesh_liszt_"..tostring(i))
  end

  -- read out the state here somehow?
end

main()
