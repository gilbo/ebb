import 'compiler.liszt'
local vdb = L.require 'lib.vdb'
--L.default_processor = L.GPU

local Tetmesh = L.require 'examples.fem.tetmesh'
local VEGFileIO = L.require 'examples.fem.vegfileio'
local PN = L.require 'lib.pathname'

local mesh_files = {
    [1] = './examples/fem/turtle-volumetric-homogeneous.veg', -- default, tiny, 347 vertices, 1185 tets
    [2] = './examples/fem/asianDragon-homogeneous.veg', -- 959 vertices, 2591 tets
    [3] = '/home/liszt/vega-models/hose_small.veg', -- use on lightroast
    [4] = '/home/liszt/vega-models/hose_medium.veg', -- use on lightroast
}
local volumetricMeshFileName = mesh_files[1]
print("Loading " .. volumetricMeshFileName)
local mesh = VEGFileIO.LoadTetmesh(volumetricMeshFileName)

--local I = terralib.includecstring([[
--#include "cuda_profiler_api.h"
--]])

local gravity = 9.81

print("Number of edges : " .. tostring(mesh.edges:Size() .. "\n"))

function initConfigurations()
  local options = {
    volumetricMeshFilename      = volumetricMeshFileName,
    timestep                    = 0.1,
    dampingMassCoef             = 1.0, -- alt. 10.0
    dampingStiffnessCoef        = 0.01, -- alt 0.001
    deformableObjectCompliance  = 1.0,

    maxIterations               = 1,
    epsilon                     = 1e-6,
    numTimesteps                = 5,

    cgEpsilon                   = 1e-6,
    -- cgMaxIterations             = 1
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
-- Helper functions, mapped functions, variables etc

-- Compute absolute value for a given variable
local liszt fabs(num)
  var result = num
  if num < 0 then result = -num end
  return result
end

-- Compute determinant for matrix formed by vertex positions
local liszt getElementDet(t)
  var a = t.v[0].pos
  var b = t.v[1].pos
  var c = t.v[2].pos
  var d = t.v[3].pos
  return (L.dot(a - d, L.cross(b - d, c - d)))
end

-- Get element density for a mesh element (tetreahedron)
local liszt getElementDensity(a)
  return L.double(mesh.density)
end

-- Identity matrix
local liszt getId3()
  return { { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 } }
end

-- Matrix with all entries equal to value v
local liszt constantMatrix3(v)
  return { { v, v, v }, { v, v, v }, { v, v, v } }
end

-- Tensor product of 2 vectors
local liszt tensor3(a, b)
  var result = { { a[0] * b[0], a[0] * b[1], a[0] * b[2] },
                 { a[1] * b[0], a[1] * b[1], a[1] * b[2] },
                 { a[2] * b[0], a[2] * b[1], a[2] * b[2] } }
  return result
end

-- Matrix vector product
local liszt multiplyMatVec3(M, x)
  return  { M[0, 0]*x[0] + M[0, 1]*x[1] + M[0, 2]*x[2],
            M[1, 0]*x[0] + M[1, 1]*x[1] + M[1, 2]*x[2],
            M[2, 0]*x[0] + M[2, 1]*x[1] + M[2, 2]*x[2] }
end
local liszt multiplyVectors(x, y)
  return { x[0]*y[0], x[1]*y[1], x[2]*y[2]  }
end

-- Diagonals
local liszt diagonalMatrix(a)
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
-- For corresponding VEGA code, see
--    libraries/volumetricMesh/generateMassMatrix.cpp (computeMassMatrix)
--    libraries/volumetricMesh/tetMesh.cpp (computeElementMassMatrix)

-- The following implemfntation combines computing element matrix and updating
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
  local liszt buildMassMatrix (t : mesh.tetrahedra)
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
  mesh.tetrahedra:map(buildMassMatrix)
end

------------------------------------------------------------------------------
-- Initialize Lame constants. This includes lambda and mu. See
--     libraries/volumetricMesh/volumetricMeshENuMaterial.h
--     (getLambda and getMu)
-- This code is used to initialize Lame constants when stiffnexx matrix/ internal_forces
-- are initialized.

local liszt getLambda(t)
  var E : L.double = mesh.E
  var Nu : L.double = mesh.Nu
  return ( (Nu * E) / ( ( 1 + Nu ) * ( 1 - 2 * Nu ) ) )
end

local liszt getMu(t)
  var E : L.double = mesh.E
  var Nu : L.double = mesh.Nu
  return ( ( E / ( 2 * ( 1 + Nu) ) ) )
end

local liszt initializeLameConstants (t : mesh.tetrahedra)
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
local liszt precomputeStVKIntegrals (t : mesh.tetrahedra)
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
local liszt tetDots(Phig)
  var dots : L.mat4d
  for i=0,4 do
    var Phigi : L.vec3d = { Phig[i, 0], Phig[i, 1], Phig[i, 2] }
    for j=0,4 do
      var Phigj : L.vec3d = { Phig[j, 0], Phig[j, 1], Phig[j, 2] }
      dots[i, j] = L.dot(Phigi, Phigj)
    end
  end
  return dots
end

local liszt tetCoefA(volume, phi, i, j)
  return ( volume * tensor3( { phi[i, 0], phi[i, 1], phi[i, 2] },
                             { phi[j, 0], phi[j, 1], phi[j, 2] } ) )
end

local liszt tetCoefB(volume, dots, i, j)
  return volume * dots[i, j]
end

local liszt tetCoefC(volume, phi, dots, i, j, k)
  var res : L.vec3d = volume * dots[j, k] * 
                      { phi[i, 0], phi[i, 1], phi[i, 2] }
  return res
end

local liszt tetCoefD(volume, dots, i, j, k, l)
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
local liszt addIFLinearTerms (t : mesh.tetrahedra)
  var phi = t.Phig
  var dots = tetDots(phi)
  var lambda = t.lambdaLame
  var mu = t.muLame
  var volume = t.volume
  for ci = 0,4 do
    var c = t.v[ci]
    var internal_forces : L.vec3d = { 0, 0, 0 }
    for ai = 0,4 do
      var qa = t.v[ai].q
      var tetCoefAca = tetCoefA(volume, phi, ci, ai)
      var tetCoefAac = tetCoefA(volume, phi, ai, ci)
      var tetCoefBac = tetCoefB(volume, dots, ai, ci)
      var force = lambda *
                  multiplyMatVec3(tetCoefAca, qa) +
                  (mu * tetCoefBac) * qa +
                  mu * multiplyMatVec3(tetCoefAac, qa)
      internal_forces += force
    end
    c.internal_forces += internal_forces
  end
end

-- Quadratic contributions to internal internal_forces
local liszt addIFQuadraticTerms (t : mesh.tetrahedra)
  var phi = t.Phig
  var dots = tetDots(phi)
  var lambda = t.lambdaLame
  var mu = t.muLame
  var volume = t.volume
  for ci = 0,4 do
    var c = t.v[ci]
    var internal_forces : L.vec3d = { 0, 0, 0 }
    for ai = 0,4 do
      var qa = t.v[ai].q
      for bi = 0,4 do
        var qb = t.v[bi].q
        var dotp = L.dot(qa, qb)
        var tetCoefCabc = tetCoefC(volume, phi, dots, ai, bi, ci)
        var tetCoefCbac = tetCoefC(volume, phi, dots, bi, ai, ci)
        var tetCoefCcab = tetCoefC(volume, phi, dots, ci, ai, bi)
        var forceTerm1 = 0.5 * lambda * dotp * tetCoefCcab
                         mu * dotp * tetCoefCabc
        var C = lambda * tetCoefCabc +
                mu * ( tetCoefCcab + tetCoefCbac )
        var dotCqa = L.dot(C, qa)
        internal_forces += forceTerm1 + dotCqa * qb
      end
    end
    c.internal_forces += internal_forces
  end
end

-- Cubic contributions to internal internal_forces
local liszt addIFCubicTerms (t : mesh.tetrahedra)
  var phi = t.Phig
  var dots = tetDots(phi)
  var lambda = t.lambdaLame
  var mu = t.muLame
  var volume = t.volume
  for ci = 0,4 do
    var c = t.v[ci]
    var internal_forces : L.vec3d = { 0, 0, 0 }
    for ai = 0,4 do
      var qa = t.v[ai].q
      for bi = 0,4 do
        var qb = t.v[bi].q
        for di = 0,4 do
          var d = t.v[di]
          var qd = d.q
          var dotp = L.dot(qa, qb)
          var tetCoefDabcd = tetCoefD(volume, dots, ai, bi, ci, di)
          var tetCoefDacbd = tetCoefD(volume, dots, ai, ci, bi, di)
          var scalar = dotp * ( 0.5 * lambda * tetCoefDabcd +
                                mu * tetCoefDacbd )
          internal_forces += scalar * qd
        end
      end
    end
    c.internal_forces += internal_forces
  end
end

local liszt resetInternalForces (v : mesh.vertices)
  v.internal_forces = {0, 0, 0}
end

local computeInternalForcesHelper = function(tetrahedra)
  local timer = Timer.New()
  -- print("Computing linear contributions to internal_forces ...")
  timer:Start()
  tetrahedra:map(addIFLinearTerms)
  print("Time for internal forces linear terms is "..(timer:Stop()*1E6).." us")
  -- print("Computing quadratic contributions to internal_forces ...")
  timer:Start()
  tetrahedra:map(addIFQuadraticTerms)
  print("Time for internal forces quadratic terms is "..(timer:Stop()*1E6).." us")
  -- print("Computing cubic contributions to internal_forces ...")
  timer:Start()
  tetrahedra:map(addIFCubicTerms)
  print("Time for internal forces cubic terms is "..(timer:Stop()*1E6).." us")
end

function computeInternalForces(mesh)
  -- print("Computing internal_forces ...")
  mesh.vertices:map(resetInternalForces)
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
local liszt addStiffLinearTerms (t : mesh.tetrahedra)
  var phi = t.Phig
  var dots = tetDots(phi)
  var lambda = t.lambdaLame
  var mu = t.muLame
  var volume = t.volume
  for ci = 0,4 do
    for ai = 0,4 do
      var tetCoefAca = tetCoefA(volume, phi, ci, ai)
      var tetCoefAac = tetCoefA(volume, phi, ai, ci)
      var tetCoefBac = tetCoefB(volume, dots, ai, ci)
      var mat = diagonalMatrix(mu * tetCoefBac)
      mat += (lambda * tetCoefAca + (mu * tetCoefAac))
      t.e[ci, ai].stiffness += mat
    end
  end
end

-- Quadratic contributions to stiffness matrix
local liszt addStiffQuadraticTerms (t : mesh.tetrahedra)
  var phi = t.Phig
  var dots = tetDots(phi)
  var lambda = t.lambdaLame
  var mu = t.muLame
  var volume = t.volume
  for ci = 0,4 do
    for ai = 0,4 do
      var qa = t.v[ai].q
      var mat : L.mat3d = constantMatrix3(0)
      for ei = 0,4 do
        var tetCoefCcae = tetCoefC(volume, phi, dots, ci, ai, ei)
        var tetCoefCcea = tetCoefC(volume, phi, dots, ci, ei, ai)
        var tetCoefCeac = tetCoefC(volume, phi, dots, ei, ai, ci)
        var tetCoefCaec = tetCoefC(volume, phi, dots, ai, ei, ci)
        var c0v = lambda * tetCoefCcae +
                  mu * ( tetCoefCeac + tetCoefCaec )
        mat += tensor3(qa, c0v)
        var c1v = lambda * tetCoefCeac +
                  mu * ( tetCoefCcea + tetCoefCaec )
        mat += tensor3(qa, c1v)
        var c2v = lambda * tetCoefCaec +
                  mu * ( tetCoefCcae + tetCoefCeac )
        var dotp = L.dot(qa, c2v)
        mat += diagonalMatrix(dotp)
      end
      t.e[ci, ai].stiffness += mat
    end
  end
end

-- Cubic contributions to stiffness matrix
local liszt addStiffCubicTerms (t : mesh.tetrahedra)
  var phi = t.Phig
  var dots = tetDots(phi)
  var lambda = t.lambdaLame
  var mu = t.muLame
  var volume = t.volume
  for ci = 0,4 do
    for ei = 0,4 do
      var mat : L.mat3d = constantMatrix3(0)
      for ai = 0,4 do
        var qa = t.v[ai].q
        for bi = 0,4 do
          var qb = t.v[bi].q
          var tetCoefDacbe = tetCoefD(volume, dots, ai, ci, bi, ei)
          var tetCoefDaebc = tetCoefD(volume, dots, ai, ei, bi, ci)
          var tetCoefDabce = tetCoefD(volume, dots, ai, bi, ci, ei)
          var d0 = lambda * tetCoefDacbe +
                   mu * ( tetCoefDaebc + tetCoefDabce )
          mat += d0 * (tensor3(qa, qb))
          var d1 = 0.5 * lambda * tetCoefDabce + mu * tetCoefDacbe
          var dotpd = d1 * L.dot(qa, qb)
          mat += diagonalMatrix(dotpd)
        end
      end
      t.e[ci, ei].stiffness += mat
    end
  end
end

local liszt resetStiffnessMatrix (e : mesh.edges)
  e.stiffness = constantMatrix3(0)
end

local computeStiffnessMatrixHelper = function(tetrahedra)
  local timer = Timer.New()
  -- print("Computing linear contributions to stiffness matrix ...")
  timer:Start()
  tetrahedra:map(addStiffLinearTerms)
  print("Time for stiffness linear terms is "..(timer:Stop()*1E6).." us")
  -- print("Computing quadratic contributions to stiffness matrix ...")
  timer:Start()
  tetrahedra:map(addStiffQuadraticTerms)
  print("Time for stiffness quadratic terms is "..(timer:Stop()*1E6).." us")
  -- print("Computing cubic contributions to stiffness matrix ...")
  timer:Start()
  tetrahedra:map(addStiffCubicTerms)
  print("Time for stiffness cubic terms is "..(timer:Stop()*1E6).." us")
end

function computeStiffnessMatrix(mesh)
  -- print("Computing stiffness matrix ...")
  mesh.edges:map(resetStiffnessMatrix)
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

function ImplicitBackwardEulerIntegrator:setupFieldsFunctions(mesh)

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

  self.err = L.Global(L.double, 0)
  self.normRes = L.Global(L.double, 0)
  self.alphaDenom = L.Global(L.double, 0)
  self.alpha = L.Global(L.double, 0)
  self.beta = L.Global(L.double, 0)

  liszt self.initializeQFields (v : mesh.vertices)
    v.q_1 = v.q
    v.qvel_1 = v.qvel
    v.qaccel = { 0, 0, 0 }
    v.qaccel_1 = { 0, 0, 0 }
  end

  liszt self.initializeqdelta (v : mesh.vertices)
    v.qdelta = v.qresidual
  end

  liszt self.scaleInternalForces (v : mesh.vertices)
    v.internal_forces = self.internalForcesScalingFactor * v.internal_forces
  end

  liszt self.scaleStiffnessMatrix (e : mesh.edges)
    e.stiffness = self.internalForcesScalingFactor * e.stiffness
  end

  liszt self.createRayleighDampMatrix (e : mesh.edges)
    e.raydamp = self.dampingStiffnessCoef * e.stiffness +
                diagonalMatrix(self.dampingMassCoef * e.mass)
  end

  liszt self.updateqresidual1 (v : mesh.vertices)
    for e in v.edges do
      v.qresidual += multiplyMatVec3(e.stiffness, (e.head.q_1 - e.head.q))
    end
  end

  liszt self.updateqresidual2 (v : mesh.vertices)
    for e in v.edges do
      v.qresidual += multiplyMatVec3(e.stiffness, e.head.qvel)
    end
  end

  liszt self.updateqresidual3 (v : mesh.vertices)
    v.qresidual += (v.internal_forces - v.external_forces)
    v.qresidual = - ( self.timestep * v.qresidual )
  end

  liszt self.updateqresidual4 (v : mesh.vertices)
    for e in v.edges do
      v.qresidual += e.mass * (e.head.qvel_1 - e.head.qvel)
    end
  end

  liszt self.updateStiffness1 (e : mesh.edges)
    e.stiffness = self.timestep * e.stiffness
    e.stiffness += e.raydamp
    -- TODO: This damping matrix seems to be zero unless set otherwise.
    -- e.stiffness = e.stiffness + e.dampingmatrix
  end

  liszt self.updateStiffness11 (e : mesh.edges)
    e.stiffness = self.timestep * e.stiffness
  end

  liszt self.updateStiffness12 (e : mesh.edges)
    e.stiffness += e.raydamp
  end

  liszt self.updateStiffness2 (e : mesh.edges)
    e.stiffness = self.timestep * e.stiffness
    e.stiffness += diagonalMatrix(e.mass)
  end

  liszt self.getError (v : mesh.vertices)
    var qd = v.qdelta
    var err = L.dot(qd, qd)
    self.err += err
  end

  liszt self.pcgCalculatePreconditioner (v : mesh.vertices)
    var stiff = v.diag.stiffness
    var diag = { stiff[0,0], stiff[1,1], stiff[2,2] }
    v.precond = { 1.0/diag[0], 1.0/diag[1], 1.0/diag[2] }
  end

  liszt self.pcgCalculateExactResidual (v : mesh.vertices)
    v.r = { 0, 0, 0 }
    for e in v.edges do
      v.r += multiplyMatVec3(e.stiffness, e.head.x)
    end
    v.r = v.qdelta - v.r
  end

  liszt self.pcgCalculateNormResidual (v : mesh.vertices)
    self.normRes += L.dot(multiplyVectors(v.r, v.precond), v.r)
  end

  liszt self.pcgInitialize (v : mesh.vertices)
    v.p = multiplyVectors(v.r, v.precond)
  end

  liszt self.pcgComputeAp (v : mesh.vertices)
    var Ap : L.vec3d = { 0, 0, 0 }
    for e in v.edges do
      var A = e.stiffness
      var p = e.head.p
      Ap += multiplyMatVec3(A, p)
    end
    v.Ap = Ap
  end

  liszt self.pcgComputeAlphaDenom (v : mesh.vertices)
    self.alphaDenom += L.dot(v.p, v.Ap)
  end

  liszt self.pcgUpdateX (v : mesh.vertices)
    v.x += self.alpha * v.p
  end

  liszt self.pcgUpdateResidual (v : mesh.vertices)
    v.r -= self.alpha * v.Ap
  end

  liszt self.pcgUpdateP (v : mesh.vertices)
    v.p = self.beta * v.p + multiplyVectors(v.precond, v.r)
  end

  liszt self.updateAfterSolve (v : mesh.vertices)
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
local t_solver = Timer.New()
local t_normres = Timer.New()
local t_computeap = Timer.New()
local t_alphadenom = Timer.New()
local t_updatex = Timer.New()
local t_updatep = Timer.New()
function ImplicitBackwardEulerIntegrator:solvePCG(mesh)
  -- I.cudaProfilerStart()
  t_solver:Start()
  local timer_total = Timer.New()
  timer_total:Start()
  mesh.vertices.x:Load({ 0, 0, 0 })
  mesh.vertices:map(self.pcgCalculatePreconditioner, {blocksize=16})
  local iter = 1
  mesh.vertices:map(self.pcgCalculateExactResidual, {blocksize=16})
  self.normRes:set(0)
  mesh.vertices:map(self.pcgInitialize, {blocksize=16})
  t_normres:Start()
  mesh.vertices:map(self.pcgCalculateNormResidual, {blocksize=16})
  t_normres:Pause()
  local normRes = self.normRes:get()
  local thresh = self.cgEpsilon * self.cgEpsilon * normRes
  while normRes > thresh and
        iter <= self.cgMaxIterations do
    t_computeap:Start()
    mesh.vertices:map(self.pcgComputeAp, {blocksize=16})
    t_computeap:Pause()
    self.alphaDenom:set(0)
    t_alphadenom:Start()
    mesh.vertices:map(self.pcgComputeAlphaDenom, {blocksize=64})
    t_alphadenom:Pause()
    self.alpha:set( normRes / self.alphaDenom:get() )
    t_updatex:Start()
    mesh.vertices:map(self.pcgUpdateX, {blocksize=64})
    t_updatex:Pause()
    if iter % 30 == 0 then
      mesh.vertices:map(self.pcgCalculateExactResidual, {blocksize=16})
    else
      mesh.vertices:map(self.pcgUpdateResidual, {blocksize=64})
    end
    local normResOld = normRes
    self.normRes:set(0)
    t_normres:Start()
    mesh.vertices:map(self.pcgCalculateNormResidual, {blocksize=64})
    t_normres:Pause()
    normRes = self.normRes:get()
    self.beta:set( normRes / normResOld )
    t_updatep:Start()
    mesh.vertices:map(self.pcgUpdateP, {blocksize=64})
    t_updatep:Pause()
    iter = iter + 1
  end
  if normRes > thresh then
      error("PCG solver did not converge!")
  end
  t_solver:Pause()
  -- I.cudaProfilerStop()
  print("Time for solver is "..(timer_total:Stop()*1E6).." us")
end

local timer_solver_reset = true
function ImplicitBackwardEulerIntegrator:doTimestep(mesh)
  local timer_total = Timer.New()
  local timer_inner = Timer.New()
  timer_total:Start()

  local err0 = 0 -- L.Global?
  local errQuotient

  -- store current amplitudes and set initial gues for qaccel, qvel
  -- AHHHHH, THE BELOW CAN BE IMPLEMENTED USING COPIES
  -- LOOP OVER THE STATE VECTOR (vertices for now)
  mesh.vertices:map(self.initializeQFields)

  -- Limit our total number of iterations allowed per timestep
  for numIter = 1, self.maxIterations do

    -- print("#dotimestep iteration = "..numIter.." ...")

    timer_inner:Start()
    computeInternalForces(mesh)
    print("Time to assemble force is "..(timer_inner:Stop()*1E6).." us")
    timer_inner:Start()
    computeStiffnessMatrix(mesh)
    print("Time to assemble stiffness matrix is "..(timer_inner:Stop()*1E6).." us")

    mesh.vertices:map(self.scaleInternalForces)
    mesh.edges:map(self.scaleStiffnessMatrix)

    -- ZERO out the residual field
    mesh.vertices.qresidual:Load({ 0, 0, 0 })

    -- NOTE: useStaticSolver == FALSE
    --    We just assume this everywhere
    mesh.edges:map(self.createRayleighDampMatrix)

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
      mesh.vertices:map(self.updateqresidual1)
    end

    -- some magic incantations corresponding to the above
    mesh.edges:map(self.updateStiffness11)
    mesh.edges:map(self.updateStiffness12)
    mesh.vertices:map(self.updateqresidual2)
    mesh.edges:map(self.updateStiffness2)

    -- Add external/ internal internal_forces
    mesh.vertices:map(self.updateqresidual3)

    -- superfluous on iteration 1, but safe to run
    if numIter ~= 1 then
      mesh.vertices:map(self.updateqresidual4)
    end

    -- TODO: this should be a copy and not a separate function in the end
    mesh.vertices:map(self.initializeqdelta)

    -- TODO: This code doesn't have any way of handling fixed vertices
    -- at the moment.  Should enforce that here somehow
    self.err:set(0)
    mesh.vertices:map(self.getError)

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
    if timer_solver_reset == true then
        t_solver:Reset()
        t_normres:Reset()
        t_computeap:Reset()
        t_alphadenom:Reset()
        t_updatex:Reset()
        t_updatep:Reset()
        timer_solver_reset = false
    end

    -- Reinsert the rows?

    mesh.vertices:map(self.updateAfterSolve)

    -- Constrain (zero) fields for the subset of constrained vertices
  end
  print("DoTimeStep time is "..(timer_total:Stop()*1E6).." us")

end

------------------------------------------------------------------------------

function clearExternalForces(mesh)
  mesh.vertices.external_forces:Load({ 0, 0, 0 })
end

local liszt setExternalForces (v : mesh.vertices)
  var pos = v.pos
  v.external_forces = { 10000, -80*(50-pos[1]), 0 }
end

function setExternalConditions(mesh, iter)
  if iter == 1 then
    mesh.vertices:map(setExternalForces)
  end
end

------------------------------------------------------------------------------

function main()
  local options = initConfigurations()

  local volumetric_mesh = mesh

  local nvertices = volumetric_mesh:nVerts()
  -- No fixed vertices for now
  local numFixedVertices = 0
  local numFixedDOFs     = 0
  local fixedDOFs        = nil

  -- print("Computing mass matrix ...")
  computeMassMatrix(volumetric_mesh)

  -- print("Computing integrals ...")
  mesh.tetrahedra:map(precomputeStVKIntegrals)
  -- print("Precomputed integrals")

  -- print("Initializing Lame constants ...")
  mesh.tetrahedra:map(initializeLameConstants)

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
  integrator:setupFieldsFunctions(mesh)

  -- print("Performing time steps ...")
  DumpDeformationToFile(volumetric_mesh, "out/mesh_liszt_"..tostring(0))

  local timer = Timer.New()
  for i=1,options.numTimesteps do
    -- print("#timestep = "..i)
    timer:Start()
    setExternalConditions(volumetric_mesh, i)
    integrator:doTimestep(volumetric_mesh)
    print("Time for step "..i.." is "..(timer:Stop()*1E6).." us")
    print("")
    DumpDeformationToFile(volumetric_mesh, "out/mesh_liszt_"..tostring(i))
  end

  print("Total solver time = " .. t_solver:GetTime()*1E3 .. "ms")
  print("Total normres time = " .. t_normres:GetTime()*1E3 .. "ms")
  print("Total computeap time = " .. t_computeap:GetTime()*1E3 .. "ms")
  print("Total alphadenom time = " .. t_alphadenom:GetTime()*1E3 .. "ms")
  print("Total updatex time = " .. t_updatex:GetTime()*1E3 .. "ms")
  print("Total updatep time = " .. t_updatep:GetTime()*1E3 .. "ms")

  -- read out the state here somehow?
end

main()
