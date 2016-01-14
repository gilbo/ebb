import "ebb"
local L = require "ebblib"
local PN = require 'ebb.lib.pathname'
local U = require 'INTERNAL_devapps.fem.utils'

local S = {}
S.__index = S
package.loaded["INTERNAL_devapps.fem.stvk"] = S
S.profile = false


--------------------------------------------------------------------------------
-- All Ebb kernels go here. These are wrapped into Lua function calls (at the
-- end of the file) which are called by any code external to this module.
--------------------------------------------------------------------------------

function S:setupFieldsFunctions(mesh)

  mesh.tetrahedra:NewField('Phig', L.mat4x3d)

  ------------------------------------------------------------------------------
  -- For corresponding VEGA code, see
  --    libraries/stvk/StVKTetABCD.cpp (most of the file)
  
  -- Here, we precompute PhiG which is used to compute and cache dots, and
  -- compute A, b, C, and D as required, on a per element basis.
  ebb self.precomputeStVKIntegrals (t : mesh.tetrahedra)
    var det = t.elementDet
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
        t.volume = L.fabs(det) / 6
      end
    end
  end
  
  -- The VEGA Code seems to compute the dots matrix once, and then
  -- cache it for the duration of a per-tet computation rather than
  -- allocate disk space
  ebb self.tetDots(Phig)
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
  
  ebb self.tetCoefA(volume, phi, i, j)
    return ( volume * U.tensor3( { phi[i, 0], phi[i, 1], phi[i, 2] },
                                 { phi[j, 0], phi[j, 1], phi[j, 2] } ) )
  end
  
  ebb self.tetCoefB(volume, dots, i, j)
    return volume * dots[i, j]
  end
  
  ebb self.tetCoefC(volume, phi, dots, i, j, k)
    var res : L.vec3d = volume * dots[j, k] * 
                        { phi[i, 0], phi[i, 1], phi[i, 2] }
    return res
  end
  
  ebb self.tetCoefD(volume, dots, i, j, k, l)
    return ( volume * dots[i, j] * dots[k, l] )
  end
  
  
  ------------------------------------------------------------------------------
  -- For corresponding VEGA code, see
  --    libraries/stvk/StVKInternalinternal_forces.cpp (most of the file)
  
  -- extra functions supplied by this module
  -- Outer loop is generally over all elements (tetrahedra).
  -- Result is stored as a 3D vector field over all the vertices.
  
  -- Linear contributions to internal internal_forces
  ebb self.addIFLinearTerms (t : mesh.tetrahedra)
    var phi = t.Phig
    var dots = self.tetDots(phi)
    var lambda = t.lambdaLame
    var mu = t.muLame
    var volume = t.volume
    for ci = 0,4 do
      var c = t.v[ci]
      var internal_forces : L.vec3d = { 0, 0, 0 }
      for ai = 0,4 do
        var qa = t.v[ai].q
        var tetCoefAca = self.tetCoefA(volume, phi, ci, ai)
        var tetCoefAac = self.tetCoefA(volume, phi, ai, ci)
        var tetCoefBac = self.tetCoefB(volume, dots, ai, ci)
        var force = lambda *
                    U.multiplyMatVec3(tetCoefAca, qa) +
                    (mu * tetCoefBac) * qa +
                    mu * U.multiplyMatVec3(tetCoefAac, qa)
        internal_forces += force
      end
      c.internal_forces += internal_forces
    end
  end
  
  -- Quadratic contributions to internal internal_forces
  ebb self.addIFQuadraticTerms (t : mesh.tetrahedra)
    var phi = t.Phig
    var dots = self.tetDots(phi)
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
          var tetCoefCabc = self.tetCoefC(volume, phi, dots, ai, bi, ci)
          var tetCoefCbac = self.tetCoefC(volume, phi, dots, bi, ai, ci)
          var tetCoefCcab = self.tetCoefC(volume, phi, dots, ci, ai, bi)
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
  ebb self.addIFCubicTerms (t : mesh.tetrahedra)
    var phi = t.Phig
    var dots = self.tetDots(phi)
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
            var tetCoefDabcd = self.tetCoefD(volume, dots, ai, bi, ci, di)
            var tetCoefDacbd = self.tetCoefD(volume, dots, ai, ci, bi, di)
            var scalar = dotp * ( 0.5 * lambda * tetCoefDabcd +
                                  mu * tetCoefDacbd )
            internal_forces += scalar * qd
          end
        end
      end
      c.internal_forces += internal_forces
    end
  end
  
  ebb self.resetInternalForces (v : mesh.vertices)
    v.internal_forces = {0, 0, 0}
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
  
  -- PERFORMANCE NOTE:
  -- All operations are written as scatter operations. The code may perform
  -- better if we rewrite the operations as gather operations over edges, as
  -- against scatter from tetrahedra.
  
  -- Linear contributions to stiffness matrix
  ebb self.addStiffLinearTerms (t : mesh.tetrahedra)
    var phi = t.Phig
    var dots = self.tetDots(phi)
    var lambda = t.lambdaLame
    var mu = t.muLame
    var volume = t.volume
    for ci = 0,4 do
      for ai = 0,4 do
        var tetCoefAca = self.tetCoefA(volume, phi, ci, ai)
        var tetCoefAac = self.tetCoefA(volume, phi, ai, ci)
        var tetCoefBac = self.tetCoefB(volume, dots, ai, ci)
        var mat = U.diagonalMatrix(mu * tetCoefBac)
        mat += (lambda * tetCoefAca + (mu * tetCoefAac))
        t.e[ci, ai].stiffness += mat
      end
    end
  end
  
  -- Quadratic contributions to stiffness matrix
  ebb self.addStiffQuadraticTerms (t : mesh.tetrahedra)
    var phi = t.Phig
    var dots = self.tetDots(phi)
    var lambda = t.lambdaLame
    var mu = t.muLame
    var volume = t.volume
    for ci = 0,4 do
      for ai = 0,4 do
        var qa = t.v[ai].q
        var mat : L.mat3d = { { 0, 0, 0 }, { 0,0, 0 }, { 0, 0, 0 } }
        for ei = 0,4 do
          var tetCoefCcae = self.tetCoefC(volume, phi, dots, ci, ai, ei)
          var tetCoefCcea = self.tetCoefC(volume, phi, dots, ci, ei, ai)
          var tetCoefCeac = self.tetCoefC(volume, phi, dots, ei, ai, ci)
          var tetCoefCaec = self.tetCoefC(volume, phi, dots, ai, ei, ci)
          var c0v = lambda * tetCoefCcae +
                    mu * ( tetCoefCeac + tetCoefCaec )
          mat += U.tensor3(qa, c0v)
          var c1v = lambda * tetCoefCeac +
                    mu * ( tetCoefCcea + tetCoefCaec )
          mat += U.tensor3(qa, c1v)
          var c2v = lambda * tetCoefCaec +
                    mu * ( tetCoefCcae + tetCoefCeac )
          var dotp = L.dot(qa, c2v)
          mat += U.diagonalMatrix(dotp)
        end
        t.e[ci, ai].stiffness += mat
      end
    end
  end
  
  -- Cubic contributions to stiffness matrix
  ebb self.addStiffCubicTerms (t : mesh.tetrahedra)
    var phi = t.Phig
    var dots = self.tetDots(phi)
    var lambda = t.lambdaLame
    var mu = t.muLame
    var volume = t.volume
    for ci = 0,4 do
      for ei = 0,4 do
        var mat : L.mat3d = { { 0, 0, 0 }, { 0, 0, 0 }, { 0, 0, 0 } }
        for ai = 0,4 do
          var qa = t.v[ai].q
          for bi = 0,4 do
            var qb = t.v[bi].q
            var tetCoefDacbe = self.tetCoefD(volume, dots, ai, ci, bi, ei)
            var tetCoefDaebc = self.tetCoefD(volume, dots, ai, ei, bi, ci)
            var tetCoefDabce = self.tetCoefD(volume, dots, ai, bi, ci, ei)
            var d0 = lambda * tetCoefDacbe +
                     mu * ( tetCoefDaebc + tetCoefDabce )
            mat += d0 * (U.tensor3(qa, qb))
            var d1 = 0.5 * lambda * tetCoefDabce + mu * tetCoefDacbe
            var dotpd = d1 * L.dot(qa, qb)
            mat += U.diagonalMatrix(dotpd)
          end
        end
        t.e[ci, ei].stiffness += mat
      end
    end
  end
  
  ebb self.resetStiffnessMatrix (e : mesh.edges)
    e.stiffness = { { 0, 0, 0 }, { 0, 0, 0 }, { 0, 0, 0 } }
  end

  ------------------------------------------------------------------------------
  -- Invoke all precomputation.

  mesh.tetrahedra:foreach(self.precomputeStVKIntegrals)

end


--------------------------------------------------------------------------------
-- Wrapper functions to compute internal forces and stiffness matrix
--------------------------------------------------------------------------------

local computeInternalForcesHelper = function(tetrahedra)
  if S.profile then
    local timer = U.Timer.New()
    timer:Start()
    tetrahedra:foreach(S.addIFLinearTerms)
    print("Time for internal forces linear terms is "..(timer:Stop()*1E6).." us")
    timer:Start()
    tetrahedra:foreach(S.addIFQuadraticTerms)
    print("Time for internal forces quadratic terms is "..(timer:Stop()*1E6).." us")
    timer:Start()
    tetrahedra:foreach(S.addIFCubicTerms)
    print("Time for internal forces cubic terms is "..(timer:Stop()*1E6).." us")
  else
    tetrahedra:foreach(S.addIFLinearTerms)
    tetrahedra:foreach(S.addIFQuadraticTerms)
    tetrahedra:foreach(S.addIFCubicTerms)
  end
end

local computeStiffnessMatrixHelper = function(tetrahedra)
  if S.profile then
    local timer = U.Timer.New()
    timer:Start()
    tetrahedra:foreach(S.addStiffLinearTerms)
    print("Time for stiffness linear terms is "..(timer:Stop()*1E6).." us")
    timer:Start()
    tetrahedra:foreach(S.addStiffQuadraticTerms)
    print("Time for stiffness quadratic terms is "..(timer:Stop()*1E6).." us")
    timer:Start()
    tetrahedra:foreach(S.addStiffCubicTerms)
    print("Time for stiffness cubic terms is "..(timer:Stop()*1E6).." us")
  else
    tetrahedra:foreach(S.addStiffLinearTerms)
    tetrahedra:foreach(S.addStiffQuadraticTerms)
    tetrahedra:foreach(S.addStiffCubicTerms)
  end
end

local function computeInternalForces(mesh)
  mesh.vertices:foreach(S.resetInternalForces)
  local timer = U.Timer.New()
  timer:Start()
  computeInternalForcesHelper(mesh.tetrahedra)
  print("Time to assemble force is "..(timer:Stop()*1E6).." us")
end

local function computeStiffnessMatrix(mesh)
  mesh.edges:foreach(S.resetStiffnessMatrix)
  local timer = U.Timer.New()
  timer:Start()
  computeStiffnessMatrixHelper(mesh.tetrahedra)
  print("Time to assemble stiffness matrix is "..(timer:Stop()*1E6).." us")
end

local ts = 0
function S.computeInternalForcesAndStiffnessMatrix(mesh)
  ts = ts + 1
  local timer = U.Timer.New()
  timer:Start()
  computeInternalForces(mesh)
  -- mesh:dumpVertFieldToFile('internal_forces', "ebb_output/stvk-out/internal_forces_"..tostring(ts))
  computeStiffnessMatrix(mesh)
  -- mesh:dumpEdgeFieldToFile('stiffness', "ebb_output/stvk-out/stiffness_"..tostring(ts))
  print("Time to assemble force and stiffness matrix is "..(timer:Stop()*1E6).." us")
end
