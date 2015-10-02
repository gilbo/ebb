import "ebb"
local PN = require 'ebb.lib.pathname'
local U = require 'devapps.fem.utils'

local N = {}
N.__index = N
package.loaded["devapps.fem.neohookean"] = N
N.profile = false


--------------------------------------------------------------------------------
-- All Ebb kernels go here. These are wrapped into Lua function calls (at the
-- end of the file) which are called by any code external to this module.
--
-- This code is based on the Matlab implementation of neohookean model by
-- David Levin (Simit team). The equations also correspond to Vega's
-- implementation (isotropicHyperelasticFEM and neoHookeanIsotropicMaterial),
-- but:
-- 1. avoids SVD decomposition (operates on deformation and stress directly)
-- 2. plugs in the values for various gradients directly for neohookean model,
-- instead of going through an indirection that calls gradient of energy
-- density.
--------------------------------------------------------------------------------

function N:setupFieldsFunctions(mesh)

  -- additional macros/ constants for mesh

  -- Fields for neohookean computation

  -- list of precomputed fields: Bm, W
  mesh.tetrahedra:NewField('Bm', L.mat3d)    -- inverse of 'reference shape matrix'
  mesh.tetrahedra:NewField('W',  L.double)   -- volume of tetrahedron

  -- list of temporaries : F, FinvT, Fdet
  -- TODO: Add J too!
  mesh.tetrahedra:NewField('F',     L.mat3d)     -- deformation gradient
  mesh.tetrahedra:NewField('FinvT', L.mat3d)     -- (deformation gradient) inverse transpose
  mesh.tetrahedra:NewField('Fdet',  L.double)    -- determinant of (deformation gradient)

  ------------------------------------------------------------------------------
  -- Setup kernels to precompute some quantities.

  -- Compute B and W (similar computation, hence in one kernel)
  -- For corresponding Matlab code, see module computeB/ siggraph notes
  -- algorithm 1
  ebb self.computeBAndW(t : mesh.tetrahedra)
    var Dm : L.mat3d
    var x4 : L.vec3d = t.v[3].pos
    for j = 0,3  do
      var le : L.vec3d = t.v[j].pos - x4
      for i = 0,3 do
        Dm[i,j] = le[i]
      end
    end
    var det = L.fabs(U.detMatrix3d(Dm))
    t.W  = det/6.0
    t.Bm = U.invertMatrix3d(Dm)
  end

  ------------------------------------------------------------------------------
  -- Helper functions for computing stiffness and internal forces, and kernels
  -- to save some intermediate quantities into temporary fields.

  -- Compute Piola-Kirchoff sress 1
  -- For corresponding Matlab code, see module PK1/ siggraph notes
  -- u = t.muLame, l = t.lambdaLame (should be this, but u in the Matlab code is
  -- slightly different)
  ebb self.PK1(t)
    var F     = t.F
    var FinvT = t.FinvT
    var PP    = t.muLame * (t.F - FinvT) + (t.lambdaLame * L.log(t.Fdet)) * FinvT
    return PP
  end

  -- Compute gradient of Piola-Kirchoff stress 1 with respect to deformation.
  -- For corresponding Matlab code, see module dPdF/ siggraph notes page 24/ 32
  -- u = t.muLame, l = t.lambdaLame (should be this, but u in the Matlab code is
  -- slightly different)
  ebb self.dPdF(t, dF)
    var dFT      = U.transposeMatrix3(dF)
    var FinvT    = t.FinvT
    var c1       = t.muLame - t.lambdaLame * L.log(t.Fdet)
    var dFTFinvT = U.multiplyMatrices3d(dFT, FinvT)
    var c2       = t.lambdaLame * (dFTFinvT[0,0] + dFTFinvT[1,1] + dFTFinvT[2,2])
    var FinvTdFTFinvT = U.multiplyMatrices3d(FinvT, dFTFinvT)
    var dP = (t.muLame * dF) + (c1 * FinvTdFTFinvT) + c2 * FinvT
    return dP
  end

  ------------------------------------------------------------------------------
  -- Recompute temporaies for the new time step (F, Finv, J)
  -- For corresponding code, see Matlab code fem.m/ siggraph notes algorithm 1
  -- Reset internal forces and stiffness matrix

  ebb self.recomputeAndResetTetTemporaries(t : mesh.tetrahedra)
   -- recompute
    var Ds : L.mat3d  = { { 0, 0, 0 }, { 0, 0, 0 }, { 0, 0, 0 } }
    var x4 = t.v[3].pos + t.v[3].q
    for j = 0, 3 do
      var le : L.vec3d = (t.v[j].pos + t.v[j].q) - x4
      for i = 0,3 do
        Ds[i,j] = le[i]
      end
    end
    var Bm : L.mat3d = t.Bm
    var F  : L.mat3d  = U.multiplyMatrices3d(Ds, Bm)
    t.F = F
    var Finv = U.invertMatrix3d(F)
    t.FinvT  = U.transposeMatrix3(Finv)
    t.Fdet   = L.fabs(U.detMatrix3d(F))
  end

  ebb self.recomputeAndResetEdgeTemporaries(e : mesh.edges)
    e.stiffness = { { 0, 0, 0 }, { 0, 0, 0 }, { 0, 0, 0 } }
  end

  ebb self.recomputeAndResetVertexTemporaries(v : mesh.vertices)
    v.internal_forces = { 0, 0, 0 }
  end

  ------------------------------------------------------------------------------
  -- Assemble intrnal forces for each tetrahedral element.
  -- For corresponding Matlab code, see getForceEle in fem.m/ siggraph notes
  -- algorithm 1
  -- u = t.muLame, l = t.lambdaLame (should be this, but u in the Matlab code is
  -- slightly different)
  ebb self.computeInternalForces(t : mesh.tetrahedra)
    var  P  : L.mat3d = self.PK1(t)
    var BmT : L.mat3d = U.transposeMatrix3(t.Bm)
    var rhs : L.mat3d = U.multiplyMatrices3d(P, BmT)
    var  H  : L.mat3d = (t.W) * U.multiplyMatrices3d(P, BmT)
    for i = 0,3 do
      var fi : L.vec3d = { H[0,i], H[1,i], H[2,i] }
      t.v[i].internal_forces += ( fi)
      t.v[3].internal_forces += (-fi)
    end
  end

  ------------------------------------------------------------------------------
  -- Assemble stiffness matrix for each tetrahedral element.
  -- For corresponding Matlab code, see stiffnessEle in fem.m ~
  -- ~ siggraph notes algorithm 2 (I couldn't translate the algorithm)
  -- u = t.muLame, l = t.lambdaLame (should be this, but u in the Matlab code is
  -- slightly different)
  ebb self.computeStiffnessMatrix(t : mesh.tetrahedra)
    var Bm  : L.mat3d = t.Bm
    var BmT : L.mat3d = U.transposeMatrix3(t.Bm)
    var dFRow : L.mat4x3d = { { 0, 0, 0 }, { 0, 0, 0 }, { 0, 0, 0 }, { 0, 0, 0 } }
    -- assemble dFRow
    for i = 0,3 do
      for j = 0,3 do
        dFRow[i,j]  =  Bm[i,j]
        dFRow[3,j] += -Bm[i,j]
      end
    end
    -- for every vertex, assemble interactions with every other vertex
    for v = 0,4 do
      var s0 : L.mat3d = { { 0, 0, 0 }, { 0, 0, 0 }, { 0, 0, 0 } }
      var s1 : L.mat3d = { { 0, 0, 0 }, { 0, 0, 0 }, { 0, 0, 0 } }
      var s2 : L.mat3d = { { 0, 0, 0 }, { 0, 0, 0 }, { 0, 0, 0 } }
      var s3 : L.mat3d = { { 0, 0, 0 }, { 0, 0, 0 }, { 0, 0, 0 } }
      for k = 0,3 do
        var dF : L.mat3d = { { 0, 0, 0 }, { 0, 0, 0 }, { 0, 0, 0 } }
        for j = 0,3 do
          dF[k,j] = dFRow[v,j]
        end
        var dP : L.mat3d = self.dPdF(t, dF)
        var dH : L.mat3d = (t.W) * U.multiplyMatrices3d(dP, BmT)
        -- was before:
        -- t.e[i,v].stiffness[r, k] += dH[r, i]
        -- t.e[3,v].stiffness[r, k] -= dH[r, i]
        for r = 0,3 do
          s0[r, k] =    dH[r, 0]
          s1[r, k] =    dH[r, 1]
          s2[r, k] =    dH[r, 2]
          s3[r, k] = -( dH[r, 0] + dH[r, 1] + dH[r, 2] )
        end
      end
      t.e[0, v].stiffness += s0
      t.e[1, v].stiffness += s1
      t.e[2, v].stiffness += s2
      t.e[3, v].stiffness += s3
    end
  end

  ------------------------------------------------------------------------------
  -- Invoke all precomputation.

  mesh.tetrahedra:foreach(self.computeBAndW)

end


--------------------------------------------------------------------------------
-- Wrapper functions to compute internal forces and stiffness matrix
--------------------------------------------------------------------------------

local ts = 0
function N.computeInternalForcesAndStiffnessMatrix(mesh)
  ts = ts + 1
  mesh.tetrahedra:foreach(N.recomputeAndResetTetTemporaries)
  mesh.edges:foreach(N.recomputeAndResetEdgeTemporaries)
  mesh.vertices:foreach(N.recomputeAndResetVertexTemporaries)
  local timer = U.Timer.New()
  timer:Start()
  mesh.tetrahedra:foreach(N.computeInternalForces)
  local t_if = timer:Stop() * 1E6
  print("Time to assemble force is "..(t_if).." us")
  -- mesh:dumpVertFieldToFile('internal_forces', "ebb_output/nh-out/internal_forces_"..tostring(ts))
  timer:Start()
  mesh.tetrahedra:foreach(N.computeStiffnessMatrix)
  local t_stiff = timer:Stop() * 1E6
  print("Time to assemble stiffness matrix is "..(t_stiff).." us")
  -- mesh:dumpEdgeFieldToFile('stiffness', "ebb_output/nh-out/stiffness_"..tostring(ts))
  print("Time to assemble force and stiffness matrix is "..(t_if + t_stiff).." us")
end
