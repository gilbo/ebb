import "compiler.liszt"
local PN = L.require 'lib.pathname'
local U = L.require 'examples.fem.utils'

local N = {}
N.__index = N
package.loaded["examples.fem.neohookean"] = N
N.profile = false


--------------------------------------------------------------------------------
-- All Liszt kernels go here. These are wrapped into Lua function calls (at the
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
  -- list of fields: B, W
  mesh.tetrahedra:NewField('Bm', L.mat3X3d)  -- inverse of [v1-v4, v2-v4, v3-v4]
  mesh.tetrahedra:NewField('W',  L.double)   -- volume of tetrahedron

  -- TODO: Add temporary fields for per element quantities that are expensive
  -- to compute and are computed too often.
  -- list of temporaries : F, Finv, JJ

  ------------------------------------------------------------------------------
  -- Setup kernels to precompute some quantities.

  -- Compute B and W (similar computation, hence in one kernel)
  -- For corresponding Matlab code, see module computeB/ siggraph notes
  -- algorithm 1
  liszt self.computeBandW (t : mesh.tetrahedra)
    var Dm : L.mat3d
    var x4 : L.vec3d = t.v[4]
    for j = 0,3  do
      var le : L.vec3d = t.v[j] - x4
      for i = 0,3 do
        Dm[i,j] = le[i]
      end
    end
    var det = U.detMatrix3d(Dm)
    t.W  = det/6.0
    t.Bm = U.invertMatrix3dGivenDet(Dm)
  end

  ------------------------------------------------------------------------------
  -- Helper functions for computing stiffness and internal forces, and kernels
  -- to save some intermediate quantities into temporary fields.

  -- Compute Piola-Kirchoff sress 1
  -- For corresponding Matlab code, see module PK1/ siggraph notes
  -- u = t.muLame, l = t.lambdaLame (should be this, but u in the Matlab code is
  -- slightly different)
  liszt self.PK1(t)
    var FinvT = t.FinvT
    var PP    = t.muLame * (t.F - t.FinvT) + (t.lambdaLame * t.JJ) * t.FinvT
    return PP
  end

  -- Compute gradient of Piola-Kirchoff stress 1 with respect to deformation.
  -- For corresponding Matlab code, see module dPdF/ siggraph notes page 24/ 32
  -- u = t.muLame, l = t.lambdaLame (should be this, but u in the Matlab code is
  -- slightly different)
  liszt self.dPdF(t, dF)
    var dfT      = U.transposeMatrix3(dF)
    var FinvT    = t.FinvT
    var c1       = t.muLame - t.lambdaLame * L.log(JJ);
    var dfTFinvT = U.multiplyMatrix3d(dFT, FinvT)
    var c2       = t.lambdaLame * (dfTFinvT[0,0] + dfTFinvT[1,1] + dfTFinvT[2,2])
    var FinvTdFTFinvT = U.multiplyMatrix3d(FinvT, dfTFinvT)
    var dP = (t.muLame * dF) + (c1 * FinvTdfTFinvT) + c2 * FinvT
  end

  ------------------------------------------------------------------------------
  -- Recompute temporaies for the new time step (F, Finv, J)
  -- For corresponding code, see Matlab code fem.m/ siggraph notes algorithm 1
  -- Reset internal forces and stiffness matrix
  liszt self.recomputeAndRestTemporaries(t : mesh.tetrahedra)
    var Ds : L.mat3d  = { { 0, 0, 0 }, { 0, 0, 0 }, { 0, 0, 0 } }
    var x4 = t.v[4].pos
    for j = 0, 3 do
      var le : L.vec3d = t.v[j].pos - x4
      for i = 0,3 do
        Ds[i,j] = le[i]
      end
    end
    var Bm : L.mat3d = t.Bm
    var F  : L.mat3d  = U.multiplyMatrices3d(Ds, Bm)
    t.F = F
    var Finv = U.invertMatrix3d(F)
    t.FinvT  = U.transposeMatrix3(Finv)
    T.JJ = L.log(U.detMatrix3d(F))
  end

  ------------------------------------------------------------------------------
  -- Assemble intrnal forces for each tetrahedral element.
  -- For corresponding Matlab code, see getForceEle in fem.m/ siggraph notes
  -- algorithm 1
  -- u = t.muLame, l = t.lambdaLame (should be this, but u in the Matlab code is
  -- slightly different)
  liszt self.computeInternalForces(t : mesh.tetrahedra)
    var P  : L.mat3d = PK1(t)
    va BmT : L.mat3d = U.transposeMatrix3(t.Bm)
    var H  : L.mat3d = (-t.W) * U.multiplyMatrices3d(P, BmT)
    for i = 0,3 do
      var fi : L.vec3d = { H[0,i], H[1,i], H[2,i] }
      t.v[i].f +=  fi
      t.v[4].f += -fi
    end
  end

  ------------------------------------------------------------------------------
  -- Assemble stiffness matrix for each tetrahedral element.
  -- For corresponding Matlab code, see stiffnessEle in fem.m/ siggraph notes
  -- algorithm 2
  -- u = t.muLame, l = t.lambdaLame (should be this, but u in the Matlab code is
  -- slightly different)
  liszt self.computeStiffnessMatrix(t : mesh.tetrahedra)
    var dDs : L.mat3d  = { { 0, 0, 0 }, { 0, 0, 0 }, { 0, 0, 0 } }
    var dx4 = t.v[4].q
    for j = 0, 3 do
      var le : L.vec3d = t.v[j].q - dx4
      for i = 0,3 do
        dDs[i,j] = le[i]
      end
    end
    var Bm : L.mat3d = t.Bm
    var dF : L.mat3d = U.multiplyMatrices3d(dDs, Bm)
    var dP : L.mat3d = self.dPdF(t, dF)
    va BmT : L.mat3d = U.transposeMatrix3(t.Bm)
    var dH : L.mat3d = (-t.W) * U.multiplyMatrices3d(dP, BmT)
    for i = 0,3 do
      var dfi : L.vec3d = { dH[0,i], dH[1,i], dH[2,i] }
      t.v[i].stiffness +=  fi
      t.v[4].stiffness += -fi
    end
  end

  ------------------------------------------------------------------------------
  -- Invoke all precomputation.

  mesh.tetrahedra:map(self.computeBandW)

end


--------------------------------------------------------------------------------
-- Wrapper functions to compute internal forces and stiffness matrix
--------------------------------------------------------------------------------

local function computeInternalForcesAndStiffnessMatrix(mesh)
  mesh.edges:map(S.resetStiffnessMatrix)
  local timer = U.Timer.New()
  timer:Start()
  mesh.tetrahedra:map(N.computeInternalForces)
  local t_if = timer:Stop() * 1E6
  print("Time to assemble force is "..(t_if).." us")
  timer:Start()
  mesh.tetrahedra:map(N.computeStiffnessMatrix)
  local t_stiff = timer:Stop() * 1E6
  print("Time to assemble stiffness matrix is "..(t_stiff).." us")
  print("Time to assemble force and stiffness matrix is "..(t_if + t_stiff).." us")
end
