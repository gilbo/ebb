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

  -- Fields for neohookean computation
  mesh.tetrahedra:NewField('B', L.mat3X3d)  -- inverse of [v1-v4, v2-v4, v3-v4]
  mesh.tetrahedra:NewField('W', L.double)   -- volume of tetrahedron

  -- TODO: Add temporary fields for per element quantities that are expensive
  -- to compute and are computed too often.

  ------------------------------------------------------------------------------
  -- Setup kernels to precompute some quantities.

  -- Compute B and W (similar computation, hence in one kernel)
  liszt self.computeBandW (t : mesh.tetrahedra)
  end

  ------------------------------------------------------------------------------
  -- Helper functions for computing stiffness and internal forces, and kernels
  -- to save some intermediate quantities into temporary fields.

  -- Compute Piola-Kirchoff sress 1
  -- TODO: might want to store some of these quantities into temporary fields.
  liszt self.PK1()
  end

  -- Compute gradient of Piola-Kirchoff stress 1 with respect to deformation.
  -- TODO: might want to store some of these quantities into temporary fields.
  liszt self.dPdF()
  end

  ------------------------------------------------------------------------------
  -- Assemble intrnal forces for each tetrahedral element.
  -- For corresponding Matlab code, see getForceEle in fem.m
  liszt self.computeInternalForces(t : mesh.tetrahedra)
  end

  ------------------------------------------------------------------------------
  -- Assemble stiffness matrix for each tetrahedral element.
  -- For corresponding Matlab code, see stiffnessEle in fem.m
  liszt self.computeStiffnessMatrix(t : mesh.tetrahedra)
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
