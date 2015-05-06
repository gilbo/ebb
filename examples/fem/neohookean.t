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
--------------------------------------------------------------------------------

function N:setupFieldsFunctions(mesh)

end


--------------------------------------------------------------------------------
-- Wrapper functions to compute internal forces and stiffness matrix
--------------------------------------------------------------------------------

