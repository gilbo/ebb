import 'compiler/liszt'


--------------------------------------------------------------------------------
--[[ Load mesh relations, boundary sets                                     ]]--
--------------------------------------------------------------------------------
local M  = L.initMeshRelationsFromFile("examples/fem_mesh.lmesh")
M.left   = L.loadSetFromMesh(M, M.faces, 'inlet')
M.right  = L.loadSetFromMesh(M, M.faces, 'outlet')


--------------------------------------------------------------------------------
--[[ Allocate/initialize vertex fields                                      ]]--
--------------------------------------------------------------------------------
local terra f3zero(mem : &vector(float, 3), i : uint)
	@mem = vectorof(float, 0, 0, 0)
end

M.vertices:NewField('initialPos', L.vector(L.float, 3)):LoadFromCallback(f3zero)
M.vertices:NewField('v_n',        L.vector(L.float, 3)):LoadFromCallback(f3zero)
M.vertices:NewField('v_p',        L.vector(L.float, 3)):LoadFromCallback(f3zero)
M.vertices:NewField('d_n',        L.vector(L.float, 3)):LoadFromCallback(f3zero)
M.vertices:NewField('a_n',        L.vector(L.float, 3)):LoadFromCallback(f3zero)

M.vertices:NewField('v_n_h',      L.vector(L.float, 3)):LoadFromCallback(f3zero)
M.vertices:NewField('fext',       L.vector(L.float, 3)):LoadFromCallback(f3zero)
M.vertices:NewField('fint',       L.vector(L.float, 3)):LoadFromCallback(f3zero)

M.vertices:NewField('mass', L.float):LoadFromCallback(
	terra (mem: &float, i : uint) mem[0] = 2.0 end
)


--------------------------------------------------------------------------------
--[[ Allocate/initialize spring fields                                      ]]--
--------------------------------------------------------------------------------
function init_to (ival)
	return terra (mem : &float, i : uint)
		mem[0] = ival
	end
end

M.cells:NewField('springConstant',    L.float):LoadFromCallback(init_to(.3))
M.edges:NewField('initialEdgeLength', L.float):LoadFromCallback(init_to(.0))
M.edges:NewField('currentEdgeLength', L.float):LoadFromCallback(init_to(.0))


--------------------------------------------------------------------------------
--[[ Constants                                                              ]]--
--------------------------------------------------------------------------------
-- Initialize time index (n) and time (t^n)
local t_n    = 0
local t_n_h  = 0
local dt_n_h = .000005
local tmax   = .002

-- Constituitive constants for steel
local youngsMod = 200000000000
local poisson   = .3
local mu        = youngsMod / (2 * (1 + poisson))
local lambda    = (youngsMod * poisson) / ((1 + poisson) * (1 - 2 * poisson))
