import 'compiler/liszt'


--------------------------------------------------------------------------------
--[[ Load mesh relations, boundary sets                                     ]]--
--------------------------------------------------------------------------------
local M  = L.initMeshRelationsFromFile("examples/fem_mesh.lmesh")
M.left   = L.loadSetFromMesh(M, M.faces, 'inlet',  'face')
M.right  = L.loadSetFromMesh(M, M.faces, 'outlet', 'face')


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
--[[ Create structured topology relations                                   ]]--
--------------------------------------------------------------------------------
-- Since domain is a cube mesh, want to access vertices of face as 
-- f.v0, f.v1, f.v2, f.v3
local vd = M.verticesofface.vertex.data
local function vcall (j)
	return terra (mem : &uint64, i : uint) mem[0] = vd[4*i+j] end
end

F:NewField('v0', V):LoadFromCallback(vcall(0))
F:NewField('v1', V):LoadFromCallback(vcall(1))
F:NewField('v2', V):LoadFromCallback(vcall(2))
F:NewField('v3', V):LoadFromCallback(vcall(3))

-- Similarly, want cell.v0, ... cell.v8
local cd = M.verticesofcell.vertex.data
local function vcall (j)
	return terra (mem : &uint64, i : uint) mem[0] = cd[8*i+j] end
end

C:NewField('v0', V):LoadFromCallback(vcall(0))
C:NewField('v1', V):LoadFromCallback(vcall(1))
C:NewField('v2', V):LoadFromCallback(vcall(2))
C:NewField('v3', V):LoadFromCallback(vcall(3))
C:NewField('v4', V):LoadFromCallback(vcall(4))
C:NewField('v5', V):LoadFromCallback(vcall(5))
C:NewField('v6', V):LoadFromCallback(vcall(6))
C:NewField('v7', V):LoadFromCallback(vcall(7))


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
