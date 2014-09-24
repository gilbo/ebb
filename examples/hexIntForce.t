import 'compiler.liszt'

	--[[
	-- NOTE: because Liszt does not yet support matrices, we are storing MxN
	-- matrices as vectors of length m*n.  The vectors are a row-major
	-- representation of the matrices, where M(i,j) = M[n*i+j]
	]]--

--------------------------------------------------------------------------------
--[[ Libraries                                                              ]]--
--------------------------------------------------------------------------------
local m      = terralib.includec("math.h")
local sqrt   = m.sqrt
local printf = terralib.includec("stdio.h").printf
local PN     = L.require('lib.pathname')

--------------------------------------------------------------------------------
--[[ Load mesh relations, boundary sets                                     ]]--
--------------------------------------------------------------------------------
local LMesh = L.require "domains.lmesh"
local M  = LMesh.Load(PN.scriptdir():concat("fem_mesh.lmesh"):tostring())
M.left   = M.inlet
M.right  = M.outlet
local C, V, F, E = M.cells, M.vertices, M.faces, M.edges


--------------------------------------------------------------------------------
--[[ Allocate/initialize vertex fields                                      ]]--
--------------------------------------------------------------------------------

local zero = {0,0,0}
M.vertices:NewField('initialPos', L.vector(L.double, 3)):LoadConstant(zero)
M.vertices:NewField('v_n',        L.vector(L.double, 3)):LoadConstant(zero)
M.vertices:NewField('v_p',        L.vector(L.double, 3)):LoadConstant(zero)
M.vertices:NewField('a_n',        L.vector(L.double, 3)):LoadConstant(zero)
M.vertices:NewField('v_n_h',      L.vector(L.double, 3)):LoadConstant(zero)
M.vertices:NewField('fext',       L.vector(L.double, 3)):LoadConstant(zero)
M.vertices:NewField('fint',       L.vector(L.double, 3)):LoadConstant(zero)

M.vertices:NewField('mass', L.double):LoadConstant(2.0)

-- Create vertex inlet boundary subsets
V:NewField('inlet', L.bool):LoadConstant(false)
V:NewField('outlet', L.bool):LoadConstant(false)


--------------------------------------------------------------------------------
--[[ Create structured topology relations                                   ]]--
--------------------------------------------------------------------------------
local Clib = terralib.includec("stdlib.h")

-- Since domain is a cube mesh, want to access vertices of face as 
-- f.v0, f.v1, f.v2, f.v3
function build_structured_face()
	local vd = M.verticesofface.vertex:DataPtr()
	local function vcall (j)
		return terra (mem : &uint64, i : uint) mem[0] = vd[4*i+j] end
	end

	local fd     = M.verticesofface.face:DataPtr()
	local vd     = M.verticesofface.vertex:DataPtr()
	local offset = terralib.cast(&uint64, Clib.malloc(terralib.sizeof(uint64) * (M.faces:Size() + 1)))

	local face_no = -1
	for i = 0, M.verticesofface:Size() do
		if fd[i] ~= face_no then
			face_no = face_no + 1
			offset[face_no] = i
		end
	end
	local function vcall (j)
		return (function (i) return vd[offset[i] + j] end)
	end
	F:NewField('v0', V):LoadFunction(vcall(0))
	F:NewField('v1', V):LoadFunction(vcall(1))
	F:NewField('v2', V):LoadFunction(vcall(2))
	F:NewField('v3', V):LoadFunction(vcall(3))
	Clib.free(offset)
end

-- Similarly, want cell.v0, ... cell.v8
function build_structured_cell()
	local cd     = M.verticesofcell.cell:DataPtr()
	local vd     = M.verticesofcell.vertex:DataPtr()
	local offset = terralib.cast(&uint64, Clib.malloc(terralib.sizeof(uint64) * (M.cells:Size() + 1)))

	local cell_no = -1
	for i = 0, M.verticesofcell:Size() do
		if cd[i] ~= cell_no then
			cell_no = cell_no + 1
			offset[cell_no] = i
		end
	end
	local function ccall (j)
		return (function (i) return vd[offset[i]+j] end)
	end
	C:NewField('v0', V):LoadFunction(ccall(0))
	C:NewField('v1', V):LoadFunction(ccall(1))
	C:NewField('v2', V):LoadFunction(ccall(2))
	C:NewField('v3', V):LoadFunction(ccall(3))
	C:NewField('v4', V):LoadFunction(ccall(4))
	C:NewField('v5', V):LoadFunction(ccall(5))
	C:NewField('v6', V):LoadFunction(ccall(6))
	C:NewField('v7', V):LoadFunction(ccall(7))
	Clib.free(offset)
end

build_structured_face()
build_structured_cell()


--------------------------------------------------------------------------------
--[[ Build inlet/outlet vertex sets                                         ]]--
--------------------------------------------------------------------------------
function buildVertexInletOutletSets ()
	local mark_inlet_vertices = liszt kernel (f : M.left)
		f.value.v0.inlet or= true	
		f.value.v1.inlet or= true
		f.value.v2.inlet or= true
		f.value.v3.inlet or= true
	end
	local mark_outlet_vertices = liszt kernel (f : M.right)
		f.value.v0.outlet or= true
		f.value.v1.outlet or= true
		f.value.v2.outlet or= true
		f.value.v3.outlet or= true
	end
	mark_inlet_vertices(M.left)
	mark_outlet_vertices(M.right)

	local is_inlet   = V.inlet:DataPtr()
	local is_outlet  = V.outlet:DataPtr()
	V:NewSubsetFromFunction("inlet_vertices",  function(i) return is_inlet[i]  end)
	V:NewSubsetFromFunction("outlet_vertices", function(i) return is_outlet[i] end)
end
buildVertexInletOutletSets()


--------------------------------------------------------------------------------
--[[ Constants                                                              ]]--
--------------------------------------------------------------------------------
-- Initialize time index (n) and time (t^n)
local dt_n_h = .000005
local tmax   = .002

-- Constituitive constants for steel
local youngsMod = 200000000000
local poisson   = .3
local mu        = youngsMod / (2 * (1 + poisson))
local lambda    = (youngsMod * poisson) / ((1 + poisson) * (1 - 2 * poisson))


--------------------------------------------------------------------------------
--[[ Interior force calculation: kernel and terra helper functions          ]]--
--------------------------------------------------------------------------------
-- return vec8f; xi, eta, zeta are doubles
local shapeFunction = liszt function(xi, eta, zeta)
	return 1.0f/8.0f * {
		(1-xi) * (1-eta) * (1-zeta),
	    (1+xi) * (1-eta) * (1-zeta),
	    (1+xi) * (1+eta) * (1-zeta),
	    (1-xi) * (1+eta) * (1-zeta),
	    (1-xi) * (1-eta) * (1+zeta),
	    (1+xi) * (1-eta) * (1+zeta),
	    (1+xi) * (1+eta) * (1+zeta),
	    (1-xi) * (1+eta) * (1+zeta) 
	}
end

-- returns vec24f; eta, xi, zeta are doubles
local derivative = liszt function(eta, xi, zeta)
	return 1.0f/8.0f * {
		-(eta - 1)*(zeta - 1), -(xi - 1)*(zeta - 1), -(eta - 1)*(xi - 1),
		 (eta - 1)*(zeta - 1),  (xi + 1)*(zeta - 1),  (eta - 1)*(xi + 1),
		-(eta + 1)*(zeta - 1), -(xi + 1)*(zeta - 1), -(eta + 1)*(xi + 1),
		 (eta + 1)*(zeta - 1),  (xi - 1)*(zeta - 1),  (eta + 1)*(xi - 1),
		 (eta - 1)*(zeta + 1),  (xi - 1)*(zeta + 1),  (eta - 1)*(xi - 1),
		-(eta - 1)*(zeta + 1), -(xi + 1)*(zeta + 1), -(eta - 1)*(xi + 1),
		 (eta + 1)*(zeta + 1),  (xi + 1)*(zeta + 1),  (eta + 1)*(xi + 1),
		-(eta + 1)*(zeta + 1), -(xi - 1)*(zeta + 1), -(eta + 1)*(xi - 1) }
end

local dot3 = liszt function (a, b)
	return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]
end

local dot6 = liszt function (a, b)
	return  a[0]*b[0] + a[1]*b[1] + a[2]*b[2] 
	      + a[3]*b[3] + a[4]*b[4] + a[5]*b[5]
end

local dot8 = liszt function (a, b)
	return  a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3]
	      + a[4]*b[4] + a[5]*b[5] + a[6]*b[6] + a[7]*b[7]
end

-- M is an 8x3 matrix, and cols 1, 2 and 3 are the columns of a 3x3 matrix
-- returns another 8x3 matrix
local mult_8x3_3x3 = liszt function(M, col1, col2, col3)
		var MR1 = {M[0],  M[1],  M[2]}
	    var MR2 = {M[3],  M[4],  M[5]}
	    var MR3 = {M[6],  M[7],  M[8]}
	    var MR4 = {M[9],  M[10], M[11]}
	    var MR5 = {M[12], M[13], M[14]}
	    var MR6 = {M[15], M[16], M[17]}
	    var MR7 = {M[18], M[19], M[20]}
	    var MR8 = {M[21], M[22], M[23]}

	    return {
	    	dot3(MR1, col1), dot3(MR1, col2), dot3(MR1, col3),
	    	dot3(MR2, col1), dot3(MR2, col2), dot3(MR2, col3),
	    	dot3(MR3, col1), dot3(MR3, col2), dot3(MR3, col3),
	    	dot3(MR4, col1), dot3(MR4, col2), dot3(MR4, col3),
	    	dot3(MR5, col1), dot3(MR5, col2), dot3(MR5, col3),
	    	dot3(MR6, col1), dot3(MR6, col2), dot3(MR6, col3),
	    	dot3(MR7, col1), dot3(MR7, col2), dot3(MR7, col3),
	    	dot3(MR8, col1), dot3(MR8, col2), dot3(MR8, col3) }
end

-- F is a 3x3 matrix, returning a 3x3 matrix
local calculate_stress_tensor = liszt function (F)
	var FT1 = {F[0], F[3], F[6]}
	var FT2 = {F[1], F[4], F[7]}
	var FT3 = {F[2], F[5], F[8]}

	-- S = F_transpose*F - I
	var S1 = .5 * {dot3(FT1, FT1)-1.0, dot3(FT1, FT2),     dot3(FT1, FT3)}
	var S2 = .5 * {dot3(FT2, FT1),     dot3(FT2, FT2)-1.0, dot3(FT2, FT3)}
	var S3 = .5 * {dot3(FT3, FT1),     dot3(FT3, FT2),     dot3(FT3, FT3)-1.0}

	var B = { S1[0], S2[1], S3[2], 2*S2[2], 2*S3[0], 2*S1[1] }

	var A1 = { 2.0 * mu + lambda, lambda, lambda, 0.0, 0.0, 0.0 }
	var A2 = { lambda, 2.0 * mu + lambda, lambda, 0.0, 0.0, 0.0 }
	var A3 = { lambda, lambda, 2.0 * mu + lambda, 0.0, 0.0, 0.0 }
	var A4 = { 0.0, 0.0, 0.0,  mu, 0.0, 0.0 }
	var A5 = { 0.0 ,0.0, 0.0, 0.0,  mu, 0.0 }
	var A6 = { 0.0, 0.0, 0.0, 0.0, 0.0,  mu }

	var C  = {dot6(A1, B), dot6(A2, B), dot6(A3, B),
	          dot6(A4, B), dot6(A5, B), dot6(A6, B)}

	var str_col1 = { C[0], C[5], C[4] }
	var str_col2 = { C[5], C[1], C[3] }
	var str_col3 = { C[4], C[3], C[2] }

	var F1 = { F[0], F[1], F[2] }
	var F2 = { F[3], F[4], F[5] }
	var F3 = { F[6], F[7], F[8] }

	var stress_tensor = {
		dot3(F1, str_col1), dot3(F1, str_col2), dot3(F1, str_col3),
		dot3(F2, str_col1), dot3(F2, str_col2), dot3(F2, str_col3),
		dot3(F3, str_col1), dot3(F3, str_col2), dot3(F3, str_col3)
	}

	return stress_tensor
end

local F = L.double
local calculate_internal_force = liszt kernel (c : M.cells)
	-- ignore outside cell
	if L.id(c) ~= 0	then
		-- col1, col2, col3 are the columns of a matrix X, where the rows of X are
		-- the initial positions of the 8 vertices of this cell
		var col1 = {
			c.v0.initialPos[0], c.v1.initialPos[0], c.v2.initialPos[0], c.v3.initialPos[0],
			c.v4.initialPos[0], c.v5.initialPos[0], c.v6.initialPos[0], c.v7.initialPos[0] }
		var col2 = {
			c.v0.initialPos[1], c.v1.initialPos[1], c.v2.initialPos[1], c.v3.initialPos[1],
			c.v4.initialPos[1], c.v5.initialPos[1], c.v6.initialPos[1], c.v7.initialPos[1] }
		var col3 = {
			c.v0.initialPos[2], c.v1.initialPos[2], c.v2.initialPos[2], c.v3.initialPos[2],
			c.v4.initialPos[2], c.v5.initialPos[2], c.v6.initialPos[2], c.v7.initialPos[2] }

		-- Initializing internal forces
		var f_int_1 : L.vector(L.double, 3) = {0,0,0}
		var f_int_2 : L.vector(L.double, 3) = {0,0,0}
		var f_int_3 : L.vector(L.double, 3) = {0,0,0}
		var f_int_4 : L.vector(L.double, 3) = {0,0,0}
		var f_int_5 : L.vector(L.double, 3) = {0,0,0}
		var f_int_6 : L.vector(L.double, 3) = {0,0,0}
		var f_int_7 : L.vector(L.double, 3) = {0,0,0}
		var f_int_8 : L.vector(L.double, 3) = {0,0,0}

		-- IP is an 8x3 matrix
		var IP : L.vector(L.double, 24) = 1.0f / L.double(sqrt(3)) * {
			-1.0f, -1.0f, -1.0f,
			-1.0f, -1.0f,  1.0f,
			-1.0f,  1.0f, -1.0f,
			 1.0f, -1.0f, -1.0f,
			-1.0f,  1.0f,  1.0f,
			 1.0f, -1.0f,  1.0f,
			 1.0f,  1.0f, -1.0f,
			 1.0f,  1.0f,  1.0f
		}

		var i = 0
		while i < 8 do
			-- shapefunc and shapederiv are computed from the i'th row of IP
			var shapeFunc  = shapeFunction(IP[3*i], IP[3*i+1], IP[3*i+2])
			-- shapeDeriv is actually an 8x3 matrix
			var shapeDeriv = derivative(   IP[3*i], IP[3*i+1], IP[3*i+2])

			-- sD1, sD2, sD3 are the rows of matrix SD
			var sD1 = { shapeDeriv[0], shapeDeriv[3], shapeDeriv[6],  shapeDeriv[9], shapeDeriv[12], shapeDeriv[15], shapeDeriv[18], shapeDeriv[21] }
			var sD2 = { shapeDeriv[1], shapeDeriv[4], shapeDeriv[7], shapeDeriv[10], shapeDeriv[13], shapeDeriv[16], shapeDeriv[19], shapeDeriv[22] }
			var sD3 = { shapeDeriv[2], shapeDeriv[5], shapeDeriv[8], shapeDeriv[11], shapeDeriv[14], shapeDeriv[17], shapeDeriv[20], shapeDeriv[23] }

			var J : L.vector(L.double, 9) = {
				L.dot(sD1, col1), L.dot(sD1, col2),	L.dot(sD1, col3),
				L.dot(sD2, col1), L.dot(sD2, col2),	L.dot(sD2, col3),
				L.dot(sD3, col1), L.dot(sD3, col2),	L.dot(sD3, col3) 
			}

			var JT : L.vector(L.double, 9) = {
				J[0], J[3], J[6],
				J[1], J[4], J[7],
				J[2], J[5], J[8]
			}

			var JT_det    =   JT[0] * (JT[8]*JT[4] - JT[7]*JT[5])
			                - JT[3] * (JT[8]*JT[1] - JT[7]*JT[2])
			                + JT[6] * (JT[5]*JT[1] - JT[4]*JT[2])

			var J_tr_in_1 = {  JT[8]*JT[4] - JT[7]*JT[5],
			                 -(JT[8]*JT[1] - JT[7]*JT[2]),
			                   JT[5]*JT[1] - JT[4]*JT[2] }

			var J_tr_in_2 = { -(JT[8]*JT[1] - JT[6]*JT[5]),
			                    JT[8]*JT[0] - JT[6]*JT[2],
			                  -(JT[5]*JT[0] - JT[3]*JT[2])}

			var J_tr_in_3 = {  JT[7]*JT[1] - JT[6]*JT[4],
			                 -(JT[7]*JT[0] - JT[6]*JT[1]),
			                   JT[4]*JT[0] - JT[3]*JT[1] }

			var JTI_col1 = 1/JT_det * {J_tr_in_1[0], J_tr_in_2[0], J_tr_in_3[0]}
			var JTI_col2 = 1/JT_det * {J_tr_in_1[1], J_tr_in_2[1], J_tr_in_3[1]}
			var JTI_col3 = 1/JT_det * {J_tr_in_1[2], J_tr_in_2[2], J_tr_in_3[2]}

			var nGrad : L.vector(L.double, 24) = mult_8x3_3x3(shapeDeriv, JTI_col1, JTI_col2, JTI_col3)

			var nGrad_col1 = {nGrad[0], nGrad[3], nGrad[6], nGrad[9],  nGrad[12], nGrad[15], nGrad[18], nGrad[21]}
			var nGrad_col2 = {nGrad[1], nGrad[4], nGrad[7], nGrad[10], nGrad[13], nGrad[16], nGrad[19], nGrad[22]}
			var nGrad_col3 = {nGrad[2], nGrad[5], nGrad[8], nGrad[11], nGrad[14], nGrad[17], nGrad[20], nGrad[23]}

			var XT_row1 = { F(c.v0.position[0]), F(c.v1.position[0]), F(c.v2.position[0]), F(c.v3.position[0]),
			                F(c.v4.position[0]), F(c.v5.position[0]), F(c.v6.position[0]), F(c.v7.position[0])  }
			var XT_row2 = { F(c.v0.position[1]), F(c.v1.position[1]), F(c.v2.position[1]), F(c.v3.position[1]),
			                F(c.v4.position[1]), F(c.v5.position[1]), F(c.v6.position[1]), F(c.v7.position[1])  }
			var XT_row3 = { F(c.v0.position[2]), F(c.v1.position[2]), F(c.v2.position[2]), F(c.v3.position[2]),
			                F(c.v4.position[2]), F(c.v5.position[2]), F(c.v6.position[2]), F(c.v7.position[2])  }

			var F = { L.dot(XT_row1, nGrad_col1), L.dot(XT_row1, nGrad_col2), L.dot(XT_row1, nGrad_col3),
			          L.dot(XT_row2, nGrad_col1), L.dot(XT_row2, nGrad_col2), L.dot(XT_row2, nGrad_col3),
			          L.dot(XT_row3, nGrad_col1), L.dot(XT_row3, nGrad_col2), L.dot(XT_row3, nGrad_col3)  }

			var P = calculate_stress_tensor(F)

			var P1 = { P[0], P[1], P[2] }
			var P2 = { P[3], P[4], P[5] }
			var P3 = { P[6], P[7], P[8] }

			var nGrad1 = { nGrad[0],  nGrad[1],  nGrad[2]  }
			var nGrad2 = { nGrad[3],  nGrad[4],  nGrad[5]  }
			var nGrad3 = { nGrad[6],  nGrad[7],  nGrad[8]  }
			var nGrad4 = { nGrad[9],  nGrad[10], nGrad[11] }
			var nGrad5 = { nGrad[12], nGrad[13], nGrad[14] }
			var nGrad6 = { nGrad[15], nGrad[16], nGrad[17] }
			var nGrad7 = { nGrad[18], nGrad[19], nGrad[20] }
			var nGrad8 = { nGrad[21], nGrad[22], nGrad[23] }

			f_int_1 += JT_det * L.vec3f({ L.dot(P1, nGrad1), L.dot(P2, nGrad1), L.dot(P3, nGrad1) })
			f_int_2 += JT_det * L.vec3f({ L.dot(P1, nGrad2), L.dot(P2, nGrad2), L.dot(P3, nGrad2) })
			f_int_3 += JT_det * L.vec3f({ L.dot(P1, nGrad3), L.dot(P2, nGrad3), L.dot(P3, nGrad3) })
			f_int_4 += JT_det * L.vec3f({ L.dot(P1, nGrad4), L.dot(P2, nGrad4), L.dot(P3, nGrad4) })
			f_int_5 += JT_det * L.vec3f({ L.dot(P1, nGrad5), L.dot(P2, nGrad5), L.dot(P3, nGrad5) })
			f_int_6 += JT_det * L.vec3f({ L.dot(P1, nGrad6), L.dot(P2, nGrad6), L.dot(P3, nGrad6) })
			f_int_7 += JT_det * L.vec3f({ L.dot(P1, nGrad7), L.dot(P2, nGrad7), L.dot(P3, nGrad7) })
			f_int_8 += JT_det * L.vec3f({ L.dot(P1, nGrad8), L.dot(P2, nGrad8), L.dot(P3, nGrad8) })

			i = i + 1
		end

		c.v0.fint += f_int_1
		c.v1.fint += f_int_2
		c.v2.fint += f_int_3
		c.v3.fint += f_int_4
		c.v4.fint += f_int_5
		c.v5.fint += f_int_6
		c.v6.fint += f_int_7
		c.v7.fint += f_int_8
	end
end


--------------------------------------------------------------------------------
--[[ Global micro-kernels                                                   ]]--
--------------------------------------------------------------------------------
local reset_internal_forces = liszt kernel (v : M.vertices) v.fint = {0,0,0} end

local update_position = liszt kernel (v : M.vertices)
	v.position += dt_n_h * v.v_n_h
end

local compute_acceleration = liszt kernel (v : M.vertices)
	v.a_n = (v.fext - v.fint) / v.mass
end

local update_velocity = liszt kernel (v : M.vertices)
	v.v_n = v.v_n_h + L.vec3f(.5f * dt_n_h * v.a_n)
end


--------------------------------------------------------------------------------
--[[ Main                                                                   ]]--
--------------------------------------------------------------------------------
local function main ()
	-- Initialize position
	(liszt kernel (v : M.vertices)
		v.initialPos = L.vec3f(v.position)
	end)(M.vertices)


	--[[ Initialize external forces: ]]--
	(liszt kernel (v : M.vertices)
		v.fext = L.vec3f({10000000, 0, 0})
	end)(V.inlet_vertices)

	(liszt kernel (v : M.vertices)
		v.fext = L.vec3f({-10000000, 0, 0})
	end)(V.outlet_vertices)

	-- Initialize acceleration based on initial forces
	compute_acceleration(M.vertices)

	local t_n   = 0
	local t_n_h = 0
	while (t_n < tmax) do

		-- Update half time:  t^{n+1/2} = t^n + 1/2*deltat^{n+1/2}
		t_n_h = t_n + dt_n_h/2

		reset_internal_forces(M.vertices)
		-- Update nodal velocities (requires inline kernel to escape current t values)
		(liszt kernel (v : M.vertices)
			v.v_n_h = v.v_n + L.vec3f((t_n_h - t_n) * v.a_n)
		end)(M.vertices)

		update_position(M.vertices)
		calculate_internal_force(M.cells)
		compute_acceleration(M.vertices)
		update_velocity(M.vertices)

		-- Time update: t^n = t^{n-1} + deltat^{n-1/2}
		t_n = t_n + dt_n_h
	end

	-- DEBUG
	(liszt kernel (v : M.vertices)
		L.print(v.position)
	end)(M.vertices)
end

main()

























