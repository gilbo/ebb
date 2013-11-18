--------------------------------------------------------------------------------
--[[
   ... CENTRAL DIFFERENCE TIME LOOP

   DESCRIPTION: this routine implements the explicit central difference time-integrator
                for the solution of the second-order differential equation:
                (a) LINEAR mech or acou: M*(d^2u/dt^2) + C*(du/dt) + K*u = fext(u)
                (b) NONLINEAR mech: M*(d^2u/dt^2) + C*(du/dt) + fint(u) = fext(u)
  
   WARNINGS:    1. Viscous damping is supported, but to keep the scheme explicit the equilibrium 
                   condition is expressed as M*a^{n+1} + C*v^{n+1/2} + K*u^{n+1} = fext^{n+1}
                   where v^{n+1/2} = v^n + dt/2*a^n
                2. Velocity and/or acceleration controls (ACTUATORS) are not strictly correct since we
                   use v^n and a^n to compute fext^{n+1}
]]--
--------------------------------------------------------------------------------

import 'compiler.liszt'

--------------------------------------------------------------------------------
--[[ Load relations from lmesh                                              ]]--
--------------------------------------------------------------------------------
local LMesh = terralib.require "compiler.lmesh"
local M  = LMesh.Load("examples/fem_mesh.lmesh")
M.left   = M.inlet
M.right  = M.outlet
local C, V, F, E = M.cells, M.vertices, M.faces, M.edges


--------------------------------------------------------------------------------
--[[ Callback initializers for fields                                       ]]--
--------------------------------------------------------------------------------
local function init(c) 
	return terra (mem : &float, i : uint)
		mem[0] = c
	end
end

local float3_zero = terra (mem : &vector(float, 3), i : uint)
	mem[0] = vectorof(float, 0, 0, 0)
end


--------------------------------------------------------------------------------
--[[ FEM field allocation                                                   ]]--
--------------------------------------------------------------------------------
V:NewField('v_n',   L.vector(L.float, 3)):LoadFromCallback(float3_zero)
V:NewField('v_p',   L.vector(L.float, 3)):LoadFromCallback(float3_zero)
V:NewField('d_n',   L.vector(L.float, 3)):LoadFromCallback(float3_zero)
V:NewField('a_n',   L.vector(L.float, 3)):LoadFromCallback(float3_zero)
V:NewField('v_n_h', L.vector(L.float, 3)):LoadFromCallback(float3_zero)
V:NewField('fext',  L.vector(L.float, 3)):LoadFromCallback(float3_zero)
V:NewField('fint',  L.vector(L.float, 3)):LoadFromCallback(float3_zero)

V:NewField('mass', L.float):LoadFromCallback(init(1.0))

C:NewField('springConstant',    L.float):LoadFromCallback(init(0.3))
E:NewField('initialEdgeLength', L.float):LoadFromCallback(init(0.0))
E:NewField('currEdgeLength',    L.float):LoadFromCallback(init(0.0))


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

-- Want edge.c1...
-- Not all edges have the same number of cells, so we will have to build an array
-- for indexing into the cellsofedge.cell field to determine which cell to assign
-- to each edge.
local cd = M.cellsofedge.cell.data
local ed = M.cellsofedge.edge.data

local Clib   = terralib.includec("stdlib.h")
local offset = terralib.cast(&uint64, Clib.malloc(terralib.sizeof(uint64) * (M.edges:size() + 1)))

local edge_no = -1
for i = 0, M.cellsofedge:size() do
	if ed[i] ~= edge_no then
		edge_no = edge_no + 1
		offset[edge_no] = i
	end
end

local function ccall (j)
	return terra (mem : &uint64, i : uint) mem[0] = cd[offset[i]+j] end
end
E:NewField('c1', C):LoadFromCallback(ccall(1))


--------------------------------------------------------------------------------
--[[ Constants                                                              ]]--
--------------------------------------------------------------------------------
local t_n_h  = 0
local dt_n_h = 1
local tmax   = 1000


--------------------------------------------------------------------------------
--[[ Global kernels                                                         ]]--
--------------------------------------------------------------------------------
local reset_internal_forces = liszt_kernel (v in M.vertices)
   	v.fint = {0, 0, 0} 
end

-- Update the displacement at t^(n+1): d^{n+1} = d^n + dt^{n+1/2}*v^{n+1/2}
local update_pos_and_disp = liszt_kernel (v in M.vertices)
	v.d_n      += dt_n_h * v.v_n_h
	v.position += dt_n_h * v.v_n_h
end

-- calculate edge length displacement
local calc_edge_disp = liszt_kernel (e in M.edges)
	var v1 = e.head
	var v2 = e.tail

	var dist = v1.position - v2.position
	e.currEdgeLength = L.length(dist)
end

local calc_internal_force = liszt_kernel (e in M.edges)
	var edgeLengthDisplacement = e.currEdgeLength - e.initialEdgeLength
		
	var v1 = e.head
	var v2 = e.tail

	var disp = v1.position - v2.position
	var len  = L.length(disp)
	var norm = disp / len

	v1.fint += -e.c1.springConstant * edgeLengthDisplacement * norm
	v2.fint -= -e.c1.springConstant * edgeLengthDisplacement * norm
end

-- Compute the acceleration at t^{n+1}: a^{n+1} = M^{-1}(fext^{n+1}-fint^{n+1}-C*v^{n+1/2})
local compute_accel = liszt_kernel (v in M.vertices)
	v.a_n = v.fext + v.fint / v.mass
end

local update_previous_velocity = liszt_kernel (v in M.vertices) v.v_p = v.v_n end

-- Update the velocity at t^{n+1}: v^{n+1} = v^{n+1/2}+dt^{n+1/2}/2*a^n
local update_velocity = liszt_kernel (v in M.vertices)
	v.v_n = v.v_n_h + .5 * dt_n_h * v.a_n
end


--------------------------------------------------------------------------------
--[[ Main                                                                   ]]--
--------------------------------------------------------------------------------
local function main()

	--[[ Initialize external forces: ]]--
	(liszt_kernel (f in M.left)
		f.value.v0.fext = {.01, 0, 0}
		f.value.v1.fext = {.01, 0, 0}
		f.value.v2.fext = {.01, 0, 0}
		f.value.v3.fext = {.01, 0, 0}
	end)()

	(liszt_kernel (f in M.right)
		f.value.v0.fext = {-.01, 0, 0}
		f.value.v1.fext = {-.01, 0, 0}
		f.value.v2.fext = {-.01, 0, 0}
		f.value.v3.fext = {-.01, 0, 0}
	end)()

	--[[ Initialize acceleration based on initial forces ]]--
	(liszt_kernel (v in M.vertices)
		v.a_n = (v.fext - v.fint) / v.mass
	end)()

	--[[ Initialize edge lengths]]--
	(liszt_kernel (e in M.edges)
		var v1 = e.head
		var v2 = e.tail

		var distance = v1.position - v2.position
		var length   = L.length(distance)

		e.initialEdgeLength = length
		e.currEdgeLength    = length
	end)()

	--[[ MAIN LOOP: ]]--
	local t_n = 0

	while (t_n < tmax - .01 * dt_n_h) do

		--Update half time:  t^{n+1/2} = t^n + 1/2*deltat^{n+1/2}
	  	t_n_h = t_n + dt_n_h/2 

		-- nodal velocity kernel depends on changing t_n
		local update_nodal_velocities = liszt_kernel (v in M.vertices)
			v.v_n_h = v.v_n + (t_n_h - t_n) * v.a_n
		end

		--[[ Execute! ]]--
		reset_internal_forces()

		update_nodal_velocities()
		update_pos_and_disp()

		calc_edge_disp()
		calc_internal_force()

		compute_accel()
		update_previous_velocity()
		update_velocity()

		-- Time update: t^n = t^{n-1} + deltat^{n-1/2}
		t_n = t_n + dt_n_h
	end
end

main()

-- Print results
(liszt_kernel (v in M.vertices)
	L.print(v.position)
end)()


