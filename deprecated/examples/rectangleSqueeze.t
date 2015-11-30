-------------------------------------------------------------------------------
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
-------------------------------------------------------------------------------

import 'compiler.liszt'

-------------------------------------------------------------------------------
--[[ Load relations from lmesh                                             ]]--
-------------------------------------------------------------------------------
local PN     = L.require('lib.pathname')
local LMesh  = L.require "domains.lmesh"
local M  = LMesh.Load(PN.scriptdir():concat("fem_mesh.lmesh"):tostring())
M.left   = M.inlet
M.right  = M.outlet
local C, V, F, E = M.cells, M.vertices, M.faces, M.edges



-------------------------------------------------------------------------------
--[[ FEM field allocation                                                  ]]--
-------------------------------------------------------------------------------
V:NewField('v_n',   L.vector(L.float, 3)):LoadConstant( {0, 0, 0} )
V:NewField('v_p',   L.vector(L.float, 3)):LoadConstant( {0, 0, 0} )
V:NewField('d_n',   L.vector(L.float, 3)):LoadConstant( {0, 0, 0} )
V:NewField('a_n',   L.vector(L.float, 3)):LoadConstant( {0, 0, 0} )
V:NewField('v_n_h', L.vector(L.float, 3)):LoadConstant( {0, 0, 0} )
V:NewField('fext',  L.vector(L.float, 3)):LoadConstant( {0, 0, 0} )
V:NewField('fint',  L.vector(L.float, 3)):LoadConstant( {0, 0, 0} )

V:NewField('mass', L.float):LoadConstant(1.0)

C:NewField('springConstant',    L.float):LoadConstant(0.3)
E:NewField('initialEdgeLength', L.float):LoadConstant(0.0)
E:NewField('currEdgeLength',    L.float):LoadConstant(0.0)

-- Create vertex inlet boundary subsets
V:NewField('inlet', L.bool):LoadConstant(false)
V:NewField('outlet', L.bool):LoadConstant(false)


-------------------------------------------------------------------------------
--[[ Create structured topology relations                                  ]]--
-------------------------------------------------------------------------------
-- Since domain is a cube mesh, want to access vertices of face as 
-- f.v0, f.v1, f.v2, f.v3
local vd = M.verticesofface.vertex:DataPtr()
--local function vcall (j)
--	return terra (mem : &uint64, i : uint) mem[0] = vd[4*i+j] end
--end
local function vcall (j)
	return (function(i) return vd[4*i + j] end)
end

F:NewField('v0', V):LoadFunction(vcall(0))
F:NewField('v1', V):LoadFunction(vcall(1))
F:NewField('v2', V):LoadFunction(vcall(2))
F:NewField('v3', V):LoadFunction(vcall(3))

-- Similarly, want cell.v0, ... cell.v8
local cd = M.verticesofcell.vertex:DataPtr()
--local function vcall (j)
--	return terra (mem : &uint64, i : uint) mem[0] = cd[8*i+j] end
--end
local function vcall (j)
	return (function(i) return cd[8*i + j] end)
end

C:NewField('v0', V):LoadFunction(vcall(0))
C:NewField('v1', V):LoadFunction(vcall(1))
C:NewField('v2', V):LoadFunction(vcall(2))
C:NewField('v3', V):LoadFunction(vcall(3))
C:NewField('v4', V):LoadFunction(vcall(4))
C:NewField('v5', V):LoadFunction(vcall(5))
C:NewField('v6', V):LoadFunction(vcall(6))
C:NewField('v7', V):LoadFunction(vcall(7))

-- Want edge.c1...
-- Not all edges have the same number of cells, so we will have to build an array
-- for indexing into the cellsofedge.cell field to determine which cell to assign
-- to each edge.
local cd = M.cellsofedge.cell:DataPtr()
local ed = M.cellsofedge.edge:DataPtr()

local Clib   = terralib.includec("stdlib.h")
local offset = terralib.cast(&uint64, Clib.malloc(terralib.sizeof(uint64) * (M.edges:Size() + 1)))

local edge_no = -1
for i = 0, M.cellsofedge:Size() do
	if ed[i] ~= edge_no then
		edge_no = edge_no + 1
		offset[edge_no] = i
	end
end

--local function ccall (j)
--	return terra (mem : &uint64, i : uint) mem[0] = cd[offset[i]+j] end
--end
local function ccall (j)
	return (function(i) return cd[offset[i] + j] end)
end
E:NewField('c1', C):LoadFunction(ccall(1))


-------------------------------------------------------------------------------
--[[ Build inlet/outlet vertex sets                                        ]]--
-------------------------------------------------------------------------------
function buildVertexInletOutletSets ()
	local liszt mark_inlet_vertices (f : M.left)
		f.value.v0.inlet or= true	
		f.value.v1.inlet or= true
		f.value.v2.inlet or= true
		f.value.v3.inlet or= true
	end
	local liszt mark_outlet_vertices (f : M.right)
		f.value.v0.outlet or= true
		f.value.v1.outlet or= true
		f.value.v2.outlet or= true
		f.value.v3.outlet or= true
	end
	M.right:foreach(mark_inlet_vertices)
	M.right:foreach(mark_outlet_vertices)

	local is_inlet   = V.inlet:DataPtr()
	local is_outlet  = V.outlet:DataPtr()
	V:NewSubsetFromFunction("inlet_vertices",  function(i) return is_inlet[i]  end)
	V:NewSubsetFromFunction("outlet_vertices", function(i) return is_outlet[i] end)
end
buildVertexInletOutletSets()


-------------------------------------------------------------------------------
--[[ Constants                                                             ]]--
-------------------------------------------------------------------------------
local t_n_h  = 0
local dt_n_h = 1
local tmax   = 1000


-------------------------------------------------------------------------------
--[[ Global Functions                                                      ]]--
-------------------------------------------------------------------------------
local liszt reset_internal_forces (v : M.vertices)
   	v.fint = {0, 0, 0} 
end

-- Update the displacement at t^(n+1): d^{n+1} = d^n + dt^{n+1/2}*v^{n+1/2}
local liszt update_pos_and_disp (v : M.vertices)
	v.d_n      += dt_n_h * v.v_n_h
	v.position += dt_n_h * v.v_n_h
end

-- calculate edge length displacement
local liszt calc_edge_disp (e : M.edges)
	var v1 = e.head
	var v2 = e.tail

	var dist = v1.position - v2.position
	e.currEdgeLength = L.length(dist)
end

local liszt calc_internal_force (e : M.edges)
	var edgeLengthDisplacement = e.currEdgeLength - e.initialEdgeLength
		
	var v1 = e.head
	var v2 = e.tail

	var disp = L.vec3f(v1.position - v2.position)
	var len  = L.length(disp)
	var norm = disp / len

	v1.fint += -e.c1.springConstant * edgeLengthDisplacement * norm
	v2.fint +=  e.c1.springConstant * edgeLengthDisplacement * norm
end

-- Compute the acceleration at t^{n+1}: a^{n+1} = M^{-1}(fext^{n+1}-fint^{n+1}-C*v^{n+1/2})
local liszt compute_accel (v : M.vertices)
	v.a_n = v.fext + v.fint / v.mass
end

local liszt update_previous_velocity (v : M.vertices) v.v_p = v.v_n end

-- Update the velocity at t^{n+1}: v^{n+1} = v^{n+1/2}+dt^{n+1/2}/2*a^n
local liszt update_velocity (v : M.vertices)
	v.v_n = v.v_n_h + L.float(.5f) * dt_n_h * v.a_n
end


-------------------------------------------------------------------------------
--[[ Choose Architecture                                                   ]]--
-------------------------------------------------------------------------------
function moveToArch (l_arch)
	L.SetDefaultProcessor(l_arch)
	M.vertices:MoveTo(l_arch)
	M.edges:MoveTo(l_arch)
	M.cells:MoveTo(l_arch)
end
moveToArch(L.GPU)


-------------------------------------------------------------------------------
--[[ Main                                                                  ]]--
-------------------------------------------------------------------------------
local function main()

	--[[ Initialize external forces: ]]--
	V.inlet_vertices:foreach(liszt (v : M.vertices)
		v.fext = L.vec3f({.01, 0, 0})
	end)

	V.outlet_vertices:foreach(liszt (v : M.vertices)
		v.fext = L.vec3f({-.01, 0, 0})
	end)

	--[[ Initialize acceleration based on initial forces ]]--
	M.vertices:foreach(liszt (v : M.vertices)
		v.a_n = (v.fext - v.fint) / v.mass
	end)

	--[[ Initialize edge lengths]]--
	M.edges:foreach(liszt (e : M.edges)
		var v1 = e.head
		var v2 = e.tail

		var distance = v1.position - v2.position
		var length   = L.length(distance)

		e.initialEdgeLength = length
		e.currEdgeLength    = length
	end)

	--[[ MAIN LOOP: ]]--
	local t_n = 0

	while (t_n < tmax - .01 * dt_n_h) do

		--Update half time:  t^{n+1/2} = t^n + 1/2*deltat^{n+1/2}
	  	t_n_h = t_n + dt_n_h/2 

		-- nodal velocity update function depends on changing t_n
		local liszt update_nodal_velocities (v : M.vertices)
			v.v_n_h = v.v_n + L.float(t_n_h - t_n) * v.a_n
		end

		--[[ Execute! ]]--
		M.vertices:foreach(reset_internal_forces)

		M.vertices:foreach(update_nodal_velocities)
		M.vertices:foreach(update_pos_and_disp)

		M.edges:foreach(calc_edge_disp)
		M.edges:foreach(calc_internal_force)

		M.vertices:foreach(compute_accel)
		M.vertices:foreach(update_previous_velocity)
		M.vertices:foreach(update_velocity)

		-- Time update: t^n = t^{n-1} + deltat^{n-1/2}
		t_n = t_n + dt_n_h
	end
end

main()

-- Print results
V.position:Print()


