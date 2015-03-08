import "compiler.liszt"

local LMesh = L.require "domains.lmesh"
local M = LMesh.Load("examples/mesh.lmesh")

local V = M.vertices
local P = V.position

local loc_data = {}
function init_loc_data (loc_data)
	P:MoveTo(L.CPU)
	local Pdata = P:DataPtr()
	for i = 0, V:Size() - 1 do
		loc_data[i] = {Pdata[i].d[0], Pdata[i].d[1], Pdata[i].d[2]}
	end
	P:MoveTo(L.default_processor)
end
init_loc_data(loc_data)

function shift(x,y,z)
	local liszt shift_func (v : M.vertices)
	    v.position += {x,y,z}
	end
	M.vertices:map(shift_func)

	P:MoveTo(L.CPU)
	local Pdata = P:DataPtr()
	for i = 0, V:Size() - 1 do

		local v = Pdata[i]
		local d = loc_data[i]

		d[1] = d[1] + x
		d[2] = d[2] + y
		d[3] = d[3] + z

		--print("Pos " .. tostring(i) .. ': (' .. tostring(v[0]) .. ', ' .. tostring(v[1]) .. ', ' .. tostring(v[2]) .. ')')
		--print("Loc " .. tostring(i) .. ': (' .. tostring(d[1]) .. ', ' .. tostring(d[2]) .. ', ' .. tostring(d[3]) .. ')')
		assert(v.d[0] == d[1])
		assert(v.d[1] == d[2])
		assert(v.d[2] == d[3])
	end
	P:MoveTo(L.default_processor)
end

shift(0,0,0)
shift(5,5,5)
shift(-1,6,3)

---------------------------------
--  Centered Matrix reduction: --
---------------------------------
local F = M.faces

F:NewField("mat", L.mat3d)

local liszt m_set(f : F)
	var d = L.double(L.id(f))
	f.mat = {{d, 0.0, 0.0},
             {0.0, d, 0.0},
             {0.0, 0.0, d}}
end

local liszt m_reduce_centered (f : F)
	f.mat += {
		{.11, .11, .11},
		{.22, .22, .22},
		{.33, .33, .33}
	}
end

F:map(m_set)
F:map(m_reduce_centered)

F.mat:print()

-----------------------------------
--  Uncentered Matrix reduction: --
-----------------------------------
-- This will produce the invocation
-- of a second reduction kernel on the GPU runtime
local E = M.edges

V:NewField("mat", L.mat3d)

local liszt m_set_v(v : V)
	var d = L.double(L.id(v))
	v.mat = {{d, 0.0, 0.0},
             {0.0, d, 0.0},
             {0.0, 0.0, d}}
end
local liszt m_reduce_uncentered (e : E)
	e.head.mat += .5*{
		{.11, .11, .11},
		{.22, .22, .22},
		{.33, .33, .33}
	}
	e.tail.mat += .5*{
		{.11, .11, .11},
		{.22, .22, .22},
		{.33, .33, .33}
	}
end

V:map(m_set_v)
E:map(m_reduce_uncentered)

V.mat:print()