import 'compiler.liszt'

local PN    = L.require 'lib.pathname'
local LMesh = L.require "domains.lmesh"
local M     = LMesh.Load(PN.scriptdir():concat("rmesh.lmesh"):tostring())

local function init_temp (i)
	if i == 0 then
		return 1000
	else
		return 0
	end
end

M.vertices:NewField('flux',        L.float):Load(0)
M.vertices:NewField('jacobistep',  L.float):Load(0)
M.vertices:NewField('temperature', L.float):Load(init_temp)

local compute_step = liszt kernel(e : M.edges)
	var v1   = e.head
	var v2   = e.tail
	var dp   = v1.position - v2.position
	var dt   = v1.temperature - v2.temperature
	var step = 1.0 / L.length(dp)

	v1.flux += -dt * step
	v2.flux +=  dt * step

	v1.jacobistep += step
	v2.jacobistep += step
end

local propagate_temp = liszt kernel (p : M.vertices)
	p.temperature += L.float(.01) * p.flux / p.jacobistep
end

local clear = liszt kernel (p : M.vertices)
	p.flux       = 0
	p.jacobistep = 0
end

for i = 1, 1000 do
	compute_step(M.edges)
	propagate_temp(M.vertices)
	clear(M.vertices)
end

M.vertices.temperature:print()

