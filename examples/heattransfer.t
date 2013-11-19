import 'compiler.liszt'
local length, lprint = L.length, L.print

-- Test code
local M = L.initMeshRelationsFromFile("examples/rmesh.lmesh")

local init_to_zero = terra (mem : &float, i : int) mem[0] = 0 end
local init_temp    = terra (mem : &float, i : int)
	if i == 0 then
		mem[0] = 1000
	else
		mem[0] = 0
	end
end

M.vertices:NewField('flux',        L.float):LoadFromCallback(init_to_zero)
M.vertices:NewField('jacobistep',  L.float):LoadFromCallback(init_to_zero)
M.vertices:NewField('temperature', L.float):LoadFromCallback(init_temp)

local compute_step = liszt_kernel(e in M.edges)
	var v1   = e.head
	var v2   = e.tail
	var dp   = v1.position - v2.position
	var dt   = v1.temperature - v2.temperature
	var step = 1.0 / length(dp)

	v1.flux = v1.flux - dt * step
	v2.flux = v2.flux + dt * step

	v1.jacobistep = v1.jacobistep + step
	v2.jacobistep = v2.jacobistep + step
end

local propagate_temp = liszt_kernel (p in M.vertices)
	p.temperature = p.temperature + .01 * p.flux / p.jacobistep
end

local clear = liszt_kernel (p in M.vertices)
	p.flux = 0
	p.jacobistep = 0
end

for i = 1, 1000 do
	compute_step()
	propagate_temp()
	clear()
end

M.vertices.temperature:print()

