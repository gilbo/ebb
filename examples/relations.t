local L = terralib.require("include/liszt")

-- Test code
local M = L.initMeshRelationsFromFile("examples/rmesh.lmesh")

local init_to_zero = terra (mem : &float, i : int)
	mem[0] = 0
end

local init_temp = terra (mem : &float, i : int)
	if i == 0 then
		mem[0] = 1000
	else
		mem[0] = 0
	end
end

M.vertices:NewField('flux',        L.float):LoadFromCallback(init_to_zero)
M.vertices:NewField('jacobistep',  L.float):LoadFromCallback(init_to_zero)
M.vertices:NewField('temperature', L.float):LoadFromCallback(init_temp)

local C = terralib.includecstring([[
#include "stdio.h"
#include "math.h"
]])

terra print_temp()
	C.printf("TEMPERATURE:\n")
	for _i = 0, M.vertices._size do
		C.printf("    T(%d): %f\n", _i, M.vertices.temperature.data[_i])
	end
	C.printf("\n")
end

terra print_flux()
	C.printf("FLUX:\n")
	for _i = 0, M.vertices._size do
		C.printf("    F(%d): %f\n", _i, M.vertices.flux.data[_i])
	end
	C.printf("\n")
end

terra loop ()
	for _i = 0, M.edges._size do
		var v1_i = M.edges.head.data[_i]
		var v2_i = M.edges.tail.data[_i]

		var v1p : vector(float, 3) = M.vertices.position.data[v1_i]
		var v2p : vector(float, 3) = M.vertices.position.data[v2_i]

		var dp  = v1p - v2p

		var t1 = M.vertices.temperature.data[v1_i]
		var t2 = M.vertices.temperature.data[v2_i]
		var dt : float = t1 - t2

		var length_dp = dp[0] * dp[0] + dp[1] * dp[1] + dp[2] * dp[2]
		length_dp = C.sqrt(length_dp)

		var step = 1.0 / length_dp

		M.vertices.flux.data[v1_i] = M.vertices.flux.data[v1_i] - dt * step
		M.vertices.flux.data[v2_i] = M.vertices.flux.data[v2_i] + dt * step

		M.vertices.jacobistep.data[v1_i] = M.vertices.jacobistep.data[v1_i] + step
		M.vertices.jacobistep.data[v2_i] = M.vertices.jacobistep.data[v2_i] + step

		--[[
		C.printf("ITER %d:\n", _i)
		C.printf("v1: %d (%f %f %f)\n", v1_i, v1p[0], v1p[1], v1p[2])
		C.printf("v2: %d (%f %f %f)\n", v2_i, v2p[0], v2p[1], v2p[2])

		C.printf("step : %f\n", step)
		C.printf("dt   : %f\n", dt)
		C.printf("\n")
		]]--
	end

	for i = 0, M.vertices._size do
		M.vertices.temperature.data[i] = M.vertices.temperature.data[i] + .01 * M.vertices.flux.data[i] / M.vertices.jacobistep.data[i]
	end

	for i = 0, M.vertices._size do
		M.vertices.flux.data[i]       = 0
		M.vertices.jacobistep.data[i] = 0
	end

end

local i = 0
while i < 1000 do
	loop()
	i = i + 1
end

print_temp()

--[[

local step = liszt_kernel (e in edges)
	var v1 = e.head
	var v2 = e.tail
	var dp = v1.position - v2.position
	var dt = v1.temperature - v2.temperature
	var step = 1.0 / length(dp)

	v1.flux = v1.flux + dt * step
	v2.flux = v2.flux - dt * step

	v1.jacobistep = v1.jacobistep + step
	v2.jacobistep = v2.jacobistep + step
end

local propogate_temp = liszt_kernel (p in vertices)
	p.temperature = p.temperature + .01 * p.flux / p.jacobistep
end

local clear = liszt_kernel (p in vertices)
	p.flux = 0
	p.jacobistep = 0
end

while i < 1000 do
	step()
	propogate_temp()
	clear()
end


]]