import 'compiler.liszt'
local length = L.length

-- Test code
local PN = terralib.require 'compiler.pathname'
local LMesh = terralib.require "compiler.lmesh"
local Particle = terralib.require "compiler.particle"
terralib.linklibrary("examples/vdb.a")
local VDB = terralib.includec("examples/vdb.h")
local M = LMesh.Load(PN.scriptdir():concat("fem_mesh.lmesh"):tostring())

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

local compute_step = liszt kernel(e in M.edges)
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

local propagate_temp = liszt kernel(p in M.vertices)
	p.temperature = p.temperature + .01 * p.flux / p.jacobistep
end

local clear = liszt kernel(p in M.vertices)
	p.flux = 0
	p.jacobistep = 0
end

local x = L.NewMacro(function(v) return liszt `L.dot(v, {1, 0, 0}) end)
local y = L.NewMacro(function(v) return liszt `L.dot(v, {0, 1, 0}) end)
local z = L.NewMacro(function(v) return liszt `L.dot(v, {0, 0, 1}) end)
local color = VDB.vdb_color
local line = VDB.vdb_line
local visualize = liszt kernel(e in M.edges)
    var ave_temp = (e.head.temperature + e.tail.temperature) * 0.5 / 3
    if ave_temp > 0.5 then
        color(1, 0, 1 - (ave_temp - 0.5) * 2)
    else
        color(ave_temp * 2, 0, 1)
    end
    var p1 = e.head.position
    var p2 = e.tail.position
    line(x(p1), y(p1), z(p1), x(p2), y(p2), z(p2))
end

for i = 1, 20000 do
	compute_step()
	propagate_temp()
    -- compute_deltas()
    -- propagate_point_gradients()
    -- interpolate_gradients()
    -- assign_particle_gradients()
    if i % 20 == 0 then
        VDB.vdb_frame()
        VDB.vdb_begin()
        visualize()
        VDB.vdb_end()
    end
	clear()
end

M.vertices.temperature:print()

