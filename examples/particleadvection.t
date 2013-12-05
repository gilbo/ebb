import 'compiler.liszt'
local length = L.length

-- Test code
local PN = terralib.require 'compiler.pathname'
local LMesh = terralib.require "compiler.lmesh"
local Particle = terralib.require "compiler.particle"
local c = terralib.require "compiler.c"
terralib.linklibrary("examples/vdb.a")
local VDB = terralib.includec("examples/vdb.h")
local M = LMesh.Load(PN.scriptdir():concat("fem_mesh.lmesh"):tostring())

local init_to_zero = terra (mem : &float, i : int) mem[0] = 0 end
local init_to_zero_vec = terra (mem : &vector(float, 3), i : int)
    mem[0] = vectorof(float, 0, 0, 0)
end
local init_temp    = terra (mem : &float, i : int)
	if i == 0 then
		mem[0] = 1000
	else
		mem[0] = 0
	end
end

local RAND_MAX = constant(32767) -- until we can import the real RAND_MAX
local random = terra()
    return (c.rand() % RAND_MAX) * 1.0 / RAND_MAX
end 
local init_random = terra (mem : &vector(float, 3), i : uint)
    mem[0] = vectorof(float, random(), random() * 0.5, random() * 0.5)
end

Particle.initUniformGrid(M, 100, {10, 5, 5}, {0, 0, 0}, {1, 0.5, 0.5}):LoadFromCallback(init_random)
M.vertices:NewField('flux',        L.float):LoadFromCallback(init_to_zero)
M.vertices:NewField('jacobistep',  L.float):LoadFromCallback(init_to_zero)
M.vertices:NewField('rawgradient', L.vector(L.float, 3)):LoadFromCallback(init_to_zero_vec)
M.particles:NewField('rawgradient', L.vector(L.float, 3)):LoadFromCallback(init_to_zero_vec)
M.vertices:NewField('gradient', L.vector(L.float, 3)):LoadFromCallback(init_to_zero_vec)
M.particles:NewField('gradient', L.vector(L.float, 3)):LoadFromCallback(init_to_zero_vec)
M.vertices:NewField('temperature', L.float):LoadFromCallback(init_temp)
M:updateParticles()

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
local point = VDB.vdb_point
local function visualize()
    (liszt kernel(e in M.edges)
        var ave_temp = (e.head.temperature + e.tail.temperature) * 0.5 / 3
        if ave_temp > 0.5 then
            color(1, 0, 1 - (ave_temp - 0.5) * 2)
        else
            color(ave_temp * 2, 0, 1)
        end
        var p1 = e.head.position
        var p2 = e.tail.position
        line(x(p1), y(p1), z(p1), x(p2), y(p2), z(p2))
    end)()
    color(1, 1, 0)
    (liszt kernel(p in M.particles)
        var pos = p.position
        point(x(pos), y(pos), z(pos))
    end)()
end

for i = 1, 20000 do
	compute_step()
	propagate_temp()
    -- compute_deltas()
    -- propagate_point_gradients()
    -- interpolate_gradients()
    -- assign_particle_gradients()
    if i % 20 == 0 then
        VDB.vdb_begin()
        VDB.vdb_frame()
        visualize()
        VDB.vdb_end()
    end
	clear()
end

M.vertices.temperature:print()
M.vertices.position:print()
