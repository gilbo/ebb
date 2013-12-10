import 'compiler.liszt'
local length = L.length

-- Test code
local PN = terralib.require 'compiler.pathname'
local LMesh = terralib.require "compiler.lmesh"
local Particle = terralib.require "compiler.particle"
local c = terralib.require "compiler.c"
terralib.linklibrary("examples/vdb.a")
local VDB = terralib.includec("examples/vdb.h")
local M = LMesh.LoadUniformGrid(100, {10, 5, 5}, {0, 0, 0}, {1, 0.5, 0.5})

local init_to_zero = terra (mem : &float, i : int) mem[0] = 0 end
local init_to_false = terra (mem : &bool, i : int) mem[0] = false end
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

M.particles.position:LoadFromCallback(init_random)
M.vertices:NewField('flux',        L.float):LoadFromCallback(init_to_zero)
M.vertices:NewField('jacobistep',  L.float):LoadFromCallback(init_to_zero)
M.vertices:NewField('degree',      L.float):LoadFromCallback(init_to_zero)
M.vertices:NewField('temperature', L.float):LoadFromCallback(init_temp)
M.vertices:NewField('rawgradient', L.vector(L.float, 3)):LoadFromCallback(init_to_zero_vec)
M.vertices:NewField('gradient',    L.vector(L.float, 3)):LoadFromCallback(init_to_zero_vec)
M.particles:NewField('normalization', L.float):LoadFromCallback(init_to_zero)
M.particles:NewField('rawgradient',   L.vector(L.float, 3)):LoadFromCallback(init_to_zero_vec)
M.particles:NewField('gradient',      L.vector(L.float, 3)):LoadFromCallback(init_to_zero_vec)
M.edges:NewField('hasparticles', L.bool):LoadFromCallback(init_to_false)
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

local compute_deltas = liszt kernel(e in M.edges)
    var dT = e.head.temperature - e.tail.temperature
    var dr = e.head.position - e.tail.position
    var grad = dT / length(dr) * dr / length(dr)
    e.head.rawgradient += grad
    e.tail.rawgradient += grad
    e.head.degree += 1
    e.tail.degree += 1
end

local propagate_point_gradients = liszt kernel(v in M.vertices)
    v.gradient = v.rawgradient / v.degree
end

local interpolate_gradients = liszt kernel(p in M.particles)
    if L.id(p.cell) ~= 0 then
        for v in p.cell.vertices do
            var weight = 1 / L.length(v.position - p.position)
            p.rawgradient += weight * v.gradient
            p.normalization += weight
        end
    end
end

local advect_particles = liszt kernel(p in M.particles)
    if L.id(p.cell) ~= 0 then
        var gradient = p.rawgradient / p.normalization
        p.position += 0.0005 * gradient
    end
end

local color_cells = liszt kernel(p in M.particles)
    if L.id(p.cell) ~= 0 then
        for e in p.cell.edges do
            e.hasparticles = true
        end
    end
end

local x = L.NewMacro(function(v) return liszt `L.dot(v, {1, 0, 0}) end)
local y = L.NewMacro(function(v) return liszt `L.dot(v, {0, 1, 0}) end)
local z = L.NewMacro(function(v) return liszt `L.dot(v, {0, 0, 1}) end)
local color = VDB.vdb_color
local line = VDB.vdb_line
local point = VDB.vdb_point

local visualize_edges = liszt kernel(e in M.edges)
    var r : L.float = 1.0
    var g : L.float = 0.0
    var b : L.float = 1.0

    var ave_temp = (e.head.temperature + e.tail.temperature) * 0.5 / 3
    if ave_temp > 0.5 then
        b = 1 - (ave_temp - 0.5) * 2
    else
        r = ave_temp * 2
    end

    if e.hasparticles then
        g = 1
    end

    color(r, g, b)
    var p1 = e.head.position
    var p2 = e.tail.position
    line(x(p1), y(p1), z(p1), x(p2), y(p2), z(p2))
end

local visualize_particles = liszt kernel(p in M.particles)
    var pos = p.position
    point(x(pos), y(pos), z(pos))
end

--[[
local visualize_gradients = liszt kernel(v in M.vertices)
    var start = v.position
    var g = v.gradient * 0.01
    if L.length(g) > 0.1 then g /= L.length(g) end
    var finish = start + g
    line(x(start), y(start), z(start), x(finish), y(finish), z(finish))
end
]]

local function visualize()
    VDB.vdb_begin()
    VDB.vdb_frame()
    visualize_edges()
    color(1, 1, 1)
    visualize_particles()
--[[color(1, 0, 0)
    visualize_gradients()]]
    VDB.vdb_end()
end

local clear_vertices = liszt kernel(p in M.vertices)
    p.flux = 0
    p.jacobistep = 0
    p.rawgradient = 0
    p.degree = 0
end

local clear_particles = liszt kernel(p in M.particles)
    p.rawgradient = 0
    p.normalization = 0
end

local clear_edges = liszt kernel(e in M.edges)
    e.hasparticles = false
end

local function clear()
    clear_vertices()
    clear_particles()
    clear_edges()
end

for i = 1, 10000 do
	compute_step()
	propagate_temp()
    compute_deltas()
    propagate_point_gradients()
    interpolate_gradients()
    advect_particles()
    M:updateParticles()
    color_cells()
    if i % 5 == 0 then visualize() end
    if i % 1000 == 0 then print(i) end
	clear()
end

M.vertices.temperature:print()
