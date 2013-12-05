import "compiler.liszt"
LDB = terralib.require "compiler.ldb"
LMesh = terralib.require "compiler.lmesh"
Particle = terralib.require "compiler.particle"
local print = L.print

local M = LMesh.Load("tests/twocells.lmesh")

local center_of_cell_1 = terra (mem : &vector(float, 3), i : uint)
    mem[0] = vectorof(float, 0.5, 0.5, 0.5)
end

Particle.initUniformGrid(M, 1, {2, 1, 1}):LoadFromCallback(center_of_cell_1)
M:updateParticles()

M.particles.cell:print()

(liszt kernel(p in M.particles)
    p.position = {1.5, 0.5, 0.5}
end)()
M:updateParticles()    

M.particles.cell:print()

