import 'compiler/liszt'
LDB = terralib.require('include/ldb')
local print = L.print

local M = L.initMeshRelationsFromFile("examples/rmesh.lmesh")

local center_of_cell_1 = terra (mem : &vector(float, 3), i : uint)
    mem[0] = vectorof(float, 0.5, 0.5, 0.5)
end

local numParticles = 1
M.particles = LDB.NewRelation(numParticles, "particles")
M.particles:NewField('position', L.vector(L.float, 3)):LoadFromCallback(center_of_cell_1)

(liszt kernel(p in M.particles)
    p.position = {1.5, 0.5, 0.5}
end)()

-- update_particles(L.getParticleHackPointer())    

M.particles.position:print()

