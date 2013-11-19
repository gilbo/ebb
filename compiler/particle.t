import "compiler.liszt"
local Particle = {}
package.loaded["compiler.particle"] = Particle
local print, dot = L.print, L.dot

function Particle.init(mesh, numParticles)
    mesh.particles = LDB.NewRelation(numParticles, "particles")
    local position = mesh.particles:NewField('position', L.vector(L.float, 3))
    mesh.particles:NewField('cell', mesh.cells):LoadFromCallback(terra(mem : &uint64, i : uint)
        mem[0] = 0
    end)
    return position
end

function Particle.update(mesh)
    -- TODO: collision detection on mesh using acceleration structure
    (liszt kernel(p in mesh.particles)
        if dot(p.position, {1, 0, 0}) > 1.0 then
            p.cell = 1
        else
            p.cell = 0
        end
    end)()
end
