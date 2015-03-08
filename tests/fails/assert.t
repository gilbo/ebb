import "compiler.liszt"
local LMesh = L.require "domains.lmesh"
local mesh = LMesh.Load("examples/mesh.lmesh")

local liszt fail_assert (f : mesh.faces)
    L.assert(false)
end
mesh.faces:map(fail_assert)
