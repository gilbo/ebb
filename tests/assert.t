import "compiler.liszt"

local LMesh = L.require "domains.lmesh"
local mesh = LMesh.Load("examples/mesh.lmesh")

local pass_assert = liszt(f : mesh.faces)
    L.assert(true)
end
mesh.faces:map(pass_assert)
