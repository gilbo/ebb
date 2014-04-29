import "compiler.liszt"

local LMesh = L.require "domains.lmesh"
local mesh = LMesh.Load("examples/mesh.lmesh")

local pass_assert = liszt kernel(f : mesh.faces)
    L.assert(true)
end
pass_assert(mesh.faces)
