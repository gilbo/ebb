import "compiler.liszt"
local LMesh = L.require "domains.lmesh"
local mesh = LMesh.Load("examples/mesh.lmesh")

local fail_assert = liszt kernel(f : mesh.faces)
    L.assert(2 + 2 == 5)
end
fail_assert(mesh.faces)

