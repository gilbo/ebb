import "compiler.liszt"
local LMesh = L.require "domains.lmesh"
local mesh = LMesh.Load("examples/mesh.lmesh")


local liszt kernel fail_assert (f : mesh.faces)
    var x = 5
    L.assert(x == 4)
end

fail_assert(mesh.faces)
