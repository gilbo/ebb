import "compiler.liszt"
local LMesh = L.require "domains.lmesh"
local mesh = LMesh.Load("examples/mesh.lmesh")


local liszt fail_assert (f : mesh.faces)
    var x = 5
    L.assert(x == 4)
end

mesh.faces:map(fail_assert)
