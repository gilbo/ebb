import "compiler.liszt"
local LMesh = terralib.require "compiler.lmesh"
local mesh = LMesh.Load("examples/mesh.lmesh")


local fail_assert = liszt_kernel(f in mesh.faces)
    var x = 5
    L.assert(x == 4)
end

fail_assert(mesh.faces)
