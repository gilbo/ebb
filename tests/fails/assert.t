import "compiler.liszt"
local LMesh = terralib.require "compiler.lmesh"
local mesh = LMesh.Load("examples/mesh.lmesh")

local fail_assert = liszt_kernel(f in mesh.faces)
    L.assert(false)
end
fail_assert()
