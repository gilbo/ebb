import "compiler/liszt"

local LMesh = terralib.require("compiler/liblmesh")
local mesh = LMesh.Load("examples/mesh.lmesh")

local pass_assert = liszt_kernel(f in mesh.faces)
    L.assert(true)
end
pass_assert()
