import "compiler/liszt"
local LMesh = terralib.require("compiler/liblmesh")
local mesh = LMesh.Load("examples/mesh.lmesh")

local fail_assert = liszt_kernel(f in mesh.faces)
    L.assert(2 + 2 == 5)
end
fail_assert()

