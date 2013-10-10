import "compiler/liszt"

mesh  = LoadMesh("examples/mesh.lmesh")

local fail_assert = liszt_kernel(f)
    var x = 5
    assert(x == 4)
end

mesh.faces:map(fail_assert)

