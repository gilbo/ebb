import "compiler/liszt"

mesh  = LoadMesh("examples/mesh.lmesh")

local fail_assert = liszt_kernel(f)
    assert(2 + 2 == 5)
end

mesh.faces:map(fail_assert)

