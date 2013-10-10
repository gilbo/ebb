import "compiler/liszt"

mesh  = LoadMesh("examples/mesh.lmesh")

local fail_assert = liszt_kernel(f)
    assert(false)
end

mesh.faces:map(fail_assert)

