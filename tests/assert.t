import "compiler/liszt"

mesh  = LoadMesh("examples/mesh.lmesh")

local pass_assert = liszt_kernel(f)
    assert(true)
end

mesh.faces:map(pass_assert)
assert(true)
