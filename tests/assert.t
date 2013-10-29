import "compiler/liszt"

local lassert = L.assert
mesh  = LoadMesh("examples/mesh.lmesh")

local pass_assert = liszt_kernel(f)
    lassert(true)
end

mesh.faces:map(pass_assert)
assert(true)
