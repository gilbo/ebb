import "compiler.liszt"

local print, assert, dot, cross, length = L.print, L.assert, L.dot, L.cross, L.length
local sqrt = terralib.includec('math.h').sqrt
local LMesh = L.require "domains.lmesh"
local mesh = LMesh.Load("examples/mesh.lmesh")

local v1 = L.Constant(L.vec3f, {1, 2, 3})
local v2 = L.Constant(L.vec3f, {5, 7, 11})

mesh.faces:map(liszt(f : mesh.faces)
    assert(true)
    print(42)
    assert(dot(v1, v2) == 52)
    assert(cross(v1, v2) == {1, 4, -3})
    assert(length(v1) == sqrt(1 + 4 + 9))
end)

assert(true)
print(42)
--assert(dot(v1, v2) == 52)
--assert(cross(v1, v2) == L.NewVector(L.float, {1, 4, -3}))
--assert(length(v1) == sqrt(1 + 4 + 9))
