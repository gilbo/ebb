import "compiler.liszt"
require "tests/test"

local LMesh = L.require "domains.lmesh"
local mesh = LMesh.Load("examples/mesh.lmesh")

local liszt kernel pass(f : mesh.faces)
    L.assert(L.any({true}))
    L.assert(not L.any({false}))
    L.assert(L.any({true, true, true}))
    L.assert(L.any({false, true, false, false}))
    L.assert(not L.any({false, false, false, false}))
    L.assert(L.all({true}))
    L.assert(L.all({true, true, true}))
    L.assert(not L.all({true, false}))
end
pass(mesh.faces)

assert(L.any(L.NewVector(L.bool, {true})))
assert(not L.any(L.NewVector(L.bool, {false})))
assert(L.any(L.NewVector(L.bool, {true, true, true})))
assert(L.any(L.NewVector(L.bool, {false, true, false, false})))
assert(not L.any(L.NewVector(L.bool, {false, false, false, false})))
assert(L.all(L.NewVector(L.bool, {true})))
assert(L.all(L.NewVector(L.bool, {true, true, true})))
assert(not L.all(L.NewVector(L.bool, {true, false, true})))

