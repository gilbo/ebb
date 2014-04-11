import "compiler.liszt"

local assert = L.assert
local cross  = L.cross
local LMesh = L.require "domains.lmesh"
local mesh = LMesh.Load("examples/mesh.lmesh")


local v1 = L.NewVector(L.float, {1, 2, 3})
local v2 = L.NewVector(L.float, {5, 7, 11})

local v3 = L.NewVector(L.int, {1, 2, 3})
local v4 = L.NewVector(L.int, {5, 7, 11})

local test_cross = liszt_kernel(f : mesh.faces)
    assert(cross(v1, v2) == {1, 4, -3}) -- simple test
    assert(cross(v3, v4) == {1, 4, -3}) -- int only
    assert(cross(v1, v4) == {1, 4, -3}) -- cross types
    
    var expr = v1 + 2 * v2
    assert(cross(v1, expr) == {2, 8, -6}) -- test working with local variables
    assert(cross(v1, v1 + 2 * v2) == {2, 8, -6}) -- test working with expressions
end
test_cross(mesh.faces)