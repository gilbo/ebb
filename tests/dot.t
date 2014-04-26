import "compiler.liszt"
require "tests/test"

local assert, dot = L.assert, L.dot
local LMesh = L.require "domains.lmesh"
local mesh = LMesh.Load("examples/mesh.lmesh")


local v1 = L.NewVector(L.float, {1, 2, 3})
local v2 = L.NewVector(L.float, {5, 7, 11})

local v3 = L.NewVector(L.float, {7})
local v4 = L.NewVector(L.int, {0})

local v5 = L.NewVector(L.int, {1, 2, 3})
local v6 = L.NewVector(L.int, {5, 7, 11})

local test_dot = liszt kernel(f : mesh.faces)
    assert(dot(v1, v2) == 52) -- simple test
    assert(dot(v3, v4) == 0) -- type conversion, length-1
    assert(dot(v1, v1) == 14) -- vector with itself
    assert(dot(v5, v6) == 52) -- int only
    
    var sum = v1 + v2
    assert(dot(v1, sum) == 6 + 18 + 42) -- test working with local variables
    assert(dot(v1, v1 + v2) == 6 + 18 + 42) -- test working with expressions
end
test_dot(mesh.faces)



test.fail_function(function()
  liszt kernel(f : mesh.faces)
    assert(dot(v1, v3) == 7)
  end
end, "must have equal dimensions")

local vb = L.NewVector(L.bool, {true, true, false})
test.fail_function(function()
  liszt kernel(f : mesh.faces)
    assert(dot(v1, vb) == 52)
  end
end, "must be numeric vectors")
