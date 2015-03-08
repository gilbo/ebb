import "compiler.liszt"
require "tests/test"

local assert, dot = L.assert, L.dot
local LMesh = L.require "domains.lmesh"
local mesh = LMesh.Load("examples/mesh.lmesh")


local v1 = L.Constant(L.vec3f, {1, 2, 3})
local v2 = L.Constant(L.vec3f, {5, 7, 11})

local v3 = L.Constant(L.vector(L.float, 1), {7})
local v4 = L.Constant(L.vector(L.int, 1), {0})

local v5 = L.Constant(L.vec3i, {1, 2, 3})
local v6 = L.Constant(L.vec3i, {5, 7, 11})

local test_dot = liszt(f : mesh.faces)
    assert(dot(v1, v2) == 52) -- simple test
    assert(dot(v3, v4) == 0) -- type conversion, length-1
    assert(dot(v1, v1) == 14) -- vector with itself
    assert(dot(v5, v6) == 52) -- int only
    
    var sum = v1 + v2
    assert(dot(v1, sum) == 6 + 18 + 42) -- test working with local variables
    assert(dot(v1, v1 + v2) == 6 + 18 + 42) -- test working with expressions
end
mesh.faces:map(test_dot)



test.fail_function(function()
  local liszt t(f : mesh.faces)
    assert(dot(v1, v3) == 7)
  end
  mesh.faces:map(t)
end, "must have equal dimensions")

local vb = L.Constant(L.vec3b, {true, true, false})
test.fail_function(function()
  local liszt t(f : mesh.faces)
    assert(dot(v1, vb) == 52)
  end
  mesh.faces:map(t)
end, "must be numeric vectors")
