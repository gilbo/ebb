import "compiler/liszt"

local assert = L.assert
local dot = L.dot
mesh = LoadMesh("examples/mesh.lmesh")

local v1 = Vector.new(L.float, {1, 2, 3})
local v2 = Vector.new(L.float, {5, 7, 11})

local v3 = Vector.new(L.float, {7})
local v4 = Vector.new(L.int, {0})

local v5 = Vector.new(L.int, {1, 2, 3})
local v6 = Vector.new(L.int, {5, 7, 11})

local test_dot = liszt_kernel(f)
    assert(dot(v1, v2) == 52) -- simple test
    assert(dot(v3, v4) == 0) -- type conversion, length-1
    assert(dot(v1, v1) == 14) -- vector with itself
    assert(dot(v5, v6) == 52) -- int only
    
    var sum = v1 + v2
    assert(dot(v1, sum) == 6 + 18 + 42) -- test working with local variables
    assert(dot(v1, v1 + v2) == 6 + 18 + 42) -- test working with expressions
end

mesh.faces:map(test_dot)
