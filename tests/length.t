import "compiler/liszt"

local assert = L.assert
local dot = L.dot
local length = L.length
mesh = LoadMesh("examples/mesh.lmesh")

local v1 = Vector.new(L.float, {1, 2, 3})
local v2 = Vector.new(L.int, {1, 2, 3})
local v3 = Vector.new(L.float, {7})
local v4 = Vector.new(L.int, {0})

local sqrt = terralib.includec('math.h').sqrt
local ans1 = sqrt(1 + 4 + 9)
local ans2 = sqrt(4 + 16 + 36)

local test_dot = liszt_kernel(f)
    assert(length(v1) == ans1) -- float(2)
    assert(length(v2) == ans1) -- int(2)
    assert(length(v3) == 7) -- float(1)
    assert(length(v4) == 0) -- int(1)
    
    var sum = v1 + v2
    assert(length(sum) == ans2) -- test working with local variables
    assert(length(v1 + v2) == ans2) -- test working with expressions
end

mesh.faces:map(test_dot)
