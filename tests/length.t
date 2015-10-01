import "ebb.liszt"

local assert = L.assert
local dot = L.dot
local length = L.length
local R = L.NewRelation { name="R", size=5 }


local v1 = L.Constant(L.vec3f, {1, 2, 3})
local v2 = L.Constant(L.vec3i, {1, 2, 3})
local v3 = L.Constant(L.vector(L.float, 1), {7})
local v4 = L.Constant(L.vector(L.int,   1), {0})

local sqrt = terralib.includec('math.h').sqrt
local ans1 = sqrt(1 + 4 + 9)
local ans2 = sqrt(4 + 16 + 36)

local liszt test_dot (r : R)
    assert(length(v1) == ans1) -- float(3)
    assert(length(v2) == ans1) -- int(3)
    assert(length(v3) == 7) -- float(1)
    assert(length(v4) == 0) -- int(1)
    
    var sum = v1 + v2
    assert(length(sum) == ans2) -- test working with local variables
    assert(length(v1 + v2) == ans2) -- test working with expressions
end
R:foreach(test_dot)
