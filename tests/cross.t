import "compiler.liszt"

local assert = L.assert
local cross  = L.cross
local R = L.NewRelation { name="R", size=5 }


local v1 = L.Constant(L.vec3f, {1, 2, 3})
local v2 = L.Constant(L.vec3f, {5, 7, 11})

local v3 = L.Constant(L.vec3i, {1, 2, 3})
local v4 = L.Constant(L.vec3i, {5, 7, 11})

local test_cross = liszt(r : R)
    assert(cross(v1, v2) == {1, 4, -3}) -- simple test
    assert(cross(v3, v4) == {1, 4, -3}) -- int only
    assert(cross(v1, v4) == {1, 4, -3}) -- cross types
    
    var expr = v1 + 2 * v2
    assert(cross(v1, expr) == {2, 8, -6}) -- test working with local variables
    assert(cross(v1, v1 + 2 * v2) == {2, 8, -6}) -- test working with expressions
end
R:map(test_cross)