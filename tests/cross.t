--DISABLE-PARTITIONED
-- The MIT License (MIT)
-- 
-- Copyright (c) 2015 Stanford University.
-- All rights reserved.
-- 
-- Permission is hereby granted, free of charge, to any person obtaining a
-- copy of this software and associated documentation files (the "Software"),
-- to deal in the Software without restriction, including without limitation
-- the rights to use, copy, modify, merge, publish, distribute, sublicense,
-- and/or sell copies of the Software, and to permit persons to whom the
-- Software is furnished to do so, subject to the following conditions:
-- 
-- The above copyright notice and this permission notice shall be included
-- in all copies or substantial portions of the Software.
-- 
-- THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
-- IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
-- FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
-- AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
-- LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
-- FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
-- DEALINGS IN THE SOFTWARE.
import 'ebb'
local L = require 'ebblib'

local assert = L.assert
local cross  = L.cross
local R = L.NewRelation { name="R", size=5 }


local v1 = L.Constant(L.vec3f, {1, 2, 3})
local v2 = L.Constant(L.vec3f, {5, 7, 11})

local v3 = L.Constant(L.vec3i, {1, 2, 3})
local v4 = L.Constant(L.vec3i, {5, 7, 11})

local test_cross = ebb(r : R)
    assert(cross(v1, v2) == {1, 4, -3}) -- simple test
    assert(cross(v3, v4) == {1, 4, -3}) -- int only
    assert(cross(v1, v4) == {1, 4, -3}) -- cross types
    
    var expr = v1 + 2 * v2
    assert(cross(v1, expr) == {2, 8, -6}) -- test working with local variables
    assert(cross(v1, v1 + 2 * v2) == {2, 8, -6}) -- test working with expressions
end
R:foreach(test_cross)