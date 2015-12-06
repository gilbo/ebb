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

local ebb test_dot (r : R)
    assert(length(v1) == ans1) -- float(3)
    assert(length(v2) == ans1) -- int(3)
    assert(length(v3) == 7) -- float(1)
    assert(length(v4) == 0) -- int(1)
    
    var sum = v1 + v2
    assert(length(sum) == ans2) -- test working with local variables
    assert(length(v1 + v2) == ans2) -- test working with expressions
end
R:foreach(test_dot)
