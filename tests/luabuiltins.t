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

local print, assert, dot, cross, length =
  L.print, L.assert, L.dot, L.cross, L.length
local sqrt = L.sqrt
local R = L.NewRelation { name="R", size=6 }

local v1 = L.Constant(L.vec3f, {1, 2, 3})
local v2 = L.Constant(L.vec3f, {5, 7, 11})

R:foreach(ebb(r : R)
    assert(true)
    print(42)
    assert(dot(v1, v2) == 52)
    assert(cross(v1, v2) == {1, 4, -3})
    assert(length(v1) == sqrt(1 + 4 + 9))
end)

assert(true)
print(42)
--assert(dot(v1, v2) == 52)
--assert(cross(v1, v2) == L.NewVector(L.float, {1, 4, -3}))
--assert(length(v1) == sqrt(1 + 4 + 9))
