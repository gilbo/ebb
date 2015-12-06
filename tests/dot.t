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
require "tests/test"

local assert, dot = L.assert, L.dot
local R = L.NewRelation { name="R", size=5 }


local v1 = L.Constant(L.vec3f, {1, 2, 3})
local v2 = L.Constant(L.vec3f, {5, 7, 11})

local v3 = L.Constant(L.vector(L.float, 1), {7})
local v4 = L.Constant(L.vector(L.int, 1), {0})

local v5 = L.Constant(L.vec3i, {1, 2, 3})
local v6 = L.Constant(L.vec3i, {5, 7, 11})

local test_dot = ebb(r : R)
    assert(dot(v1, v2) == 52) -- simple test
    assert(dot(v3, v4) == 0) -- type conversion, length-1
    assert(dot(v1, v1) == 14) -- vector with itself
    assert(dot(v5, v6) == 52) -- int only
    
    var sum = v1 + v2
    assert(dot(v1, sum) == 6 + 18 + 42) -- test working with local variables
    assert(dot(v1, v1 + v2) == 6 + 18 + 42) -- test working with expressions
end
R:foreach(test_dot)



test.fail_function(function()
  local ebb t(r : R)
    assert(dot(v1, v3) == 7)
  end
  R:foreach(t)
end, "must have equal dimensions")

local vb = L.Constant(L.vec3b, {true, true, false})
test.fail_function(function()
  local ebb t(r : R)
    assert(dot(v1, vb) == 52)
  end
  R:foreach(t)
end, "must be numeric vectors")
