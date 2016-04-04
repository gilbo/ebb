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
require "tests.test"


local R = L.NewRelation { name="R", size=5 }

test.fail_function(function()
  local ebb t(r : R)
    var x = {1, 2, 3} + 4
  end
  R:foreach(t)
end, "incompatible types")

test.fail_function(function()
  local ebb t(r : R)
    var x = {1, 2, 3} / {4, 5, 6}
  end
  R:foreach(t)
end, "invalid types")

test.fail_function(function()
  local ebb t(r : R)
    var x = 5 < true
  end
  R:foreach(t)
end, "invalid types")

local t = {}
local s = {}
test.fail_function(function()
  local ebb t(r : R)
    var x = s < t
  end
  R:foreach(t)
end, "invalid types")
