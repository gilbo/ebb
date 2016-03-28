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


local rel2 = L.NewRelation { name="rel2", dims = {4,6} }
local rel3 = L.NewRelation { name="rel3", dims = {6,4,4} }
rel2:SetPartitions{2,2}
rel3:SetPartitions{2,2,2}

rel2:NewField('ones', L.int):Load(1)
rel3:NewField('ones', L.int):Load(1)

local glob_one_count_1 = L.Global(L.int, 0)
local glob_one_count_2 = L.Global(L.int, 0)
local glob_one_count_3 = L.Global(L.int, 0)

local glob_raw_count_1 = L.Global(L.int, 0)
local glob_raw_count_2 = L.Global(L.int, 0)
local glob_raw_count_3 = L.Global(L.int, 0)

local ebb count_one2 ( c : rel2 )
  L.assert(c.ones == 1)
  glob_one_count_2 += c.ones
end
local ebb count_one3 ( c : rel3 )
  L.assert(c.ones == 1)
  glob_one_count_3 += c.ones
end

local ebb count_raw2 ( c : rel2 )
  glob_raw_count_2 += 1
end
local ebb count_raw3 ( c : rel3 )
  glob_raw_count_3 += 1
end

rel2:foreach(count_one2)
rel3:foreach(count_one3)

rel2:foreach(count_raw2)
rel3:foreach(count_raw3)

test.eq(rel2:Size(), glob_one_count_2:get())
test.eq(rel2:Size(), glob_raw_count_2:get())
test.eq(rel3:Size(), glob_one_count_3:get())
test.eq(rel3:Size(), glob_raw_count_3:get())

