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
require 'tests.test'

local R = L.NewRelation { name="R", dims={10,10} }
R:SetPartitions{2,2}

R:NewField('pos', L.vec3d):Load{0.1,0.1,0.1}

local ebb initpos( r : R )
  r.pos = L.vec3d({ L.xid(r), L.yid(r), L.yid(r) })*0.1 - {0.5,0.5,0.5}
end
R:foreach(initpos)

local max_pos = L.Global(L.vec3d, {-10, -10, -10})
local min_pos = L.Global(L.vec3d, { 10,  10,  10})

-- Test max reduction operator
local max_func = ebb (v : R)
	max_pos max= v.pos
end
R:foreach(max_func)
test.aeq(max_pos:get(), {0.4,0.4,0.4})

-- Test min reduction operator
local min_func = ebb (v : R)
	min_pos min= v.pos
end
R:foreach(min_func)
test.aeq(min_pos:get(), {-0.5,-0.5,-0.5})
