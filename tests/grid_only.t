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

--error('NEED TO WRITE GRID TESTS; DO WHILE REWRITING GRID DOMAIN')

local rel2 = L.NewRelation { name="rel2", dims = {4,6} }
local rel3 = L.NewRelation { name="rel3", dims = {6,4,4} }
rel2:SetPartitions{2,2}
rel3:SetPartitions{2,2,2}

test.eq(rel2:isGrid(), true)
test.eq(rel3:isGrid(), true)
test.aeq(rel2:Dims(), {4,6})
test.aeq(rel3:Dims(), {6,4,4})



-- test loading
rel2:NewField('v2',L.vec2d):Load(function(x,y)   return {2*x,y}   end)
rel3:NewField('v3',L.vec3d):Load(function(x,y,z) return {3*x,y,z} end)


-- test loading from a list
local tbl2 = {{ 1, 2, 3, 4},{ 5, 6, 7, 8},{ 9,10,11,12},
              {13,14,15,16},{17,18,19,20},{21,22,23,24}}
rel2:NewField('f2',L.double):Load(tbl2)
-- test that dumping preserves list/structure
test.rec_aeq(rel2.f2:Dump({}),tbl2)

-- test indexing consistency
rel2:NewField('f2func',L.double):Load(function(x,y)
  return 4*y + x + 1
end)
local ebb f2consistency( r : rel2 )
  L.assert(r.f2 == r.f2func)
end
rel2:foreach(f2consistency)


-- Test Periodicity
local prel2 = L.NewRelation {
  name = "prel2",
  dims = {6,8},
  periodic={true,true}
}
prel2:SetPartitions{2,2}
prel2:NewField('cid', L.vec2i):Load(function(x,y) return {x,y} end)
local ebb test_wrap ( r : prel2 )
  var off = L.Affine(prel2, {{1,0,1},
                             {0,1,1}}, r)
  L.assert( (r.cid[0]+1) % 6 == off.cid[0] and
            (r.cid[1]+1) % 8 == off.cid[1] )
end
prel2:foreach(test_wrap)


