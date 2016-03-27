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

local rel2 = L.NewRelation { name="rel2", dims = {2,3} }
local rel3 = L.NewRelation { name="rel3", dims = {3,2,2} }

test.eq(rel2:isGrid(), true)
test.eq(rel3:isGrid(), true)
test.aeq(rel2:Dims(), {2,3})
test.aeq(rel3:Dims(), {3,2,2})


-- Check bad arguments to create a relation with
test.fail_function(function()
  local relbad = L.NewRelation { name="relbad", mode="GRID" }
end, "Grids must specify 'dim' argument")
test.fail_function(function()
  local relbad = L.NewRelation { name="relbad", dims={5} }
end, "a table of 2 to 3 numbers")

-- test loading
rel2:NewField('v2',L.vec2d):Load(function(x,y)   return {2*x,y}   end)
rel3:NewField('v3',L.vec3d):Load(function(x,y,z) return {3*x,y,z} end)

-- test printing
rel2.v2:Print()
rel3.v3:Print()

-- test loading from a list
local tbl2 = {{1,2},{3,4},{5,6}}
local tbl3 = {{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}}
rel2:NewField('f2',L.double):Load(tbl2)
rel3:NewField('f3',L.double):Load(tbl3)
-- test that dumping preserves list/structure
test.rec_aeq(rel2.f2:Dump({}),tbl2)
test.rec_aeq(rel3.f3:Dump({}),tbl3)

-- test indexing consistency
rel3:NewField('f3func',L.double):Load(function(x,y,z)
  return 6*z + 3*y + x + 1
end)
local ebb f3consistency( r : rel3 )
  L.assert(r.f3 == r.f3func)
end
rel3:foreach(f3consistency)


-- Test Periodicity
local prel2 = L.NewRelation {
  name = "prel2",
  dims = {3,4},
  periodic={true,true}
}
prel2:NewField('cid', L.vec2i):Load(function(x,y) return {x,y} end)
local ebb test_wrap ( r : prel2 )
  var off = L.Affine(prel2, {{1,0,1},
                             {0,1,1}}, r)
  L.assert( (r.cid[0]+1) % 3 == off.cid[0] and
            (r.cid[1]+1) % 4 == off.cid[1] )
end
prel2:foreach(test_wrap)


