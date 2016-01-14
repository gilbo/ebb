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

local Grid = require 'ebb.domains.grid'

local g2d  = Grid.NewGrid2d {
  size    = {3,2},
  origin  = {1,4},
  width   = {0.9,1.0},
}

local g3d  = Grid.NewGrid3d {
  size    = {3,2,2},
  origin  = {1,4,7},
  width   = {0.9,1.0,0.8},
}


test.eq(g2d:xSize(), 3)
test.eq(g2d:ySize(), 2)

test.eq(g2d:xOrigin(), 1)
test.eq(g2d:yOrigin(), 4)

test.eq(g2d:xWidth(), 0.9)
test.eq(g2d:yWidth(), 1)

test.eq(g2d:xCellWidth(), 0.3)
test.eq(g2d:yCellWidth(), 0.5)

test.eq(g2d:xBoundaryDepth(), 1)
test.eq(g2d:yBoundaryDepth(), 1)

test.eq(g2d:xUsePeriodic(), false)
test.eq(g2d:yUsePeriodic(), false)

test.aeq(g2d:Size(),          {3,2})
test.aeq(g2d:Origin(),        {1,4})
test.aeq(g2d:Width(),         {0.9,1.0})
test.aeq(g2d:CellWidth(),     {0.3,0.5})
test.aeq(g2d:BoundaryDepth(), {1,1})
test.aeq(g2d:UsePeriodic(),   {false,false})


test.eq(g3d:xSize(), 3)
test.eq(g3d:ySize(), 2)
test.eq(g3d:zSize(), 2)

test.eq(g3d:xOrigin(), 1)
test.eq(g3d:yOrigin(), 4)
test.eq(g3d:zOrigin(), 7)

test.eq(g3d:xWidth(), 0.9)
test.eq(g3d:yWidth(), 1)
test.eq(g3d:zWidth(), 0.8)

test.eq(g3d:xCellWidth(), 0.3)
test.eq(g3d:yCellWidth(), 0.5)
test.eq(g3d:zCellWidth(), 0.4)

test.eq(g3d:xBoundaryDepth(), 1)
test.eq(g3d:yBoundaryDepth(), 1)
test.eq(g3d:zBoundaryDepth(), 1)

test.eq(g3d:xUsePeriodic(), false)
test.eq(g3d:yUsePeriodic(), false)
test.eq(g3d:zUsePeriodic(), false)

test.aeq(g3d:Size(),          {3,2,2})
test.aeq(g3d:Origin(),        {1,4,7})
test.aeq(g3d:Width(),         {0.9,1.0,0.8})
test.aeq(g3d:CellWidth(),     {0.3,0.5,0.4})
test.aeq(g3d:BoundaryDepth(), {1,1,1})
test.aeq(g3d:UsePeriodic(),   {false,false,false})


--------------------------------------------------------------------

-- bad grid creation

test.fail_function(function()
  Grid.NewGrid2d {
    size = {},
    origin = {},
    width = {},
  }
end, 'NewGrid2d should be called with named parameters')

test.fail_function(function()
  Grid.NewGrid3d {
    size = {},
    origin = {},
    width = {},
  }
end, 'NewGrid3d should be called with named parameters')

test.fail_function(function()
  Grid.NewGrid2d {
    origin = {1,2},
    width = {5,5},
  }
end, 'NewGrid2d should be called with named parameters')

test.fail_function(function()
  Grid.NewGrid3d {
    origin = {1,2,3},
    width = {5,5,5},
  }
end, 'NewGrid3d should be called with named parameters')

test.fail_function(function()
  Grid.NewGrid2d {
    size  = {1,2},
    width = {5,5},
  }
end, 'NewGrid2d should be called with named parameters')

test.fail_function(function()
  Grid.NewGrid3d {
    size  = {1,2,3},
    width = {5,5,5},
  }
end, 'NewGrid3d should be called with named parameters')

test.fail_function(function()
  Grid.NewGrid2d {
    size   = {1,2},
    origin = {5,5},
  }
end, 'NewGrid2d should be called with named parameters')


test.fail_function(function()
  Grid.NewGrid3d {
    size   = {1,2,3},
    origin = {5,5,5},
  }
end, 'NewGrid3d should be called with named parameters')




--------------------------------------------------------------------

--[[
g2d.cells:NewField('idsq', L.vec2d):Load({{ {0,0}, {0,1} },
                                          { {1,0}, {1,1} },
                                          { {4,0}, {4,1} }})
--]]


--[[

-- Check bad arguments to create a relation with
test.fail_function(function()
  local relbad = L.NewRelation { name="relbad", mode="GRID" }
end, "Grids must specify 'dim' argument")
test.fail_function(function()
  local relbad = L.NewRelation { name="relbad", dims={5} }
end, "a table of 2 to 3 numbers")

-- test loading
rel1:NewField('v1',L.double):Load(function(i)    return i         end)
rel2:NewField('v2',L.vec2d):Load(function(x,y)   return {2*x,y}   end)
rel3:NewField('v3',L.vec3d):Load(function(x,y,z) return {3*x,y,z} end)

-- test printing
rel1.v1:Print()
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


-- try to group a 2d grid; we know this will fail
rel2:NewField('r1', rel1):Load(0)
test.fail_function(function()
  rel2:GroupBy('r1')
end, "Cannot group a relation unless it's a PLAIN relation")

-- try to group the 1d relation by the 2d one
rel1:NewField('r2', rel2):Load(function(i)
  return { i%2, math.floor(i/2) }
end)
rel1:GroupBy('r2')

-- and test that we can use the grouping in a function
local ebb group_k ( r2 : rel2 )
  for r1 in L.Where(rel1.r2, r2) do
    L.assert(2*L.double(L.int(r1.v1) % 2) == r2.v2[0])
    L.assert(  L.double(L.int(r1.v1) / 2) == r2.v2[1])
  end
end
rel2:foreach(group_k)


-- test some simple affine relationships
rel2:NewField('a2',L.double):Load(function(x,y)
  return 5*x + 3*y
end)
rel3:NewField('a3',L.double):Load(function(x,y,z)
  return 3*x + 5*y
end)
local ebb affinetest3to2 ( r3 : rel3 )
  var r2 = L.Affine(rel2, {{0,1,0,0},
                           {1,0,0,0}}, r3)
  L.assert(r3.a3 == r2.a2)
end
rel3:foreach(affinetest3to2)
local ebb affinetest2to3 ( r2 : rel2 )
  var r3 = L.Affine(rel3, {{0,1,0},
                           {1,0,0},
                           {0,0,0}}, r2)
  L.assert(r3.a3 == r2.a2)
end
rel2:foreach(affinetest2to3)
-- this one shouldn't match almost at all
local ebb scramble2to3 ( r2 : rel2 )
  var r3 = L.Affine(rel3, {{1,0,0},
                           {0,0,0},
                           {0,1,0}}, r2)
  L.assert(r3.a3 == 0 or r3.a3 ~= r2.a2)
end
rel2:foreach(scramble2to3)


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

--]]


