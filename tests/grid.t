import "compiler.liszt"
require "tests/test"

--error('NEED TO WRITE GRID TESTS; DO WHILE REWRITING GRID DOMAIN')

local rel1 = L.NewRelation { name="rel1", size = 5 }
local rel2 = L.NewRelation { name="rel2", dims = {2,3} }
local rel3 = L.NewRelation { name="rel3", dims = {3,2,2} }

test.eq(rel1:isGrid(), false)
test.eq(rel2:isGrid(), true)
test.eq(rel3:isGrid(), true)
test.eq(rel1:nDims(), 1)
test.eq(rel2:nDims(), 2)
test.eq(rel3:nDims(), 3)
test.aeq(rel1:Dims(), {5})
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
rel1:NewField('v1',L.double):Load(function(i)    return i         end)
rel2:NewField('v2',L.vec2d):Load(function(x,y)   return {2*x,y}   end)
rel3:NewField('v3',L.vec3d):Load(function(x,y,z) return {3*x,y,z} end)

-- test printing
rel1.v1:print()
rel2.v2:print()
rel3.v3:print()

-- test loading from a list
local tbl2 = {{1,2},{3,4},{5,6}}
local tbl3 = {{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}}
rel2:NewField('f2',L.double):Load(tbl2)
rel3:NewField('f3',L.double):Load(tbl3)
-- test that dumping preserves list/structure
test.rec_aeq(rel2.f2:DumpToList(),tbl2)
test.rec_aeq(rel3.f3:DumpToList(),tbl3)

-- test indexing consistency
rel3:NewField('f3func',L.double):Load(function(x,y,z)
  return 6*z + 3*y + x + 1
end)
local liszt f3consistency( r : rel3 )
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
local liszt group_k ( r2 : rel2 )
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
local liszt affinetest3to2 ( r3 : rel3 )
  var r2 = L.Affine(rel2, {{0,1,0,0},
                           {1,0,0,0}}, r3)
  L.assert(r3.a3 == r2.a2)
end
rel3:foreach(affinetest3to2)
local liszt affinetest2to3 ( r2 : rel2 )
  var r3 = L.Affine(rel3, {{0,1,0},
                           {1,0,0},
                           {0,0,0}}, r2)
  L.assert(r3.a3 == r2.a2)
end
rel2:foreach(affinetest2to3)
-- this one shouldn't match almost at all
local liszt scramble2to3 ( r2 : rel2 )
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
local liszt test_wrap ( r : prel2 )
  var off = L.Affine(prel2, {{1,0,1},
                             {0,1,1}}, r)
  L.assert( (r.cid[0]+1) % 3 == off.cid[0] and
            (r.cid[1]+1) % 4 == off.cid[1] )
end
prel2:foreach(test_wrap)


