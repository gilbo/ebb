import "compiler.liszt"
require "tests/test"

--error('NEED TO WRITE GRID TESTS; DO WHILE REWRITING GRID DOMAIN')

local rel1 = L.NewRelation { name="rel1", size = 5 }
local rel2 = L.NewRelation { name="rel2", dim = {2,3} }
local rel3 = L.NewRelation { name="rel3", dim = {3,2,2} }

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
  local relbad = L.NewRelation { name="relbad", dim={5} }
end, "a table of 2 to 3 numbers")

-- try to group a 2d grid; we know this will fail
rel2:NewField('r1', rel1):Load(0)
test.fail_function(function()
  rel2:GroupBy('r1')
end, "Cannot group a relation unless it's a PLAIN relation")

-- try to group the 1d relation by the 2d one
rel1:NewField('r2', rel2):Load(function(i)
  return { i%3, math.floor(i/3) }
end)
--rel1:GroupBy('r2')

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






-------------------------------------------------------------------

-------------------------------------------------------------------

-------------------------------------------------------------------

-------------------------------------------------------------------