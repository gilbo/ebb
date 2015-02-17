import "compiler.liszt"
require "tests/test"

error('NEED TO WRITE GRID TESTS; DO WHILE REWRITING GRID DOMAIN')

local rel1 = L.NewRelation { name="rel1", size = 5 }
local rel2 = L.NewRelation { name="rel2", dim = {2,3} }
local rel3 = L.NewRelation { name="rel3", dim = {3,2,2} }

test.aeq(rel1:Dims(), {5})
test.aeq(rel2:Dims(), {2,3})
test.aeq(rel3:Dims(), {3,2,2})
-- TEST EQ nDIMS

-- Check bad arguments to create a relation with
test.fail_function(function()
  local relbad = L.NewRelation { name="relbad", mode="GRID" }
end, "Grids must specify 'dim' argument")
-- CHECK TRYING TO HAVE 1D DIM









-------------------------------------------------------------------

-------------------------------------------------------------------

-------------------------------------------------------------------

-------------------------------------------------------------------