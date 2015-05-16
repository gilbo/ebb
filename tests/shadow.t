import "compiler.liszt"
require "tests/test"

local cells = L.NewRelation { size = 10, name = 'cells' }
cells:NewField('val', L.double):Load(5)


-- Directly shadowing variables like this shouldn't be
-- a problem but some tricky ordering details in how envirnoments
-- are managed in the compiler can cause errors

local center_shadow = liszt ( c : cells )
  var c = c
  L.assert(c.val == 5)
end
cells:foreach(center_shadow)

local center_other = liszt ( c : cells )
  var v = 25
  var v = 2
end
cells:foreach(center_other)