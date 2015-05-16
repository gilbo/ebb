--DISABLE-ON-GPU  (TODO: Make this test pass on GPU)
--DISABLE-ON-LEGION

import "compiler.liszt"
require "tests/test"

local cells = L.NewRelation { size = 10, name = 'cells' }
local particles = L.NewRelation {
  size = 10,
  mode = 'ELASTIC',
  name = 'particles'
}

cells:NewField('temperature', L.double):Load(0.0)
particles:NewField('cell', cells):Load(function (id) return id end)
particles:NewField('pos', L.vec3d):Load({0, 1, 2})


-----------------------------------
-- Type Checking / Phase Checking
-----------------------------------

-- cannot delete from a relation currently being referenced by
-- another relation
test.fail_function(function()
  -- try to delete each cell
  local liszt test( c : cells )
    delete c
  end
  cells:foreach(test)
end, "Cannot delete from relation cells because it\'s not ELASTIC")

-- cannot delete indirectly
test.fail_function(function()
  local liszt test( p : particles )
    delete p.cell
  end
  particles:foreach(test)
end, "Only centered keys may be deleted")

-- CANNOT HAVE 2 DELETE STATEMENTS in the same function
test.fail_function(function()
  local liszt test( p : particles )
    if L.id(p) % 2 == 0 then
      delete p
    else
      delete p
    end
  end
  particles:foreach(test)
end, "Temporary: can only have one delete statement per function")


-----------------------------------
-- Observable Effects
-----------------------------------

-- delete half the particles
test.eq(particles:Size(), 10)

local liszt delete_even ( p : particles )
  if L.id(p) % 2 == 0 then
    delete p
  else
    p.pos[0] = 3
  end
end

local liszt post_delete_trivial( p : particles )
  L.assert(p.pos[0] == 3)
end

particles:foreach(delete_even)

test.eq(particles:Size(), 5)

-- trivial function should not blow up
particles:foreach(post_delete_trivial)