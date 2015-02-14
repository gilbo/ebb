
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
  liszt kernel test( c : cells )
    delete c
  end
end, "Cannot delete from relation cells because it\'s not ELASTIC")

-- cannot delete indirectly
test.fail_function(function()
  liszt kernel test( p : particles )
    delete p.cell
  end
end, "Only centered rows may be deleted")

-- CANNOT HAVE 2 DELETE STATEMENTS in the same kernel
test.fail_function(function()
  liszt kernel test( p : particles )
    if L.id(p) % 2 == 0 then
      delete p
    else
      delete p
    end
  end
end, "Temporary: can only have one delete statement per kernel")


-----------------------------------
-- Observable Effects
-----------------------------------

-- delete half the particles
test.eq(particles:Size(), 10)

local liszt kernel delete_even ( p : particles )
  if L.id(p) % 2 == 0 then
    delete p
  else
    p.pos[0] = 3
  end
end

local liszt kernel post_delete_trivial( p : particles )
  L.assert(p.pos[0] == 3)
end

delete_even(particles)

test.eq(particles:Size(), 5)

-- trivial kernel should not blow up
post_delete_trivial(particles)