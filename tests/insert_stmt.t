--DISABLE-ON-LEGION

import "compiler.liszt"
require "tests/test"


local cells = L.NewRelation { size = 10, name = 'cells' }
local particles = L.NewRelation {
  size = 0,
  mode = 'ELASTIC',
  name = 'particles'
}

cells:NewField('temperature', L.float):Load(0.0)
particles:NewField('cell', cells)
particles:NewField('pos', L.vec3d)


-----------------------------------
-- Type Checking / Phase Checking
-----------------------------------

-- TODO: this could be relaxed
-- block insertion into a relation we're mapping over
test.fail_function(function()
  -- try to copy each particle
  local liszt t( p : particles )
    insert { cell = L.UNSAFE_ROW( L.uint64(0), cells ), pos = {0.1,0.1,0.1} } into particles
  end
  particles:foreach(t)
end, "Cannot insert into relation particles while mapping over it")

-- TODO: this could be relaxed
-- block insertion into a relation if we're accessing that relation's fields
test.fail_function(function()
  local liszt t( p : particles )
    insert { temperature = L.float(0.1) } into cells
    p.cell.temperature += L.float(0.25)
  end
  particles:foreach(t)
end, "Cannot insert into relation cells because it\'s not ELASTIC")
--"Cannot access field cells%.temperature while inserting.* into relation cells")


-- not specifying the insert of all fields should fail
test.fail_function(function()
  local liszt t( c : cells )
    insert { cell = c } into particles
  end
  cells:foreach(t)
end, "inserted record type does not match relation")


-- specifying non-existant fields should fail
test.fail_function(function()
  local liszt t( c : cells )
    var pos = { L.double(L.id(c)), 0, 0 }
    insert { cell = c, pos = pos, diameter = 0.1 } into particles
  end
  cells:foreach(t)
end, "cannot insert a value into field particles.diameter because it is undefined")


local grouped_rel = L.NewRelation  { size = 5, name = 'grouped_rel' }
grouped_rel:NewField('cell', cells):Load(function(i)
  return math.floor(i/2) end)
grouped_rel:GroupBy('cell')

-- A relation which is grouped can't be inserted into
test.fail_function(function()
  local liszt t( c : cells )
    insert { cell = c } into grouped_rel
  end
  cells:foreach(t)
end, 'Cannot insert into relation grouped_rel because it\'s not ELASTIC')

-- Inserting into something that isn't a relation
local sum = L.Global(L.uint64, 0)
test.fail_function(function()
  local liszt t( c : cells )
    insert { cell = c } into sum
  end
  cells:foreach(t)
end, 'Expected a relation to insert into')


-- CANNOT insert twice into the same relation (do with a branch)
test.fail_function(function()
  local liszt t( c : cells )
    var pos = { L.double(L.id(c)), 0, 0 }
    if L.id(c)%2 == 0 then
      insert { cell = c, pos = pos } into particles
    else
      insert { cell = c, pos = {0.1,0.1,0.1} } into particles
    end
  end
  cells:foreach(t)
end, 'Cannot insert into relation particles twice')

-- Would like to have EXAMPLE of
-- Coercion of Record type values (unsuccessful)

-----------------------------------
-- Observable Effects
-----------------------------------

-- add one particle per cell
test.eq(particles:Size(), 0)

-- seed a particle in every cell
local seed_particles = liszt( c : cells )
  var pos = { L.double(L.id(c)), 0, 0 }
  insert { cell = c, pos = pos } into particles
end

local post_insert_trivial = liszt( p : particles )
  L.assert(p.pos[0] >= 0)
end

cells:foreach(seed_particles)

test.eq(particles:Size(), 10)

-- trivial function should not blow up
particles:foreach(post_insert_trivial)


