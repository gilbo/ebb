
import "compiler.liszt"
require "tests/test"


local cells = L.NewRelation(10, 'cells')
local particles = L.NewRelation(0, 'particles')

cells:NewField('temperature', L.double):Load(0.0)
particles:NewField('cell', cells)
particles:NewField('pos', L.vec3d)


-----------------------------------
-- Type Checking / Phase Checking
-----------------------------------

-- TODO: this could be relaxed
-- block insertion into a relation we're mapping over
test.fail_function(function()
  -- try to copy each particle
  liszt kernel( p : particles )
    insert { cell = L.UNSAFE_ROW( L.addr(0), cells ), pos = {0.1,0.1,0.1} } into particles
  end
end, "Cannot insert into relation particles while mapping over it")

-- TODO: this could be relaxed
-- block insertion into a relation if we're accessing that relation's fields
test.fail_function(function()
  liszt kernel( p : particles )
    insert { temperature = 0.1 } into cells
    p.cell.temperature += 0.25
  end
end, "Cannot insert into relation cells because%s*it\'s referred to by a field: particles.cell")
--"Cannot access field cells%.temperature while inserting.* into relation cells")


-- not specifying the insert of all fields should fail
test.fail_function(function()
  liszt kernel( c : cells )
    insert { cell = c } into particles
  end
end, "inserted record type does not match relation")


-- specifying non-existant fields should fail
test.fail_function(function()
  liszt kernel( c : cells )
    var pos = { L.double(L.id(c)), 0, 0 }
    insert { cell = c, pos = pos, diameter = 0.1 } into particles
  end
end, "cannot insert a value into field particles.diameter because it is undefined")


local grouped_rel = L.NewRelation(5,'grouped_rel')
grouped_rel:NewField('cell', cells):Load(function(i)
  return math.floor(i/2) end)
grouped_rel:GroupBy('cell')

-- A relation which is grouped can't be inserted into
test.fail_function(function()
  liszt kernel( c : cells )
    insert { cell = c } into grouped_rel
  end
end, 'Cannot insert into relation grouped_rel because it\'s grouped')

-- Inserting into something that isn't a relation
local sum = L.NewGlobal(L.addr, 0)
test.fail_function(function()
  liszt kernel( c : cells )
    insert { cell = c } into sum
  end
end, 'Expected a relation to insert into')


-- CANNOT insert twice into the same relation (do with a branch)
test.fail_function(function()
  liszt kernel( c : cells )
    var pos = { L.double(L.id(c)), 0, 0 }
    if L.id(c)%2 == 0 then
      insert { cell = c, pos = pos } into particles
    else
      insert { cell = c, pos = {0.1,0.1,0.1} } into particles
    end
  end
end, 'Cannot insert into relation particles twice')

-- Would like to have EXAMPLE of
-- Coercion of Record type values (unsuccessful)

-----------------------------------
-- Observable Effects
-----------------------------------

-- add one particle per cell
test.eq(particles:Size(), 0)

-- seed a particle in every cell
local seed_particles = liszt kernel( c : cells )
  var pos = { L.double(L.id(c)), 0, 0 }
  insert { cell = c, pos = pos } into particles
end

local post_insert_trivial = liszt kernel( p : particles )
  L.assert(p.pos[0] >= 0)
end

seed_particles(cells)

test.eq(particles:Size(), 10)

-- trivial kernel should not blow up
post_insert_trivial(particles)


