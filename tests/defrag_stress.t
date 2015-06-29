--DISABLE-ON-LEGION

import "compiler.liszt"
require "tests/test"

local Ncell = 1000


local cells = L.NewRelation { size = Ncell, name = 'cells' }
local particles = L.NewRelation {
  size = 0,
  mode = 'ELASTIC',
  name = 'particles'
}

particles:NewField('cell', cells):Load(0)


-----------------------------------



local seed_particles = liszt( c : cells )
  insert { cell = c } into particles
end

local liszt delete_even ( p : particles )
  if L.id(p.cell) % 2 == 0 then
    delete p
  end
end
local liszt delete_odd ( p : particles )
  if L.id(p.cell) % 2 == 1 then
    delete p
  end
end
local liszt delete_third ( p : particles )
  if L.id(p.cell) % 3 == 0 then
    delete p
  end
end

INSERTS_TIME = 0
DELETE_TIME = 0
-- cycle a bunch of particles in and then out
for k=1,1000 do
  local ADD_LOOPS = 10
  local startwatch = terralib.currenttimeinseconds()
  for nadd=1,ADD_LOOPS do
    cells:foreach(seed_particles)
  end
  local midwatch = terralib.currenttimeinseconds()
  INSERTS_TIME = INSERTS_TIME + (midwatch - startwatch)

  test.eq(Ncell*ADD_LOOPS, particles:Size())
  particles:foreach(delete_third)
  particles:foreach(delete_even)
  particles:foreach(delete_odd)
  test.eq(0, particles:Size())
  local endwatch = terralib.currenttimeinseconds()
  DELETE_TIME = DELETE_TIME + (endwatch - midwatch)
end

--print('totaldefrag', TOTAL_DEFRAG_TIME)
--print('totalinserts', INSERTS_TIME)
--print('totaldeletes', DELETE_TIME)
