--DISABLE-ON-LEGION

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

local Ncell = 1000


local cells = L.NewRelation { size = Ncell, name = 'cells' }
local particles = L.NewRelation {
  size = 0,
  mode = 'ELASTIC',
  name = 'particles'
}

particles:NewField('cell', cells):Load(0)


-----------------------------------



local seed_particles = ebb( c : cells )
  insert { cell = c } into particles
end

local ebb delete_even ( p : particles )
  if L.id(p.cell) % 2 == 0 then
    delete p
  end
end
local ebb delete_odd ( p : particles )
  if L.id(p.cell) % 2 == 1 then
    delete p
  end
end
local ebb delete_third ( p : particles )
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
