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
  local ebb test( c : cells )
    delete c
  end
  cells:foreach(test)
end, "Cannot delete from relation cells because it\'s not ELASTIC")

-- cannot delete indirectly
test.fail_function(function()
  local ebb test( p : particles )
    delete p.cell
  end
  particles:foreach(test)
end, "Only centered keys may be deleted")

-- CANNOT HAVE 2 DELETE STATEMENTS in the same function
test.fail_function(function()
  local ebb test( p : particles )
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

local ebb delete_even ( p : particles )
  if L.id(p) % 2 == 0 then
    delete p
  else
    p.pos[0] = 3
  end
end

local ebb post_delete_trivial( p : particles )
  L.assert(p.pos[0] == 3)
end

particles:foreach(delete_even)

test.eq(particles:Size(), 5)

-- trivial function should not blow up
particles:foreach(post_delete_trivial)

