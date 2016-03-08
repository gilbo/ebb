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

local M     = {
  vertices  = L.NewRelation { name="vertices",  size=8  },
  edges     = L.NewRelation { name="edges",     size=12 },
}
M.edges:NewField('head', M.vertices):Load({
  1, 2, 3, 0,   5, 6, 7, 4,   5, 0, 6, 7
})
M.edges:NewField('tail', M.vertices):Load({
  0, 1, 2, 3,   4, 5, 6, 7,   1, 4, 2, 3
})
M.vertices:NewField('position', L.vec3d):Load({
  {0, 0, 0},
  {1, 0, 0},
  {1, 1, 0},
  {0, 1, 0},
  {0, 0, 1},
  {1, 0, 1},
  {1, 1, 1},
  {0, 1, 1},
})

local init_to_zero = terra (mem : &float, i : int) mem[0] = 0 end
local function init_temp (i)
  if i == 0 then
    return 1000
  else
    return 0
  end
end

M.vertices:NewField('flux',        L.float):Load(0)
M.vertices:NewField('jacobistep',  L.float):Load(0)
M.vertices:NewField('temperature', L.float):Load(init_temp)

local ebb compute_step (e : M.edges)
  var v1   = e.head
  var v2   = e.tail
  var dp   = L.vec3f(v1.position - v2.position)
  var dt   = v1.temperature - v2.temperature
  var step = 1.0f / L.length(dp)

  v1.flux += -dt * step
  v2.flux +=  dt * step

  v1.jacobistep += step
  v2.jacobistep += step
end

local ebb propagate_temp (p : M.vertices)
  p.temperature += L.float(.01) * p.flux / p.jacobistep
end

local ebb clear (p : M.vertices)
  p.flux       = 0
  p.jacobistep = 0
end

for i = 1, 1000 do
  M.edges:foreach(compute_step)
  M.vertices:foreach(propagate_temp)
  M.vertices:foreach(clear)
end

M.vertices.temperature:Print()
