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
------------------------------------------------------------------------------

import "ebb"
local L = require "ebblib"

local ioVeg = require 'ebb.domains.ioVeg'
local PN    = require 'ebb.lib.pathname'
local dragon = ioVeg.LoadTetmesh(PN.scriptdir() .. 'dragon.veg')


------------------------------------------------------------------------------

-- Load and prepare the render mesh
local ioOff           = require 'ebb.domains.ioOff'
local rendermesh_file = PN.scriptdir() .. 'hires_render.off'
local coupling_file   = PN.scriptdir() .. 'hires_coupling.data'
local renderrate      = 300
local rendermesh      = ioOff.LoadTrimesh(rendermesh_file)

local tets = {}
local interps = {}
do
  local infile = io.open(tostring(coupling_file), 'r')
  local N = infile:read('*number')
  for i=1,N do
    tets[i] = infile:read('*number')
    interps[i] = { infile:read('*number'), infile:read('*number'),
                   infile:read('*number'), infile:read('*number') }
  end
  infile:close()
end
print('read file')

rendermesh.vertices:NewField('tet', dragon.tetrahedra):Load(tets)
rendermesh.vertices:NewField('interp', L.vec4d):Load(interps)

local ebb update_rendermesh( rv : rendermesh.vertices )
  var pos : L.vec3d = { 0.0, 0.0, 0.0 }
  for k=0,4 do
    pos += rv.interp[k] * rv.tet.v[k].pos
  end
  rv.pos = pos
end


------------------------------------------------------------------------------

-- Define simulation constants
local dt       = L.Constant(L.double, 0.0002)
local K        = L.Constant(L.double, 2000.0)
local damp     = L.Constant(L.double, 0.5)
local gravity  = L.Constant(L.vec3d, {0,-0.98,0})

-- Define simulation fields
dragon.vertices:NewField('vel', L.vec3d)        :Load({0,0,0})
dragon.vertices:NewField('acc', L.vec3d)        :Load({0,0,0})
dragon.edges:NewField('rest_len', L.double)     :Load(0)

-- We'll initialize the mass and resting length using parallel functions
local ebb init_rest_len ( e : dragon.edges )
  var diff = e.head.pos - e.tail.pos
  e.rest_len = L.length(diff)
end
dragon.edges:foreach(init_rest_len)

-- Also, let's translate the dragon up above the y=0 plane
-- so we can drop it onto the ground
local ebb lift_dragon ( v : dragon.vertices )
  v.pos += {0,2.3,0}
end
dragon.vertices:foreach(lift_dragon)

------------------------------------------------------------------------------

-- Now let's define functions to perform the basic updates in our simualtion
-- We're going to use a simple forward Euler integration scheme here.

local ebb compute_acceleration ( v : dragon.vertices )
  var force = { 0.0, 0.0, 0.0 }

  -- Pseudo-Physical Spring Force
  var mass = 0.0
  for e in v.edges do
    var diff  = e.head.pos - v.pos
    var scale = (e.rest_len / L.length(diff)) - 1.0
    mass     += e.rest_len
    force    -= K * scale * diff
  end
  force = force / mass

  -- Ground Force
  if v.pos[1] < 0.0 then
    force += { 0, - K * v.pos[1], 0 }
  end

  v.acc = force + gravity
end

local ebb update_vel_pos ( v : dragon.vertices )
  v.pos += dt * v.vel + 0.5 * dt * dt * v.acc
  v.vel = (1 - damp * dt) * v.vel + (dt * v.acc)
end

------------------------------------------------------------------------------

-- Again, we won't discuss VDB here, but I'm going to include
-- a little bit of code to visualize the result

-- START EXTRA VDB CODE
local sqrt3 = math.sqrt(3)
local vdb   = require('ebb.lib.vdb')
local ebb compute_normal ( t )--t : dragon.triangles )
  var p0  = t.v[0].pos
  var p1  = t.v[1].pos
  var p2  = t.v[2].pos
  var n   = L.cross(p1-p0, p2-p0)
  var len = L.length(n)
  if len < 1.0e-6 then len = 1.0e6 else len = 1.0/len end -- invert len
  return n * len
end
local ebb debug_tri_draw ( t )--t : dragon.triangles )
  -- Spoof a really simple directional light with a cos diffuse term
  var d = - L.dot({1/sqrt3, -1/sqrt3, 1/sqrt3}, compute_normal(t))
  if d > 1.0 then d = 1.0 end
  if d < -1.0 then d = -1.0 end
  var val = d * 0.5 + 0.5
  var col : L.vec3d = {val,val,val}
  vdb.color(col)
  vdb.triangle(t.v[0].pos, t.v[1].pos, t.v[2].pos)
end
-- END EXTRA VDB CODE


------------------------------------------------------------------------------

-- Now, let's try running this simulation for a while

local do_vdb_draw = true
for i=1,40000 do
  dragon.vertices:foreach(compute_acceleration)
  dragon.vertices:foreach(update_vel_pos)

  -- EXTRA: VDB (For visualization)
  if i%renderrate == 0 and do_vdb_draw then
    vdb.vbegin()
      vdb.frame() -- this call clears the canvas for a new frame
      --dragon.triangles:foreach(debug_tri_draw)
      rendermesh.vertices:foreach(update_rendermesh)
      rendermesh.triangles:foreach(debug_tri_draw)
    vdb.vend()
  end
  if i%1000 == 0 then print('iter', i) end
  -- END EXTRA
end








