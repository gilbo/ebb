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

-- Load the Bunny Triangle Mesh
local ioOff = require 'ebb.domains.ioOff'
local PN    = require 'ebb.lib.pathname'
local mesh  = ioOff.LoadTrimesh( PN.scriptdir() .. 'bunny.off' )

-- define globals and constants
local timestep   = L.Constant(L.double, 0.45)
local K          = L.Constant(L.double, 1.0)
local max_change = L.Global(L.double, 0.0)

-- define fields: temperature and change in temperature
mesh.vertices:NewField('t', L.double):Load(function(index)
  if index == 0 then return 3000.0 else return 0.0 end
end)
mesh.vertices:NewField('d_t', L.double):Load(0.0)

------------------------------------------------------------------------------

-- This could also be written as a function over the edges...
local ebb compute_diffusion ( v : mesh.vertices )
  var count = 0.0
  for nv in v.neighbors do
    v.d_t += timestep * K * (nv.t - v.t)
    count += 1.0
  end
  v.d_t = v.d_t / count
end

local ebb apply_update ( v : mesh.vertices )
  v.t += v.d_t
  max_change max= L.fabs(v.d_t)
  v.d_t = 0.0
end

------------------------------------------------------------------------------

-- EXTRA: (optional.  It demonstrates the use of VDB, a visual debugger)
local vdb  = require('ebb.lib.vdb')
local ebb debug_tri_draw ( t : mesh.triangles )
  var avg_t = (t.v[0].t + t.v[1].t + t.v[2].t) / 3.0
  vdb.color({ 0.5 * avg_t + 0.5, 0.5-avg_t, 0.5-avg_t })
  vdb.triangle(t.v[0].pos, t.v[1].pos, t.v[2].pos)
end
-- END EXTRA VDB CODE

------------------------------------------------------------------------------

for i = 1,300 do
  if i % 30 == 0 then   max_change:set(0.0) end

  mesh.vertices:foreach(compute_diffusion)
  mesh.vertices:foreach(apply_update)

  if i % 30 == 0 then   print('iter #'..i, max_change:get()) end

  -- EXTRA: VDB
  vdb.vbegin()
    vdb.frame() -- this call clears the canvas for a new frame
    --debug_tri_draw(mesh.triangles)
    mesh.triangles:foreach(debug_tri_draw)
  vdb.vend()
  -- END EXTRA
end
