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

------------------------------------------------------------------------------

mesh.triangles:NewField('area_normal', L.vec3d):Load({0,0,0})
local ebb compute_area_normals( t : mesh.triangles )
  var p0 = t.v[0].pos
  var p1 = t.v[1].pos
  var p2 = t.v[2].pos

  var n = L.cross(p1-p0, p2-p0)
  t.area_normal = n / 2
end

mesh.edges:NewField('laplacian', L.double):Load(0)
mesh.vertices:NewField('laplacian_diag', L.double):Load(0)
local ebb zero_laplacian_edge(e : mesh.edges)
  e.laplacian = 0 end
local ebb zero_laplacian_vert(v : mesh.vertices)
  v.laplacian_diag = 0 end
local ebb build_laplacian(t : mesh.triangles)
  var area : L.double = L.length(t.area_normal) / 2
  if area < 0.00001 then area = 0.00001 end

  -- this should be the cotan laplacian
  var c0 = L.dot(t.v[1].pos - t.v[0].pos, t.v[2].pos - t.v[0].pos) / area
  var c1 = L.dot(t.v[2].pos - t.v[1].pos, t.v[0].pos - t.v[1].pos) / area
  var c2 = L.dot(t.v[0].pos - t.v[2].pos, t.v[1].pos - t.v[2].pos) / area

  t.v[0].laplacian_diag += c1+c2
  t.v[1].laplacian_diag += c0+c2
  t.v[2].laplacian_diag += c0+c1

  t.e12.laplacian += c2
  t.e21.laplacian += c2
  t.e13.laplacian += c1
  t.e31.laplacian += c1
  t.e23.laplacian += c0
  t.e32.laplacian += c0
end

local function compute_laplacian(mesh)
  mesh.edges:foreach(zero_laplacian_edge)
  mesh.vertices:foreach(zero_laplacian_vert)
  mesh.triangles:foreach(compute_area_normals)
  mesh.triangles:foreach(build_laplacian)
end


------------------------------------------------------------------------------

-- define globals
local timestep = L.Global(L.double, 0.1)

mesh.vertices:NewField('d_pos', L.vec3d):Load({0,0,0})
local ebb zero_d_pos ( v : mesh.vertices )
  v.d_pos = {0,0,0}
end

local ebb compute_diffusion ( v : mesh.vertices )
  var acc : L.vec3d = {0,0,0}
  for e in v.edges do
    acc += e.laplacian * (e.head.pos - v.pos)
  end

  v.d_pos = timestep * acc / v.laplacian_diag
end

local ebb apply_diffusion ( v : mesh.vertices )
  v.pos += v.d_pos
end


------------------------------------------------------------------------------

-- draw stuff

local sqrt3 = math.sqrt(3)

-- EXTRA: (optional.  It demonstrates the use of VDB, a visual debugger)
local vdb = require('ebb.lib.vdb')
local ebb debug_tri_draw ( t : mesh.triangles )
  -- Spoof a really simple directional light
  -- with a cos diffuse term determining the triangle gray scale
  var d = L.dot({1/sqrt3, 1/sqrt3, 1/sqrt3},
                t.area_normal/L.length(t.area_normal))
  if d > 1.0 then d = 1.0 end
  if d < -1.0 then d = -1.0 end
  var val = d * 0.5 + 0.5
  var col : L.vec3d = {val,val,val}
  vdb.color(col)
  vdb.triangle(t.v[0].pos, t.v[1].pos, t.v[2].pos)
end
-- END EXTRA VDB CODE


------------------------------------------------------------------------------


-- Execute 400 iterations of the diffusion

for i = 1,400 do
  compute_laplacian(bunny)
  mesh.vertices:foreach(zero_d_pos)

  mesh.vertices:foreach(compute_diffusion)
  mesh.vertices:foreach(apply_diffusion)

  -- EXTRA: VDB
  vdb.vbegin()
    vdb.frame() -- this call clears the canvas for a new frame
    mesh.triangles:foreach(debug_tri_draw)
  vdb.vend()
  -- END EXTRA
end



------------------------------------------------------------------------------

-- For this file, we've omitted writing the output anywhere.
-- See the long form of bunny_diffusion.t for more details
-- on data output options

