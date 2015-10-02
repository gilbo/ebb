import "ebb" -- Every Ebb File should start with this command

-- This line includes the trimesh.t file.
-- As a result, the table 'Trimesh' defined in that file is bound to
-- the variable Trimesh declared right here.
local Trimesh = require 'devapps.benchmarks.trimesh'

-- PN (Pathname) is a convenience library for working with paths
local PN = require 'ebb.lib.pathname'

-- include C math functions
local cmath = terralib.includecstring '#include <math.h>'


------------------------------------------------------------------------------

-- here's the path object for our .OFF file we want to read in.
local tri_mesh_filename = PN.scriptdir() .. 'bunny.off'

-- Here we create a new triangle mesh by loading in an OFF file
-- We can look in devapps/tutorials/trimesh.t to find the implementation
-- of this function.
local bunny = Trimesh.LoadFromOFF(tri_mesh_filename)

------------------------------------------------------------------------------


bunny.triangles:NewField('area_normal', L.vec3d):Load({0,0,0})
local ebb compute_area_normals( t : bunny.triangles )
  var p0 = t.v[0].pos
  var p1 = t.v[1].pos
  var p2 = t.v[2].pos

  var n = L.cross(p1-p0, p2-p0)
  t.area_normal = n / 2
end

bunny.edges:NewField('laplacian', L.double):Load(0)
bunny.vertices:NewField('laplacian_diag', L.double):Load(0)
local ebb zero_laplacian_edge(e : bunny.edges)
  e.laplacian = 0 end
local ebb zero_laplacian_vert(v : bunny.vertices)
  v.laplacian_diag = 0 end
local ebb build_laplacian(t : bunny.triangles)
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

bunny.vertices:NewField('d_pos', L.vec3d):Load({0,0,0})
local ebb zero_d_pos ( v : bunny.vertices )
  v.d_pos = {0,0,0}
end

local ebb compute_diffusion ( v : bunny.vertices )
  var acc : L.vec3d = {0,0,0}
  for e in v.edges do
    acc += e.laplacian * (e.head.pos - v.pos)
  end

  v.d_pos = timestep * acc / v.laplacian_diag
end

local ebb apply_diffusion ( v : bunny.vertices )
  v.pos += v.d_pos
end


------------------------------------------------------------------------------

-- draw stuff

local sqrt3 = math.sqrt(3)

-- EXTRA: (optional.  It demonstrates the use of VDB, a visual debugger)
local vdb = require('ebb.lib.vdb')
local ebb debug_tri_draw ( t : bunny.triangles )
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
  bunny.vertices:foreach(zero_d_pos)

  bunny.vertices:foreach(compute_diffusion)
  bunny.vertices:foreach(apply_diffusion)

  -- EXTRA: VDB
  vdb.vbegin()
    vdb.frame() -- this call clears the canvas for a new frame
    bunny.triangles:foreach(debug_tri_draw)
  vdb.vend()
  -- END EXTRA
end



------------------------------------------------------------------------------

-- For this file, we've omitted writing the output anywhere.
-- See the long form of bunny_diffusion.t for more details
-- on data output options

