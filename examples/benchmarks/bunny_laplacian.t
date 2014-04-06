import "compiler.liszt" -- Every Liszt File should start with this command

-- This line includes the trimesh.t file.
-- As a result, the table 'Trimesh' defined in that file is bound to
-- the variable Trimesh declared right here.
local Trimesh = terralib.require 'examples.benchmarks.trimesh'

-- PN (Pathname) is a convenience library for working with paths
local PN = terralib.require 'compiler.pathname'

-- include C math functions
local cmath = terralib.includecstring '#include <math.h>'


------------------------------------------------------------------------------

-- here's the path object for our .OFF file we want to read in.
local tri_mesh_filename = PN.scriptdir() .. 'bunny.off'

-- Here we create a new triangle mesh by loading in an OFF file
-- We can look in examples/tutorials/trimesh.t to find the implementation
-- of this function.
local bunny = Trimesh.LoadFromOFF(tri_mesh_filename)

------------------------------------------------------------------------------


bunny.triangles:NewField('area_normal', L.vec3d):Load({0,0,0})
local compute_area_normals = liszt kernel( t : bunny.triangles )
  var p1 = t.v1.pos
  var p2 = t.v2.pos
  var p3 = t.v3.pos

  var n = L.cross(p2-p1, p3-p1)
  t.area_normal = n / 2
end

bunny.edges:NewField('laplacian', L.double):Load(0)
bunny.vertices:NewField('laplacian_diag', L.double):Load(0)
local zero_laplacian_edge = liszt kernel(e : bunny.edges)
  e.laplacian = 0 end
local zero_laplacian_vert = liszt kernel(v : bunny.vertices)
  v.laplacian_diag = 0 end
local build_laplacian = liszt kernel(t : bunny.triangles)
  var area : L.double = L.length(t.area_normal) / 2
  if area < 0.00001 then area = 0.00001 end

  -- this should be the cotan laplacian
  var c1 = L.dot(t.v2.pos - t.v1.pos, t.v3.pos - t.v1.pos) / area
  var c2 = L.dot(t.v1.pos - t.v2.pos, t.v3.pos - t.v2.pos) / area
  var c3 = L.dot(t.v1.pos - t.v3.pos, t.v2.pos - t.v3.pos) / area

  t.v1.laplacian_diag += c2+c3
  t.v2.laplacian_diag += c1+c3
  t.v3.laplacian_diag += c1+c2

  t.e12.laplacian += c3
  t.e21.laplacian += c3
  t.e13.laplacian += c2
  t.e31.laplacian += c2
  t.e23.laplacian += c1
  t.e32.laplacian += c1
end

local function compute_laplacian(mesh)
  zero_laplacian_edge(mesh.edges)
  zero_laplacian_vert(mesh.vertices)
  compute_area_normals(mesh.triangles)
  build_laplacian(mesh.triangles)
end


------------------------------------------------------------------------------

-- define globals
local timestep = L.NewGlobal(L.double, 0.1)

bunny.vertices:NewField('d_pos', L.vec3d):Load({0,0,0})
local zero_d_pos = liszt kernel( v : bunny.vertices )
  v.d_pos = {0,0,0}
end

local compute_diffusion = liszt kernel ( v : bunny.vertices )
  var acc : L.vec3d = {0,0,0}
  for e in v.edges do
    acc += e.laplacian * (e.head.pos - v.pos)
  end
  v.d_pos = timestep * acc / v.laplacian_diag
end

local apply_diffusion = liszt kernel ( v : bunny.vertices )
  v.pos += v.d_pos
end


------------------------------------------------------------------------------

-- draw stuff

local sqrt3 = math.sqrt(3)

-- EXTRA: (optional.  It demonstrates the use of VDB, a visual debugger)
local vdb = terralib.require('compiler.vdb')
local debug_tri_draw = liszt kernel ( t : bunny.triangles )
  -- Spoof a really simple directional light
  -- with a cos diffuse term determining the triangle gray scale
  var d = L.dot({1/sqrt3, 1/sqrt3, 1/sqrt3},
                t.area_normal/L.length(t.area_normal))
  if d > 1.0 then d = 1.0 end
  if d < -1.0 then d = -1.0 end
  var val = d * 0.5 + 0.5
  var col : L.vec3d = {val,val,val}
  vdb.color(col)
  vdb.triangle(t.v1.pos, t.v2.pos, t.v3.pos)
end
-- END EXTRA VDB CODE


------------------------------------------------------------------------------


-- Execute 400 iterations of the diffusion

for i = 1,400 do
  compute_laplacian(bunny)
  zero_d_pos(bunny.vertices)

  compute_diffusion(bunny.vertices)
  apply_diffusion(bunny.vertices)

  -- EXTRA: VDB
  vdb.vbegin()
    vdb.frame() -- this call clears the canvas for a new frame
    debug_tri_draw(bunny.triangles)
  vdb.vend()
  -- END EXTRA
end


------------------------------------------------------------------------------

-- For this file, we've omitted writing the output anywhere.
-- See the long form of bunny_diffusion.t for more details
-- on data output options

