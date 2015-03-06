import "compiler.liszt" -- Every Liszt File should start with this command

-- This line includes the trimesh.t file.
-- As a result, the table 'Trimesh' defined in that file is bound to
-- the variable Trimesh declared right here.
local Trimesh = L.require 'examples.benchmarks.trimesh'

-- PN (Pathname) is a convenience library for working with paths
local PN = L.require 'lib.pathname'

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

-- define globals
local timestep = L.Global(L.double, 0.45)
local avg_temp_change = L.Global(L.double, 0.0)

-- define constants
local conduction = 1.0

-- define fields
bunny.vertices:NewField('temperature', L.double)
bunny.vertices.temperature:Load(function(index)
  if index == 0 then return 3000.0 else return 0.0 end
end)

bunny.vertices:NewField('d_temperature', L.double)
bunny.vertices.d_temperature:Load(0.0)

------------------------------------------------------------------------------

-- we define the basic computation kernels here:

-- This could also be written as a kernel over the edges...
local compute_diffusion = liszt kernel ( v : bunny.vertices )
  for nv in v.neighbors do
    v.d_temperature += timestep * conduction *
      (nv.temperature - v.temperature)
  end
end

local apply_diffusion = liszt kernel ( v : bunny.vertices )
  var d_temp = v.d_temperature / v.degree
  v.temperature += d_temp

  avg_temp_change += cmath.fabs(d_temp)
end

local clear_temporary = liszt kernel ( v : bunny.vertices )
  v.d_temperature = 0.0
end


-- EXTRA: (optional.  It demonstrates the use of VDB, a visual debugger)
local vdb  = L.require('lib.vdb')
local cold = L.Constant(L.vec3f,{0.5,0.5,0.5})
local hot  = L.Constant(L.vec3f,{1.0,0.0,0.0})
local debug_tri_draw = liszt kernel ( t : bunny.triangles )
  -- color a triangle with the average temperature of its vertices
  var avg_temp =
    (t.v[0].temperature + t.v[1].temperature + t.v[2].temperature) / 3.0

  -- compute a display value in the range 0.0 to 1.0 from the temperature
  var scale = L.float(cmath.log(1.0 + avg_temp))
  if scale > 1.0 then scale = 1.0 end

  -- interpolate the hot and cold colors
  vdb.color((1.0-scale)*cold + scale*hot)
  vdb.triangle(t.v[0].pos, t.v[1].pos, t.v[2].pos)
end
-- END EXTRA VDB CODE


------------------------------------------------------------------------------


-- Execute 300 iterations of the diffusion

for i = 1,300 do
  compute_diffusion(bunny.vertices)

  avg_temp_change:set(0.0)
  apply_diffusion(bunny.vertices)
  avg_temp_change:set( avg_temp_change:get() / bunny:nVerts())

  clear_temporary(bunny.vertices)

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

