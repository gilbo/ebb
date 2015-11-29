import "ebb"
local L = require "ebblib" -- Every Ebb File should start with this command

-- This line includes the trimesh.t file.
-- As a result, the table 'Trimesh' defined in that file is bound to
-- the variable Trimesh declared right here.
local Trimesh = require 'devapps.tutorials.trimesh'

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

-- In trimesh.t we defined a way to compute vertex degree
-- (the number of triangles touching a vertex)
-- We can just invoke that computation here, which will install a
-- new field 'degree' on bunny.vertices
bunny:ComputeVertexDegree()

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

-- we define the basic computation functions here:

local ebb compute_diffusion ( tri : bunny.triangles )
  var e12 : L.double = 1.0
  var e23 : L.double = 1.0
  var e13 : L.double = 1.0

  var t1 = tri.v1.temperature
  var t2 = tri.v2.temperature
  var t3 = tri.v3.temperature

  var dt_1 = (timestep * conduction / tri.v1.degree) *
                  (e12 * (t2 - t1) + e13 * (t3 - t1))
  var dt_2 = (timestep * conduction / tri.v2.degree) *
                  (e12 * (t1 - t2) + e23 * (t3 - t2))
  var dt_3 = (timestep * conduction / tri.v3.degree) *
                  (e13 * (t1 - t3) + e23 * (t2 - t3))

  tri.v1.d_temperature += dt_1
  tri.v2.d_temperature += dt_2
  tri.v3.d_temperature += dt_3
end

local ebb apply_diffusion ( v : bunny.vertices )
  var d_temp = v.d_temperature
  v.temperature += d_temp

  avg_temp_change += cmath.fabs(d_temp)
end

local ebb clear_temporary ( v : bunny.vertices )
  v.d_temperature = 0.0
end


-- EXTRA: (optional.  It demonstrates the use of VDB, a visual debugger)
local vdb  = require('ebb.lib.vdb')
local cold = L.Constant(L.vec3f,{0.5,0.5,0.5})
local hot  = L.Constant(L.vec3f,{1.0,0.0,0.0})
local ebb debug_tri_draw ( t : bunny.triangles )
  -- color a triangle with the average temperature of its vertices
  var avg_temp =
    (t.v1.temperature + t.v2.temperature + t.v3.temperature) / 3.0

  -- compute a display value in the range 0.0 to 1.0 from the temperature
  var scale = L.float(cmath.log(1.0 + avg_temp))
  if scale > 1.0 then scale = 1.0 end

  -- interpolate the hot and cold colors
  vdb.color((1.0-scale)*cold + scale*hot)
  vdb.triangle(t.v1.pos, t.v2.pos, t.v3.pos)
end
-- END EXTRA VDB CODE


------------------------------------------------------------------------------


-- Execute 300 iterations of the diffusion

for i = 1,300 do
  bunny.triangles:map(compute_diffusion)

  avg_temp_change:set(0.0)
  bunny.vertices:map(apply_diffusion)
  avg_temp_change:set( avg_temp_change:get() / bunny:nVerts())

  bunny.vertices:map(clear_temporary)

  -- EXTRA: VDB
  vdb.vbegin()
    vdb.frame() -- this call clears the canvas for a new frame
    bunny.triangles:map(debug_tri_draw)
  vdb.vend()
  -- END EXTRA
end


------------------------------------------------------------------------------

-- For this file, we've omitted writing the output anywhere.
-- See the long form of bunny_diffusion.t for more details
-- on data output options

