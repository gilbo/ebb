import "ebb" -- Every Ebb File should start with this command

-- This line includes a wrapper for the Trimesh domain library
--  that knows how to load OFF format files.
local ioOff = require 'ebb.domains.ioOff'

-- PN (Pathname) is a convenience library for working with paths
local PN = require 'ebb.lib.pathname'

------------------------------------------------------------------------------

-- here's the path object for our .OFF file we want to read in.
local tri_mesh_filename = PN.scriptdir() .. 'bunny.off'

-- Here we create a new triangle mesh by loading in an OFF file
-- We can look in devapps/tutorials/trimesh.t to find the implementation
-- of this function.
local bunny = ioOff.LoadTrimesh(tri_mesh_filename)

------------------------------------------------------------------------------

-- define a field to store the degree of each vertex
-- and load in the value 0 everywhere initially
bunny.vertices:NewField('degree', L.int):Load(0)

-- and define a function to compute the field values
local ebb compute_degree ( v : bunny.vertices )
  for e in v.edges do
    v.degree += 1
  end
end

-- then run that computation
bunny.vertices:foreach(compute_degree)

------------------------------------------------------------------------------

-- define globals
local timestep = L.Constant(L.double, 0.45)
local max_temp_change = L.Global(L.double, 0.0)

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

-- This could also be written as a function over the edges...
local ebb compute_diffusion ( v : bunny.vertices )
  for nv in v.neighbors do
    v.d_temperature += timestep * conduction *
      (nv.temperature - v.temperature)
  end
end

local ebb apply_diffusion ( v : bunny.vertices )
  var d_temp = v.d_temperature / v.degree
  v.temperature += d_temp

  max_temp_change max= L.fabs(d_temp)
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
    (t.v[0].temperature + t.v[1].temperature + t.v[2].temperature) / 3.0

  -- compute a display value in the range 0.0 to 1.0 from the temperature
  var scale = L.float(L.log(1.0 + avg_temp))
  if scale > 1.0 then scale = 1.0f end

  -- interpolate the hot and cold colors
  vdb.color((1.0-scale)*cold + scale*hot)
  vdb.triangle(t.v[0].pos, t.v[1].pos, t.v[2].pos)
end
-- END EXTRA VDB CODE


------------------------------------------------------------------------------


-- Execute 300 iterations of the diffusion

for i = 1,300 do
  --compute_diffusion(bunny.vertices)
  bunny.vertices:foreach(compute_diffusion)

  max_temp_change:set(0.0)
  --apply_diffusion(bunny.vertices)
  bunny.vertices:foreach(apply_diffusion)
  local max_change = max_temp_change:get()
  if i%10 == 0 then print(i, max_change) end

  --clear_temporary(bunny.vertices)
  bunny.vertices:foreach(clear_temporary)

  -- EXTRA: VDB
  vdb.vbegin()
    vdb.frame() -- this call clears the canvas for a new frame
    --debug_tri_draw(bunny.triangles)
    bunny.triangles:foreach(debug_tri_draw)
  vdb.vend()
  -- END EXTRA
end

------------------------------------------------------------------------------

-- For this file, we've omitted writing the output anywhere.
-- See the long form of bunny_diffusion.t for more details
-- on data output options

