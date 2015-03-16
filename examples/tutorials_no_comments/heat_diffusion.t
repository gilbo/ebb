import "compiler.liszt"

local Trimesh = L.require 'domains.trimesh'
local PN = L.require 'lib.pathname'
local cmath = terralib.includecstring '#include <math.h>'

local tri_mesh_filename = PN.scriptdir() .. 'bunny.off'
local bunny = Trimesh.LoadFromOFF(tri_mesh_filename)

------------------------------------------------------------------------------

local timestep = L.Global(L.double, 0.45)
local conduction = L.Constant(L.double, 1.0)

bunny.vertices:NewField('temperature', L.double):Load(function(vi)
  if vi == 0 then return 3000.0 else return 0.0 end
end)
bunny.vertices:NewField('d_temperature', L.double):Load(0.0)

------------------------------------------------------------------------------

local liszt compute_update ( v : bunny.vertices )
  var sum_t : L.double = 0.0
  var count            = 0

  for e in v.edges do
    sum_t += e.head.temperature
    count += 1
  end

  var avg_t  = sum_t / count
  var diff_t = avg_t - v.temperature
  v.d_temperature = timestep * conduction * diff_t
end

local liszt apply_update ( v : bunny.vertices )
  v.temperature += v.d_temperature
end

------------------------------------------------------------------------------

-- WARNING / EXTRA: VDB
local vdb  = L.require('lib.vdb')
local cold = L.Constant(L.vec3f,{0.5,0.5,0.5})
local hot  = L.Constant(L.vec3f,{1.0,0.0,0.0})
local liszt debug_tri_draw ( t : bunny.triangles )
  var avg_temp = 0.0
  for i=0,3 do avg_temp += t.v[i].temperature end
  avg_temp = avg_temp / 3.0

  var scale = L.float(cmath.log(1.0 + avg_temp))
  if scale > 1.0 then scale = 1.0f end

  vdb.color((1.0-scale)*cold + scale*hot)
  vdb.triangle(t.v[0].pos, t.v[1].pos, t.v[2].pos)
end
-- END EXTRA VDB CODE

------------------------------------------------------------------------------

for i = 1,300 do
  bunny.vertices:map(compute_update)
  bunny.vertices:map(apply_update)

  vdb.vbegin()
    vdb.frame()
    bunny.triangles:map(debug_tri_draw)
  vdb.vend()
end

