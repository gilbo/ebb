import "compiler.liszt"

local ioOff = L.require 'domains.ioOff'
--local mesh  = ioOff.LoadTrimesh(
--  'examples/livecode_getting_started/octa.off')
local mesh  = ioOff.LoadTrimesh(
  'examples/livecode_getting_started/bunny.off')

local vdb   = L.require('lib.vdb')


--[[

local time = L.Global(L.double, 0)

mesh.vertices:NewField('q', L.vec3d):Load({0,0,0})
mesh.vertices:NewField('t', L.double):Load(0)       -- <-------------

local liszt set_oscillation ( v : mesh.vertices )
  v.q = 0.5*( L.sin(time) + 1) * v.pos
  v.t = 0.5*( L.sin(time) + 1)      -- <-------------
end

local liszt visualize ( v : mesh.vertices )
  vdb.color({v.t,v.t,0})            -- <-------------
  vdb.point(v.q)
end

for i=1,360 do
  for k=1,10000 do end

  time:set(i * math.pi / 180.0)
  mesh.vertices:foreach(set_oscillation)

  vdb.vbegin()
  vdb.frame()
    mesh.vertices:foreach(visualize)
  vdb.vend()
end

-----------------------------

local timestep = L.Global(L.double, 0.45)
local conduction = L.Constant(L.double, 1.0)

mesh.vertices:NewField('t', L.double):Load(0)

local function init_temperature(idx)
  if idx == 0 then return 1000 else return 0 end
end
mesh.vertices.t:Load(init_temperature)


local liszt compute_update ( v : mesh.vertices )
  var sum_t : L.double = 0.0
  var count            = 0

  for e in v.edges do
    sum_t += e.head.t
    count += 1
  end

  var avg_t  = sum_t / count
  var diff_t = avg_t - v.t
  v.t += timestep * conduction * diff_t
end


local liszt visualize ( v : mesh.vertices )
  vdb.color({ 0.5 * v.t + 0.5, 0.5-v.t, 0.5-v.t })
  vdb.point(v.pos)
end

for i=1,360 do
  mesh.vertices:foreach(compute_update)

  vdb.vbegin()
  vdb.frame()
    mesh.vertices:foreach(visualize)
  vdb.vend()
end



-----------------------------]]



local timestep = L.Global(L.double, 0.45)
local conduction = L.Constant(L.double, 1.0)

mesh.vertices:NewField('t', L.double):Load(0)
mesh.vertices:NewField('d_t', L.double):Load(0)

local function init_temperature(idx)
  if idx == 0 then return 1000 else return 0 end
end
mesh.vertices.t:Load(init_temperature)


local liszt compute_update ( v : mesh.vertices )
  var sum_t : L.double = 0.0
  var count            = 0

  for e in v.edges do
    sum_t += e.head.t
    count += 1
  end

  var avg_t  = sum_t / count
  var diff_t = avg_t - v.t
  v.d_t = timestep * conduction * diff_t
end

local liszt apply_update ( v : mesh.vertices )
  v.t += v.d_t
end


local liszt visualize ( v : mesh.vertices )
  vdb.color({ 0.5 * v.t + 0.5, 0.5-v.t, 0.5-v.t })
  vdb.point(v.pos)
end

for i=1,360 do
  mesh.vertices:foreach(compute_update)
  mesh.vertices:foreach(apply_update)

  vdb.vbegin()
  vdb.frame()
    mesh.vertices:foreach(visualize)
  vdb.vend()
end





--vdb.vbegin()
--vdb.frame() -- this call clears the canvas for a new frame
--    bunny.triangles:foreach(debug_tri_draw)
--vdb.vend()
