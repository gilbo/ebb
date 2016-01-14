import "ebb"
local L = require "ebblib"

local ioOff = require 'ebb.domains.ioOff'
--local mesh  = ioOff.LoadTrimesh(
--  'INTERNAL_devapps/livecode_getting_started/octa.off')
local mesh  = ioOff.LoadTrimesh(
  'INTERNAL_devapps/livecode_getting_started/bunny.off')




local vdb = require('ebb.lib.vdb')



local timestep    = L.Global(L.double, 0.45)
local conduction  = L.Constant(L.double, 1)

mesh.vertices:NewField('t', L.double):Load(0)
mesh.vertices:NewField('d_t', L.double):Load(0)
mesh.vertices:NewField('degree', L.int):Load(0)

local function init_temperature(idx)
  if idx == 0 then return 1000 else return 0 end
end
mesh.vertices.t:Load(init_temperature)

local ebb compute_degree (e : mesh.edges)
  e.tail.degree += 1
end
mesh.edges:foreach(compute_degree)

local ebb compute_update (e : mesh.edges )
  var diff_t = e.head.t - e.tail.t

  e.head.d_t += - timestep * conduction * diff_t / e.head.degree
  e.tail.d_t +=   timestep * conduction * diff_t / e.tail.degree
end

local ebb apply_update (v : mesh.vertices)
  v.t += v.d_t
  v.d_t = 0
end

local ebb visualize ( v : mesh.vertices )
  vdb.color({ 0.5 * v.t + 0.5, 0.5-v.t, 0.5-v.t })
  vdb.point(v.pos)
end

for i=1,360 do
  mesh.edges:foreach(compute_update)
  mesh.vertices:foreach(apply_update)

  vdb.vbegin()
    vdb.frame()
    mesh.vertices:foreach(visualize)
  vdb.vend()
end