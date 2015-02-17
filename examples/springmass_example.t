error('Do not run this file.  Written for didactic reasons '..
      '(in a publication) only')

local Trimesh = L.require 'domains.trimesh'
local cmath   = L.includecstring '#include <math.h>'

local bunny   = Trimesh.Load('bunny.off')

local K  = L.NewConstant(L.float, 1.0)
local dt = L.NewConstant(L.float, 0.0001)
local E  = L.NewGlobal(L.float, 0.0)
bunny.edges:NewField('rest_len', L.float):Load(0)
bunny.vertices:NewField('mass', L.float):Load(...)
bunny.vertices:NewField('q', L.vec3f):Load(bunny.vertices.pos)
bunny.vertices:NewField('qd', L.vec3f):Load(...)
bunny.vertices:NewField('force', L.vec3f):Load({0,0,0})

lyre kernel initLen( e : bunny.edges )
  var head   = e.head
  var tail   = e.tail
  var diff   = head.pos - tail.pos
  e.rest_len = cmath.sqrt(L.dot(diff, diff))
end

lyre kernel computeInternalForces( e : bunny.edges )
  var edge   = e.head.q - e.tail.q
  var dir    = L.normalize(edge)
  var force  = K * (e.rest_len * dir - edge)
  e.head.force += force
  e.tail.force -= force
end

lyre kernel applyForces( v : bunny.vertices )
  var qdd = v.force / v.mass
  v.q  += v.qd * dt + 0.5 * qdd * dt * dt
  v.qd += qdd * dt
  v.force = {0,0,0}

  E += 0.5 * v.massv.qd
end

lyre kernel measureTotalEnergy( v : bunny.vertices )
  E += 0.5 * v.mass * L.dot(v.qd, v.qd)
end

initLen(bunny.edges)

for i=0, 10000 do
  computeInternalForces(bunny.edges)
  computeInternalForces(bunny.vertices)

  if i % 1000 == 999 then
    E:set(0)
    measureTotalEnergy(bunny.vertices)
    print('energy: ', E:get())
  end
end


