import "ebb"

local ioOff = require 'ebb.domains.ioOff'
local mesh  = ioOff.LoadTrimesh(
  'devapps/livecode_getting_started/octa.off')
--local mesh  = ioOff.LoadTrimesh(
--  'devapps/livecode_getting_started/bunny.off')


-- Ok, let's try to get some visual output.
-- I usually use VDB while developing to do this.
-- We created a simple Ebb wrapper around VDB so it's easy to use.
-- We can load this wrapper like any other library.

local vdb   = require('ebb.lib.vdb')

--[[---------------------------

local ebb visualize ( v : mesh.vertices )
  vdb.point(v.pos)
end

mesh.vertices:foreach(visualize)

-----------------------------

local ebb visualize ( v : mesh.vertices )
  vdb.color({1,1,0})
  vdb.point(v.pos)
end

mesh.vertices:foreach(visualize)

-----------------------------


-- Let's oscillate the octahedron now
-- In order to prevent messing up the original position, let's just
-- create a new position field

mesh.vertices:NewField('q', L.vec3d):Load({0,0,0})

local ebb set_oscillation ( v : mesh.vertices )
  v.q = v.pos
end

mesh.vertices:foreach(set_oscillation)


-----------------------------]]


-- Ok, but how do we get this to change over time...?
-- Ebb supports global variables that are not fields
-- so let's set up a global variable to represent time

local time = L.Global(L.double, 0)

-- now we can set an oscillation

mesh.vertices:NewField('q', L.vec3d):Load({0,0,0})

local ebb set_oscillation ( v : mesh.vertices )
  v.q = 0.5*( L.sin(time) + 1) * v.pos
end

mesh.vertices:foreach(set_oscillation)

local ebb visualize ( v : mesh.vertices )
  vdb.color({1,1,0})
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






--vdb.vbegin()
--vdb.frame() -- this call clears the canvas for a new frame
--    bunny.triangles:foreach(debug_tri_draw)
--vdb.vend()
