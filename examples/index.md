---
layout: page
title: "Examples"
date: 
modified:
excerpt:
tags: []
image:
  feature:
---


Spring-Mass Simulation

```
import "ebb"
local L = require "ebblib"

local ioVeg = require 'ebb.domains.ioVeg'
local PN    = require 'ebb.lib.pathname'
local dragon = ioVeg.LoadTetmesh(PN.scriptdir()..'dragon.veg')

-- Define simulation constants
local dt       = L.Constant(L.double, 0.0002)
local K        = L.Constant(L.double, 2000.0)
local damp     = L.Constant(L.double, 0.5)
local gravity  = L.Constant(L.vec3d, {0,-0.98,0})

-- Define simulation fields
dragon.vertices:NewField('vel', L.vec3d)      :Load({0,0,0})
dragon.vertices:NewField('acc', L.vec3d)      :Load({0,0,0})
dragon.edges:NewField('rest_len', L.double)   :Load(0)

-- Initialize Rest Length
local ebb init_rest_len ( e : dragon.edges )
  var diff = e.head.pos - e.tail.pos
  e.rest_len = L.length(diff)
end
dragon.edges:foreach(init_rest_len)

-- Translate the dragon upwards
local ebb lift_dragon ( v : dragon.vertices )
  v.pos += {0,2.3,0}
end
dragon.vertices:foreach(lift_dragon)

------------------------------------------------------------

local ebb compute_acceleration ( v : dragon.vertices )
  var force = { 0.0, 0.0, 0.0 }

  -- Pseudo-Physical Spring Force
  var mass = 0.0
  for e in v.edges do
    var diff  = e.head.pos - v.pos
    var scale = (e.rest_len / L.length(diff)) - 1.0
    mass     += e.rest_len
    force    -= K * scale * diff
  end
  force = force / mass

  -- Ground Force
  if v.pos[1] < 0.0 then
    force += { 0, - K * v.pos[1], 0 }
  end

  v.acc = force + gravity
end

local ebb update_vel_pos ( v : dragon.vertices )
  v.pos += dt * v.vel + 0.5 * dt * dt * v.acc
  v.vel = (1 - damp * dt) * v.vel + (dt * v.acc)
end

------------------------------------------------------------

-- Simulation Loop
for i=1,40000 do
  dragon.vertices:foreach(compute_acceleration)
  dragon.vertices:foreach(update_vel_pos)
  if i%1000 == 0 then print('iter', i) end
end
```