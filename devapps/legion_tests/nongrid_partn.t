-- Test partitioning on non-grid relations.

import "ebb"

print("***************************************************")
print("**  This is an Ebb application intended to test  **")
print("**  partitioning over non-grid relations         **")
print("***************************************************")

-- This example is intended to check for:
--   - correctly setting read/ write privileges for fields
--   - partitioning write into disjoint regions
--   - correctly reading globals/ constants
-- When using partitioning, this example should throw errors when:
--   - reducing fields
--   - reducing globals

-- includes
local DLD = require "ebb.src.dld"
local PN = require 'ebb.lib.pathname'
local ioOff = require 'ebb.domains.ioOff'
local C = terralib.includecstring([[ #include <stdio.h> ]])

-- mesh
local mesh_filename = PN.scriptdir() .. 'octa.off'
local mesh = ioOff.LoadTrimesh('tests/octa.off')
local V = mesh.vertices
local T = mesh.triangles
local E = mesh.edges

-------------------------------------------------------------------------------
--  Initialization                                                           --
-------------------------------------------------------------------------------

-- field delcarations
E:NewField('delta', L.double):Load(0)
E:NewField('unit_edge', L.vec3d)
E:NewField('rest_len', L.double)
V:NewField('cur_pos', L.vec3d)
V:NewField('v', L.vec3d):Load({0, 0, 0})
V:NewField('f', L.vec3d):Load({0, 0, 0})
V:NewField('mass', L.double):Load(1)

-- field initialization
local ebb InitLength(e)
  e.rest_len = L.length(e.head.pos - e.tail.pos)
end
local ebb InitCurPos(v)
  v.cur_pos = 1.5 * v.pos
end

-- printer functions
local ebb PrintField(r, field)
  L.print(L.id(r), r[field])
end

-- invoke initialization
E:foreach(InitLength)
V:foreach(InitCurPos)

-- print out fields
E:foreach(PrintField, 'rest_len')
V:foreach(PrintField, 'pos')

-------------------------------------------------------------------------------
--  Compute new positions                                                    --
-------------------------------------------------------------------------------

local K  = L.Constant(L.double, 10)
local dt = L.Global(L.double, 0.1)

local ebb ComputeDelta(e)  -- positive delta means elongation => pull vertices
  var vec_edge   = e.head.cur_pos - e.tail.cur_pos
  var new_length = L.length(vec_edge)
  e.delta = new_length - e.rest_len
  e.unit_edge = vec_edge/new_length
end

local ebb GatherForce(v)
  for e in v.edges do
    v.f += K * e.delta * e.unit_edge
  end
end
local ebb ScatterForce(e)
  e.tail.f += K * e.delta * e.unit_edge
end

local ebb UpdateKinematics(v)
  var a = v.f / v.mass
  v.cur_pos += v.v * dt + 0.5 * a * dt * dt
  v.v += a * dt
end

-- loop/ main sim
for iter = 1, 4 do
  E:foreach(ComputeDelta)
  -- Scatter should not work with multiple partitions till we use Legion's
  -- recution API.
  V:foreach(GatherForce)
  -- E:foreach(ScatterForce)
  V:foreach(UpdateKinematics)
  V:foreach(PrintField, 'cur_pos')
  -- V:foreach(PrintField, 'f')
end
