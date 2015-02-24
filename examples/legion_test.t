-- This file is to test integration of Liszt with Legion. Add code to test
-- features as they are implemented.

print("* This is a Liszt application *")

import "compiler.liszt"
local g_scal = L.NewGlobal(L.int, 4)

-- Create relations and fields
-- local points = L.NewRelation(4, 'points')
local points = L.NewRelation { name = 'points', dim = {3,2} }

points:NewField('x', L.int)
points:NewField('y', L.int)
points:NewField('z', L.int)
points:NewField('t', L.int)

-- Globals
local g_scal = L.NewGlobal(L.int, 4)
local g_vec  = L.NewGlobal(L.vec2d, {0, 0})
-- THIS FAILS RIGHT NOW BECAUSE OF TYPE CHECKING ERRORS
-- local g_mat  = L.NewGlobal(L.mat3i, { {10, 2, 3}, {4, 50, 6}, {7, 8, 100} })

print(g_scal:get())
terralib.tree.printraw(g_vec:get())

local liszt kernel CenteredWrite(p : points)
  p.x = 1
end

local liszt kernel CenteredAdd(p : points)
  p.y = 2
  p.z = p.x + p.y
  p.z = p.z + 1
  p.z
end

local liszt kernel ReduceField(p : points)
  p.y += 7
  p.z *= 2
  p.y
  p.z
end

local liszt kernel ReduceGlobalVec(p : points)
  p.z = 3
  g_vec += L.vec2d({0.2, 0.1})
end

CenteredWrite(points)
CenteredWrite(points)
CenteredAdd(points)
CenteredAdd(points)
ReduceField(points)

ReduceGlobalVec(points)

terralib.tree.printraw(g_vec:get())

local verts = L.NewRelation { name = 'verts', size = 8 }
verts:NewField('t', L.float):Load(0)
verts:NewField('pos', L.vec3d):Load({1,2,3})

local grid = L.NewRelation { name = 'cells', dim = {4,3,2} }
grid:NewField('p', L.double):Load(0)
