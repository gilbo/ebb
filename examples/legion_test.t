-- This file is to test integration of Liszt with Legion. Add code to test
-- features as they are implemented.

print("* This is a Liszt application *")

import "compiler.liszt"
local g_scal = L.NewGlobal(L.int, 4)

-- Create relations and fields

local cells = L.NewRelation { name = 'cells', size = 3 }
-- local cells = L.NewRelation { name = 'cells', dim = {2,1} }

local dual_cells = L.NewRelation { name = 'dual_cells', size = 4 }
-- local dual_cells = L.NewRelation { name = 'dual_cells', dim = {3,1} }

cells:NewField('dual_left', cells):Load({0, 1, 2})
cells:NewField('dual_right', cells):Load({1, 2, 3})

cells:NewField('x', L.int)
cells:NewField('y', L.double)

dual_cells:NewField('z', L.double)

-- Globals
local g_scal = L.NewGlobal(L.int, 4)
local g_vec  = L.NewGlobal(L.vec2d, {0, 0})
-- THIS FAILS RIGHT NOW BECAUSE OF TYPE CHECKING ERRORS
-- local g_mat  = L.NewGlobal(L.mat3i, { {10, 2, 3}, {4, 50, 6}, {7, 8, 100} })

print(g_scal:get())
terralib.tree.printraw(g_vec:get())

local liszt kernel CenteredReads(c : cells)
  c.x
  c.y
end

local liszt kernel CenteredWrite(c : cells)
  c.x = 1
end

local liszt kernel CenteredMul(c : cells)
  c.y = 0.2 * c.x
end

local liszt kernel ReduceField(c : cells)
  c.y += 0.1
end

local liszt kernel ReduceGlobalVec(c : cells)
  c.y += 0.3
  g_vec += L.vec2d({0.2, 0.1})
end

-- CenteredWrite(cells)
-- CenteredMul(cells)
-- ReduceField(cells)
-- ReduceGlobalVec(cells)
-- CenteredReads(cells)

terralib.tree.printraw(g_vec:get())

local liszt kernel InitDual(d : dual_cells)
  d.z = 0.1
end

local liszt kernel InitCells(c : cells)
  c.x = 3
  c.y = 0.45
end

local liszt kernel CollectDual(c : cells)
  -- c.y += c.dual_left.z
  -- c.y += c.dual_right.z
  -- c.dual_right
  c.dual_left
  c.dual_left.z
end

InitDual(dual_cells)
InitCells(cells)
-- CenteredReads(cells)
CollectDual(cells)
-- CenteredReads(cells)

local verts = L.NewRelation { name = 'verts', size = 8 }
verts:NewField('t', L.float):Load(0)
verts:NewField('pos', L.vec3d):Load({1,2,3})

local grid = L.NewRelation { name = 'cells', dim = {4,3,2} }
grid:NewField('p', L.double):Load(0)
