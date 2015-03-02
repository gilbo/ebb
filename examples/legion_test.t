-- This file is to test integration of Liszt with Legion. Add code to test
-- features as they are implemented.

print("* This is a Liszt application *")

import "compiler.liszt"
local g_scal = L.NewGlobal(L.int, 4)

-- Create relations and fields

local cells = L.NewRelation { name = 'cells_1d', size = 4 }
local dual_cells = L.NewRelation { name = 'dual_cells_1d', size = 5 }
cells:NewField('dual_left', dual_cells):Load({0, 1, 2, 3})
cells:NewField('dual_right', dual_cells):Load({1, 2, 3, 4})
cells:NewSubsetFromFunction('interior',
                            function(x)
                              return (x > 0 and x < 3) end)

-- local cells = L.NewRelation { name = 'cells_2d', dim = {4,1} }
-- local dual_cells = L.NewRelation { name = 'dual_cells_2d', dim = {5,1} }
-- cells:NewField('dual_left', dual_cells):Load({{{0, 0},  {1, 0}, {2, 0}, {0, 0}}})
-- cells:NewField('dual_right', dual_cells):Load({{{1, 0}, {2, 0}, {3, 0}, {4, 0}}})

cells:NewField('x', L.int)
cells:NewField('y', L.double)

dual_cells:NewField('a', L.double)

-- Globals
local g_scal = L.NewGlobal(L.int, 4)
local g_vec  = L.NewGlobal(L.vec2d, {0, 0})
-- THIS FAILS RIGHT NOW BECAUSE OF TYPE CHECKING ERRORS
-- local g_mat  = L.NewGlobal(L.mat3i, { {10, 2, 3}, {4, 50, 6}, {7, 8, 100} })

local liszt kernel CenteredReads(c : cells)
  L.print(c.x)
  L.print(c.y)
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

local liszt kernel InitDual(d : dual_cells)
  d.a = 0.1
end

local liszt kernel InitCells(c : cells)
  c.x = 3
  c.y = 0.45
end

local liszt kernel CollectDual(c : cells)
  L.print(c.dual_left)
  c.y += c.dual_left.a
  c.y += c.dual_right.a
end

-- global and field reads, writes, reduction
local function test_accesses()
  print(g_scal:get())
  terralib.tree.printraw(g_vec:get())
  CenteredWrite(cells)
  CenteredMul(cells)
  CenteredReads(cells)
  ReduceField(cells)
  ReduceGlobalVec(cells)
  CenteredReads(cells)
  terralib.tree.printraw(g_vec:get())
end
test_accesses()

-- keyfield functionality
local function test_keyfields()
  InitDual(dual_cells)
  InitCells(cells)
  CenteredReads(cells)
  CollectDual(cells)
  CenteredReads(cells)
end
test_keyfields()

-- subset functionality
local function test_subsets()
  InitCells(cells.interior)
  CenteredReads(cells.interior)
  CenteredReads(cells)
end
test_subsets()

local verts = L.NewRelation { name = 'verts', size = 8 }
verts:NewField('t', L.float):Load(0)
verts:NewField('pos', L.vec3d):Load({1,2,3})

local grid = L.NewRelation { name = 'cells', dim = {4,3,2} }
grid:NewField('p', L.double):Load(0)
