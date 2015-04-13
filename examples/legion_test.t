-- This file is to test integration of Liszt with Legion. Add code to test
-- features as they are implemented.

print("* This is a Liszt application *")

import "compiler.liszt"
local g_scal = L.Global(L.int, 4)

-- Create relations and fields

local cells = L.NewRelation { name = 'cells_1d', size = 4 }
local dual_cells = L.NewRelation { name = 'dual_cells_1d', size = 5 }
cells:NewField('dual_left', dual_cells):Load({0, 1, 2, 3})
cells:NewField('dual_right', dual_cells):Load({1, 2, 3, 4})
cells:NewSubsetFromFunction('interior',
                            function(x)
                              return (x > 0 and x < 3) end)

-- local cells = L.NewRelation { name = 'cells_2d', dims = {4,1} }
-- local dual_cells = L.NewRelation { name = 'dual_cells_2d', dims = {5,1} }
-- cells:NewField('dual_left', dual_cells):Load({{{0, 0},  {1, 0}, {2, 0}, {0, 0}}})
-- cells:NewField('dual_right', dual_cells):Load({{{1, 0}, {2, 0}, {3, 0}, {4, 0}}})

cells:NewField('x', L.int)
cells:NewField('y', L.double)

dual_cells:NewField('a', L.double)

-- Globals
local g_scal = L.Global(L.int, 4)
local g_vec  = L.Global(L.vec2d, {0, 0})
-- THIS FAILS RIGHT NOW BECAUSE OF TYPE CHECKING ERRORS
-- local g_mat  = L.Global(L.mat3i, { {10, 2, 3}, {4, 50, 6}, {7, 8, 100} })

local liszt CenteredReads(c : cells)
  L.print(c.x)
  L.print(c.y)
end

local liszt CenteredWrite(c : cells)
  c.x = 1
end

local liszt CenteredMul(c : cells)
  c.y = 0.2 * c.x
end

local liszt ReduceField(c : cells)
  c.y += 0.1
end

local liszt ReduceGlobalVec(c : cells)
  c.y += 0.3
  g_vec += L.vec2d({0.2, 0.1})
end

local liszt InitDual(d : dual_cells)
  d.a = 0.1
end

local liszt InitCells(c : cells)
  c.x = 3
  c.y = 0.45
end

local liszt CollectDual(c : cells)
  L.print(c.dual_left)
  c.y += c.dual_left.a
  c.y += c.dual_right.a
end

-- global and field reads, writes, reduction
local function test_accesses()
  print(g_scal:get())
  terralib.tree.printraw(g_vec:get())
  cells:map(CenteredWrite)
  cells:map(CenteredMul)
  cells:map(CenteredReads)
  cells:map(ReduceField)
  cells:map(ReduceGlobalVec)
  cells:map(CenteredReads)
  terralib.tree.printraw(g_vec:get())
end
test_accesses()

-- keyfield functionality
local function test_keyfields()
  dual_cells:map(InitDual)
  cells:map(InitCells)
  cells:map(CenteredReads)
  cells:map(CollectDual)
  cells:map(CenteredReads)
end
test_keyfields()

-- subset functionality
local function test_subsets()
  cells.interior:map(InitCells)
  cells.interior:map(CenteredReads)
  cells:map(CenteredReads)
end
test_subsets()

local verts = L.NewRelation { name = 'verts', size = 8 }
verts:NewField('t', L.float):Load(0)
verts:NewField('pos', L.vec3d):Load({1,2,3})

local grid = L.NewRelation { name = 'cells', dims = {4,3,2} }
grid:NewField('p', L.double):Load(0)
