-- Test partitioning on 2d grid.

import "ebb"

print("***************************************************")
print("**  This is an Ebb application intended to test  **")
print("**  partitioning over 2d grid relations          **")
print("***************************************************")

-- This example is intended to check for:
--   - correctly setting read/ write privileges for fields
--   - partitioning write into disjoint regions
--   - correctly reading globals/ constants
-- When using partitioning, this example should throw errors when:
--   - reducing fields
--   - reducing globals
local do_global_reduction = false
local do_field_reduction  = false

-- includes
local Grid  = require 'ebb.domains.grid'

-- grid
local Nx = 8
local Ny = 4
local grid = Grid.NewGrid2d {
    size   = {Nx, Ny},
    origin = {0, 0},
    width  = {3.14, 3.14},
    periodic_boundary = {true, true},
}
local C = grid.cells
local V = grid.vertices

-------------------------------------------------------------------------------
--  Initialization                                                           --
-------------------------------------------------------------------------------

-- field declarations
C:NewField('value', L.double)
V:NewField('value', L.double)

-- globals and constants
local d = L.Global(L.double, 0.8)
local sum_val = L.Global(L.double, 0)

local ebb InitCellVal(c)
  var center = c.center
  c.value = L.sin(center[0] + center[1])
  L.print(L.xid(c), L.yid(c), c.value)
end

local ebb InitVertexVal(v)
  v.value = 0
end

local ebb PrintField(r, field)
  L.print(L.xid(r), L.yid(r), r[field])
end

-- invoke initialization
local LW = require "ebb.src.legionwrap"
LW.heavyweightBarrier()
print("Invoking initialization of cells")
C:foreach(InitCellVal)
LW.heavyweightBarrier()
--print("Invoking initialization of vertices")
--V:foreach(InitVertexVal)
--LW.heavyweightBarrier()
--print("Invoking print")
--C:foreach(PrintField, 'value')
--LW.heavyweightBarrier()

-------------------------------------------------------------------------------
--  Diffuse values                                                           --
-------------------------------------------------------------------------------

-- local ebb GatherAtVerts(v)
--   v.value += v.cell(-1,  1).value + v.cell(1, -1).value
--   v.value += v.cell(-1, -1).value + v.cell(1,  1).value
--   v.value *= (d/4)
-- end
-- 
-- local ebb ScatterToVerts(c)
--   c.vertex(-1,  1).value += (d/4) * c.value
--   c.vertex( 1, -1).value += (d/4) * c.value
--   c.vertex(-1, -1).value += (d/4) * c.value
--   c.vertex( 1,  1).value += (d/4) * c.value
-- end
-- 
-- local ebb GatherAtCells(c)
--   c.value += c.vertex(-1,  1).value + c.vertex(1, -1).value
--   c.value += c.vertex(-1, -1).value + c.vertex(1,  1).value
--   c.value *= (d/4)
-- end
-- 
-- local ebb SumValue(c)
--   sum_val += c.value
-- end
-- 
-- -- loop/ main
-- for iter = 1, 4 do
--     sum_val:set(0)
--     LW.heavyweightBarrier()
--     print("Invoking diffuse to vertices", iter)
--     if do_field_reduction then
--         C:foreach(ScatterToVerts)
--     else
--         V:foreach(GatherAtVerts)
--     end
--     LW.heavyweightBarrier()
--     print("Invoking diffuse to cells", iter)
--     C:foreach(GatherAtCells)
--     if do_global_reduction then
--         C:foreach(SumValue)
--     end
--     LW.heavyweightBarrier()
--     print("Invoking print", iter)
--     C:foreach(PrintField, 'value')
--     print("Sum of values at cells = " .. tostring(sum_val:get()))
-- end
