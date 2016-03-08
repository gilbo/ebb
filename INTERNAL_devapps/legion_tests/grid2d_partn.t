-- Test partitioning on 2d grid.

import "ebb"
local L = require "ebblib"

print("***************************************************")
print("**  This is an Ebb application intended to test  **")
print("**  partitioning over 2d grid relations          **")
print("***************************************************")

-- This example is intended to check for:
--   - correctly setting read/ write privileges for fields
--   - partitioning write into disjoint regions
--   - correctly reading globals/ constants
local do_global_reduction = true
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
end

local ebb InitVertexVal(v)
  v.value = 0
end

-- invoke initialization
C:foreach(InitCellVal)
V:foreach(InitVertexVal)

-------------------------------------------------------------------------------
--  Diffuse values                                                           --
-------------------------------------------------------------------------------

local ebb GatherAtVerts(v)
  v.value += v.cell(-1,  1).value + v.cell(1, -1).value
  v.value += v.cell(-1, -1).value + v.cell(1,  1).value
  v.value *= (d/4)
  -- L.print(L.xid(v), L.yid(v), v.value)
  -- L.print(v.cell(-1, 1).value, v.cell(1, -1).value, v.cell(-1, -1).value, v.cell(1,1).value)
end

local ebb ScatterToVerts(c)
  c.vertex(-1,  1).value += (d/4) * c.value
  c.vertex( 1, -1).value += (d/4) * c.value
  c.vertex(-1, -1).value += (d/4) * c.value
  c.vertex( 1,  1).value += (d/4) * c.value
end

local ebb GatherAtCells(c)
  c.value += c.vertex(-1,  1).value + c.vertex(1, -1).value
  c.value += c.vertex(-1, -1).value + c.vertex(1,  1).value
  c.value *= (d/4)
end

local ebb SumValue(c)
  sum_val += c.value
end

-- loop/ main
for iter = 1, 2 do
    sum_val:set(0)
    if do_field_reduction then
        C:foreach(ScatterToVerts)
    else
        V:foreach(GatherAtVerts)
    end
    C:foreach(GatherAtCells)
    if do_global_reduction then
        C:foreach(SumValue)
    end
    C.value:Print()
    V.value:Print()
    print("Sum of values at cells = " .. tostring(sum_val:get()))
end
