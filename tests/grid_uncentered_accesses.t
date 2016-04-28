-- Test partitioning on 2d grid.

import "ebb"
local L = require "ebblib"

--[[
print("***************************************************")
print("**  This is an Ebb application intended to test  **")
print("**  partitioning over 2d grid relations          **")
print("***************************************************")
]]

-- This example is intended to check for:
--   - correctly setting read/ write privileges for fields
--   - partitioning write into disjoint regions
--   - correctly reading globals/ constants
local do_global_reduction = true
local do_field_reduction  = false

-- includes
local Grid  = require 'ebb.domains.grid'
local CSV = require 'ebb.io.csv'

-- grid
local Nx = 10
local Ny = 12
local grid = Grid.NewGrid2d {
    size   = {Nx, Ny},
    origin = {0, 0},
    width  = {20, 20},
    periodic_boundary = {true, true},
}
local C = grid.cells
local V = grid.vertices
C:SetPartitions {2,3}
V:SetPartitions {2,3}

-------------------------------------------------------------------------------
--  Initialization                                                           --
-------------------------------------------------------------------------------

-- field declarations
C:NewField('value', L.double)
V:NewField('value', L.double)

-- globals and constants
local d = L.Global(L.double, 0.9)
local sum_val = L.Global(L.double, 0)

local ebb InitCellVal(c)
  var center = c.center
  c.value = center[0]*0.1 * center[1] + 0.1
end

local ebb InitVertexVal(v)
  v.value = L.double(L.xid(v) + L.yid(v))
end

-- invoke initialization
C:foreach(InitCellVal)
V:foreach(InitVertexVal)

-------------------------------------------------------------------------------
--  Diffuse values                                                           --
-------------------------------------------------------------------------------

local ebb ScatterToVerts(c)
  c.vertex(-1,  1).value += d * c.value
  c.vertex( 1, -1).value += d * c.value
  c.vertex(-1, -1).value += d * c.value
  c.vertex( 1,  1).value += d * c.value
end

local ebb ScaleVerts(v)
  v.value *= 0.2
end

local ebb GatherAtCells(c)
  c.value += c.vertex(-1,  1).value + c.vertex(1, -1).value
  c.value += c.vertex(-1, -1).value + c.vertex(1,  1).value
  c.value *= (0.2 / d) * 0.95
  var center = c.center
  c.value += 0.05 * center[0]*0.1 * center[1]
end

local ebb SumValue(c)
  sum_val += c.value
end

local ebb RoundValues(c)
  c.value = L.double(L.floor(c.value * 1000))/1000.0
end

-- loop/ main
for iter = 1, 100 do
    sum_val:set(0)
    C:foreach(ScatterToVerts)
    V:foreach(ScaleVerts)
    C:foreach(GatherAtCells)
    C:foreach(RoundValues)
    V:foreach(RoundValues)
    C:foreach(SumValue)
end

function VerifyDump(field, ref_file)
  local tmp_file = "tests/tmp_out.csv"
  field:Dump(CSV.Dump, tmp_file, { precision = 3 })
  local diff_string = 'diff ' .. ref_file .. ' ' .. tmp_file
  local success = os.execute(diff_string)
  os.execute('rm ' .. tmp_file)
  L.assert(success == 0)
end

-- dump output, diff and remove output
VerifyDump(C.value, "tests/grid_uncentered_accesses_cells.ref.csv")
VerifyDump(V.value, "tests/grid_uncentered_accesses_verts.ref.csv")
