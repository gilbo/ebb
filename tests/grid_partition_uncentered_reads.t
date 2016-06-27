-- The MIT License (MIT)
-- 
-- Copyright (c) 2015 Stanford University.
-- All rights reserved.
-- 
-- Permission is hereby granted, free of charge, to any person obtaining a
-- copy of this software and associated documentation files (the "Software"),
-- to deal in the Software without restriction, including without limitation
-- the rights to use, copy, modify, merge, publish, distribute, sublicense,
-- and/or sell copies of the Software, and to permit persons to whom the
-- Software is furnished to do so, subject to the following conditions:
-- 
-- The above copyright notice and this permission notice shall be included
-- in all copies or substantial portions of the Software.
-- 
-- THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
-- IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
-- FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
-- AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
-- LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
-- FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
-- DEALINGS IN THE SOFTWARE.
import 'ebb'
local L = require 'ebblib'

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
    partitions        = {2,3},
}
local C = grid.cells
local V = grid.vertices

-------------------------------------------------------------------------------
--  Initialization                                                           --
-------------------------------------------------------------------------------

-- field declarations
C:NewField('value', L.double)
V:NewField('value', L.double)

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
--  Read values                                                              --
-------------------------------------------------------------------------------

local ebb ScaleVerts(v)
  v.value *= 0.98
end

local ebb GatherAtCells(c)
  c.value += c.vertex(-1,  1).value + c.vertex(1, -1).value
  c.value += c.vertex(-1, -1).value + c.vertex(1,  1).value
  c.value *= 1
  var center = c.center
  c.value += 0.05 * center[0]*0.1 * center[1]
end

local ebb RoundValues(c)
  c.value = L.double(L.floor(c.value * 1000))/1000.0
end

-- loop/ main
for iter = 1, 100 do
    V:foreach(ScaleVerts)
    C:foreach(GatherAtCells)
    C:foreach(RoundValues)
    V:foreach(RoundValues)
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
VerifyDump(C.value, "tests/grid_partition_uncentered_reads_cells.ref.csv")
VerifyDump(V.value, "tests/grid_partition_uncentered_reads_verts.ref.csv")