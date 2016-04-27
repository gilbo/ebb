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

local Grid  = require 'ebb.domains.grid'
local CSV = require 'ebb.io.csv'

local Nx = 10
local Ny = 12
local Nz = 16
local grid = Grid.NewGrid3d {
    size   = {Nx, Ny, Nz},
    origin = {0, 0, 0},
    width  = {20, 20, 20},
    periodic_boundary = {true, true, true},
}
local C = grid.cells
C:SetPartitions {2,3,4}
C:NewField('f', L.double)
C:NewField('f_new', L.double)

local ebb InitCellVal(c)
  c.f = L.double(L.xid(c) + L.yid(c))
end

local ebb Diffuse(c)
  var avg = (1.0/6.0) * (c(1,0,0).f + c(-1,0,0).f + c(0,1,0).f + c(0,-1,0).f + c(0,0,1).f + c(0,0,-1).f)
  c.f_new = c.f + 0.25 * (avg - c.f)
end

local ebb RoundValues(c)
  c.f = L.double(L.floor(c.f * 1000))/1000.0
end

C:foreach(InitCellVal)
C:foreach(Diffuse)
C:foreach(RoundValues)

function VerifyDump(field, ref_file)
  local tmp_file = "tests/tmp_out.csv"
  field:Dump(CSV.Dump, tmp_file, { precision = 3 })
  local diff_string = 'diff ' .. ref_file .. ' ' .. tmp_file
  local success = os.execute(diff_string)
  os.execute('rm ' .. tmp_file)
  L.assert(success)
end

VerifyDump(C.f_new, "tests/grid_3d_face_reads.ref.csv")
