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
--
import 'ebb'
local L = require 'ebblib'
local CSV = require 'ebb.io.csv'

-- includes
local Grid  = require 'ebb.domains.grid'

-- grid parameters
local Nx = 12
local Ny = 18
local Nz = 12

-- helper functions
function VerifyDump(field, ref_file)
  local tmp_file = "tests/tmp_out.csv"
  field:Dump(CSV.Dump, tmp_file, { precision = 8 })
  local diff_string = 'diff ' .. ref_file .. ' ' .. tmp_file
  local success = os.execute(diff_string)
  os.execute('rm ' .. tmp_file)
  L.assert(success)
end

-- 3d
local grid3d = Grid.NewGrid3d {
    size   = {Nx, Ny, Nz},
    origin = {0, 0, 0},
    width  = {4, 4, 4},
    periodic_boundary = {true, true, true},
}
local C3d = grid3d.cells
local V3d = grid3d.vertices
C3d:SetPartitions{2,3,2}
V3d:SetPartitions{2,3,2}

C3d:NewField("sval", L.double)

local ebb s_set_c_3d(c : C3d)
  var dx = L.double(L.xid(c))
  var dy = L.double(L.yid(c))
  var dz = L.double(L.zid(c))
  var d = Nz*(Ny*dx + dy) + dz
  c.sval = d
end

local ebb s_reduce_uncentered_3d (c : C3d)
  var dx = L.double(L.xid(c))
  var dy = L.double(L.yid(c))
  var dz = L.double(L.yid(c))
  var d = Nz*(Ny*dx + dy) + dz
  c(-1, 0, 0).sval += 0.0023 + 0.01*d
  c( 1, 0, 0).sval += 0.0023 + 0.01*d
  c( 0,-1, 0).sval += 0.0023 + 0.01*d
  c( 0, 1, 0).sval += 0.0023 + 0.01*d
  c( 0, 0,-1).sval += 0.0023 + 0.01*d
  c( 0, 0, 1).sval += 0.0023 + 0.01*d
end

C3d:foreach(s_set_c_3d)
C3d:foreach(s_reduce_uncentered_3d)

VerifyDump(C3d.sval, "tests/fieldreduce_grid_faces_3d.csv")
