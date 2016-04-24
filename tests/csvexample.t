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
require "tests/test"

local CSV = require 'ebb.io.csv'

local cells = L.NewRelation { dims = {10,10}, name = 'cells' }
cells:SetPartitions{2,2}

cells:NewField('foo', L.double):Load(0)

local ebb init( c : cells )
  c.foo = L.double(L.xid(c)) * L.double(L.yid(c))
end
cells:foreach(init)

-- dump output, diff and remove output
local tmp_file = "tests/csvexample.gen.csv"
cells.foo:Dump(CSV.Dump, tmp_file, { precision = 3 })
local diff_string = 'diff tests/csvexample.ref.csv ' .. tmp_file
success = os.execute(diff_string)
os.execute('rm '..tmp_file)
L.assert(success)
