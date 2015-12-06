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
local test = require "tests/test"


local ioOff = require 'ebb.domains.ioOff'
local mesh  = ioOff.LoadTrimesh('tests/octa.off')

------------------
-- Should pass: --
------------------
local vk = ebb (v : mesh.vertices)
    var x = {5, 4, 3}
    v.pos += x
end
mesh.vertices:foreach(vk)

local x_out = L.Global(L.float, 0.0)
local y_out = L.Global(L.float, 0.0)
local y_idx = L.Global(L.int, 1)
local read_out_const = ebb(v : mesh.vertices)
    x_out += L.float(v.pos[0])
end
local read_out_var = ebb(v : mesh.vertices)
    y_out += L.float(v.pos[y_idx])
end
mesh.vertices:foreach(read_out_const)
mesh.vertices:foreach(read_out_var)

local avgx = x_out:get() / mesh.vertices:Size()
local avgy = y_out:get() / mesh.vertices:Size()
test.fuzzy_eq(avgx, 5)
test.fuzzy_eq(avgy, 4)

------------------
-- Should fail: --
------------------
idx = 3.5
test.fail_function(function()
  local ebb t(v : mesh.vertices)
      v.pos[idx] = 5
  end
  mesh.vertices:foreach(t)
end, "expected an integer")

