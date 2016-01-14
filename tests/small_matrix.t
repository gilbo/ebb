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
require 'tests/test'
--local types = require "ebb.src.types"


local ioOff = require 'ebb.domains.ioOff'
local mesh  = ioOff.LoadTrimesh('tests/octa.off')

mesh.vertices:NewField('tensor_pos', L.mat3d)
mesh.vertices:NewField('posmag', L.double)

------------------
-- Should pass: --
------------------
local populate_tensor = ebb (v : mesh.vertices)
  var p = v.pos
  var tensor = {
    { p[0]*p[0], p[0]*p[1], p[0]*p[2] },
    { p[1]*p[0], p[1]*p[1], p[1]*p[2] },
    { p[2]*p[0], p[2]*p[1], p[2]*p[2] }
  }
  v.tensor_pos = tensor
end
mesh.vertices:foreach(populate_tensor)

local trace = ebb (v : mesh.vertices)
  var tp = v.tensor_pos
  v.posmag = tp[0,0]*tp[0,0] + tp[1,1]*tp[1,1] + tp[2,2]*tp[2,2]
end
mesh.vertices:foreach(trace)

local k = ebb (v : mesh.vertices)
  var x       = { { 5, 5, 5 }, { 4, 4, 4 }, { 5, 5, 5 } }
  v.tensor_pos += x + { { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 } }
end
mesh.vertices:foreach(k)

local zero_corner = ebb(v : mesh.vertices)
  v.tensor_pos[0,2] = 0
end
mesh.vertices:foreach(zero_corner)

local arith_test = ebb(v : mesh.vertices)
  var id : L.mat3d = {{1,0,0},{0,1,0},{0,0,1}}
  var A  : L.mat3d = {{2,2,2},{3,3,3},{4,4,4}}
  var B  : L.mat3d = {{7,2,2},{3,8,3},{4,4,9}}
  L.assert((A + 5*id) == B)
end
mesh.vertices:foreach(arith_test)

local unsym_mat = ebb(v : mesh.vertices)
  var A : L.mat3x4i = { { 1, 2, 3, 4 }, { 5, 6, 7, 8 }, { 9, 10, 11, 12 } }
  A
end
mesh.vertices:foreach(unsym_mat)

------------------
-- Should fail: --
------------------

test.fail_function(function()
  local ebb t(v : mesh.vertices)
    var m34 = L.mat3x4f({{1.1,0,2.3},{0.1,0,0},{0,5.2,0}})
    var m33 = L.mat3f(m34)
  end
  mesh.vertices:foreach(t)
end,
'Cannot cast between primitives, vectors, matrices of different dimensions')

test.fail_function(function()
  local ebb t(v : mesh.vertices)
    var x = {{1, 2, 3},{true,false,true},{1,2,5}}
  end
  mesh.vertices:foreach(t)
end, "must be of the same type")

test.fail_function(function()
  local ebb t(v : mesh.vertices)
    var x = {{1,2}, {2,3}, {2,3,4}}
  end
  mesh.vertices:foreach(t)
end, "matrix literals must contain vectors of the same size")

idx = 3.5
test.fail_function(function()
  local ebb t(v : mesh.vertices)
      v.tensor_pos[idx,2] = 5
  end
  mesh.vertices:foreach(t)
end, "expected an integer")

test.fail_function(function()
  local ebb t(v : mesh.vertices)
      v.tensor_pos[0] = 5
  end
  mesh.vertices:foreach(t)
end, "expected vector to index into, not Matrix")

-- Parse error, so not safe to test this way
--test.fail_function(function()
--  local ebb t(v : mesh.vertices)
--      v.tensor_pos[0,0,1] = 5
--  end
--  mesh.vertices:foreach(t)
--end, "expected 2 indices")




