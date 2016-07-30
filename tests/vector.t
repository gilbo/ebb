--DISABLE-DISTRIBUTED
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
package.path = package.path .. ";./tests/?.lua;?.lua"
local test = require "test"
import 'ebb'
local L = require 'ebblib'
local types = require "ebb.src.types"


local a  = L.Constant(L.vec3f, {1,     2, 3.29})
local z  = L.Constant(L.vec3f, {4, 5.392,    6})

local a4 = L.Constant(L.vec4f, {3.4, 4.3, 5, 6.153})
local ai = L.Constant(L.vec3i, {2, 3, 4})
local ab = L.Constant(L.vec3b, {true, false, true})

local ioOff = require 'ebb.domains.ioOff'
local mesh  = ioOff.LoadTrimesh('tests/octa.off')

------------------
-- Should pass: --
------------------
local k = ebb (v : mesh.vertices)
	var x       = {5, 5, 5}
	v.pos += x + {0, 1, 1}
end
mesh.vertices:foreach(k)

--[[
-- Additive reduction over doubles currently unsupported
local s = L.Global(L.vector(L.double, 3), {0.0, 0.0, 0.0})
local sum_pos = ebb(v : mesh.vertices)
	s += v.pos
end
mesh.vertices:foreach(sum_pos)

local f = s:get() / mesh.vertices:Size()
test.fuzzy_aeq(f.data, {5, 6, 6})
]]

------------------
-- Should fail: --
------------------

test.fail_function(function()
	local ebb t(v : mesh.vertices)
		var v3 = L.vec3f({1.1, 2.2, 3.3})
		var v2 = L.vec2f(v3)
	end
  mesh.vertices:foreach(t)
end,
'Cannot cast between primitives, vectors, matrices of different dimensions')


