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

local ioOff = require 'ebb.domains.ioOff'
local mesh  = ioOff.LoadTrimesh('tests/octa.off')
local com   = L.Global(L.vector(L.float, 3), {0, 0, 0})

function center_of_mass ()
	com:set({0,0,0})
	local sum_pos = ebb (v : mesh.vertices)
		com += L.vec3f(v.pos)
	end
	mesh.vertices:foreach(sum_pos)
	local sz = mesh.vertices:Size()
	local c  = com:get()
	return { c[1]/sz, c[2]/sz, c[3]/sz }
end

local function displace_mesh (delta_x, delta_y, delta_z)
	local d = L.Constant(L.vec3d, {delta_x, delta_y, delta_z})
	local dk = ebb (v : mesh.vertices)
		v.pos += d
	end
	mesh.vertices:foreach(dk)
end

test.aeq(center_of_mass(), {0, 0, 0})

displace_mesh(1, 0, 0)
test.aeq(center_of_mass(), {1, 0, 0})

displace_mesh(.4, .5, .8)
test.fuzzy_aeq(center_of_mass(), {1.4, .5, .8})

displace_mesh(-3, -4, -5)
test.fuzzy_aeq(center_of_mass(), {-1.6, -3.5, -4.2})

