--DISABLE-TEST
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

local R = L.NewRelation { name="R", size=5 }

--[[
nf = mesh.faces:Size()
sf = L.Global(L.float, 0.0)
si = L.Global(L.int,     0)
sb = L.Global(L.bool, true)
sd = L.Global(L.double, 1.0)

sf3 = L.Global(L.vector(L.float, 3), {0, 0, 0})
si4 = L.Global(L.vector(L.int,   4), {1, 2, 3, 4})
sb5 = L.Global(L.vector(L.bool,  5), {true, true, true, true, true})

vf  = L.Constant(L.vec3f, {1, 2, 3})
vi  = L.Constant(L.vec4i,   {2, 2, 2, 2})
vb  = L.Constant(L.vector(L.bool, 5),  {true, false, true, false, true})

local two = 2

]]--