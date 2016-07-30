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
import 'ebb'
local L = require 'ebblib'
require "tests/test"



local triangles = L.NewRelation { size = 1, name = 'triangles' }
local vertices  = L.NewRelation { size = 3, name = 'vertices' }

triangles:NewField('v0', vertices)
triangles:NewField('v1', vertices)
triangles:NewField('v2', vertices)

vertices:NewField('pos', L.vector(L.float, 4))
vertices:NewField('color', L.vector(L.float, 3))

test.eq(tostring(triangles:StructuralType()),
  'Record({ v0=Key(vertices), v1=Key(vertices), v2=Key(vertices) })')
test.eq(tostring(vertices:StructuralType()),
  'Record({ color=Vector(float,3), pos=Vector(float,4) })')

