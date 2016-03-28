--DISABLE-PARTITIONED
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

local R = L.NewRelation  { size = 32, name = 'R' }
R:NewField('sqrt', L.double):Load(0.0)
R:NewField('cbrt', L.double):Load(0.0)
R:NewField('sin',  L.double):Load(0.0)
R:NewField('cos',  L.double):Load(0.0)

local root_test = ebb (r : R)
	r.cbrt = L.cbrt(L.id(r))
	r.sqrt = L.sqrt(L.id(r))
end
R:foreach(root_test)

R.cbrt:Print()
R.sqrt:Print()

local trig_test = ebb (r : R)
	r.sin = L.sin(L.id(r))
	r.cos = L.cos(L.id(r))
end
R:foreach(trig_test)

R.sin:Print()
R.cos:Print()

print()
for i = 0, 32 do
	print(L.cbrt(i))
end
print()
for i = 0, 32 do
	print(L.sqrt(i))
end
print()
for i = 0, 32 do
	print(L.sin(i))
end
print()
for i = 0, 32 do
	print(L.cos(i))
end
