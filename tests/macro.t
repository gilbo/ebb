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
local test = require('tests.test')

local R = L.NewRelation { size = 4, name = 'relation' }
R:NewField("result", L. uint64)

local G = L.Global(L.double, 0)

-- This macro emits a side effect every time it is evaluated
local side_effect = L.Macro(function(x)
	return ebb quote
  G += L.double(L.id(x))
	in
		x
	end
end)

local some_macro = L.Macro(function (y)
	return ebb `L.id(y)+ L.id(y)
end)

local ebb per_elem (r : R)
	--side effect should be evaluated twice!
	r.result = some_macro(side_effect(r))

end

R:foreach(per_elem)

test.eq(G:get(), 12)


