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

--------------------------------------------------------------------------------
--[[ Test implementation of type system for consistency                     ]]--
--------------------------------------------------------------------------------
local L = require "ebb"
local T = require "ebb.src.types"
require "tests.test"

local ptypes = { [int]   = L.int, 
	             [float] = L.float, 
	             [bool]  = L.bool   }

for ttype, ltype in pairs(ptypes) do
	--------------------------------
	--[[ Test primitives types: ]]--
	--------------------------------
	assert(ltype:isprimitive())

	-- verify runtime type enum conversions
	assert(ltype:terratype() == ttype)

	assert(T.terraToEbbType(ttype) == ltype)
	--------------------------------
	--[[ Test vector types:     ]]--
	--------------------------------
	local accum = {}
	for i = 2, 10 do
		local vtype  = T.t.vector(ltype,i)
		local vtype2 = T.t.vector(ltype,i)
		assert(vtype == vtype2)

		-- each of these vector types should be unique from all previously generated types!
		assert(accum[vtype] == nil)
		accum[vtype] = true

		assert(vtype:isvector())

		assert(vtype:basetype()      == ltype)
		assert(vtype:terrabasetype() == ttype)
		assert(vtype:terratype()     == vector(ttype,i))
		assert(T.terraToEbbType(vector(ttype, i)) == vtype)

		-- cannot call type constructor with a terra type
		local function fn()
			local tp = L.vector(ttype,i)
		end
		test.fail_function(fn, 'invalid type')

		-- user-exposed type constructor should return the correct ebb type
		assert(L.vector(ltype,i) == T.t.vector(ltype,i))
	end
end
