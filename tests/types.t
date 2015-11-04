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
