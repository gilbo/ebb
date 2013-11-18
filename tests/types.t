--------------------------------------------------------------------------------
--[[ Test implementation of type system for consistency                     ]]--
--------------------------------------------------------------------------------
local T = terralib.require "compiler.types"
local L = terralib.require "compiler.liszt"
require "tests.test"

local ptypes = { [int]   = L.int, 
	             [float] = L.float, 
	             [bool]  = L.bool   }

for ttype, ltype in pairs(ptypes) do
	--------------------------------
	--[[ Test primitives types: ]]--
	--------------------------------
	assert(ltype:isPrimitive())

	-- verify runtime type enum conversions
	assert(ltype:terraType() == ttype)

	assert(T.terraToLisztType(ttype) == ltype)
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

		assert(vtype:isVector())

		assert(vtype:baseType()      == ltype)
		assert(vtype:terraBaseType() == ttype)
		assert(vtype:terraType()     == vector(ttype,i))
		assert(T.terraToLisztType(vector(ttype, i)) == vtype)

		-- cannot call type constructor with a terra type
		local function fn()
			local tp = L.vector(ttype,i)
		end
		test.fail_function(fn, 'invalid type')

		-- user-exposed type constructor should return the correct liszt type
		assert(L.vector(ltype,i) == T.t.vector(ltype,i))
	end
end
