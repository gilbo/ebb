--------------------------------------------------------------------------------
--[[ Test implementation of type system for consistency                     ]]--
--------------------------------------------------------------------------------
local types = terralib.require("compiler/types")
terralib.require("runtime/liszt")
terralib.require("include/liszt")
require 'tests/test'
local t     = types.t
local tutil = types.usertypes

local ptypes = {[int] = t.int,         [float] = t.float,         [bool] = t.bool}
local rtypes = {[int] = runtime.L_INT, [float] = runtime.L_FLOAT, [bool] = runtime.L_BOOL}

local ptopo  = {[t.vertex] = Vertex,           [t.edge] = Edge,           
	            [t.face]   = Face,             [t.cell] = Cell}
local rtopo  = {[t.vertex] = runtime.L_VERTEX, [t.edge] = runtime.L_EDGE,
                [t.face]   = runtime.L_FACE,   [t.cell] = runtime.L_CELL}

for ttype, ltype in pairs(ptypes) do
	--------------------------------
	--[[ Test primitives types: ]]--
	--------------------------------
	assert(ltype:isPrimitive())

	-- verify runtime type enum conversions
	assert(ltype:terraType() == ttype)
	local rt, ln = ltype:runtimeType()
	assert(rt == rtypes[ttype])
	assert(ln == 1)

	-- make sure ltype converts terra type correctly
	assert(tutil.ltype(ttype) == ltype)

	--------------------------------
	--[[ Test vector types:     ]]--
	--------------------------------
	local accum = {}
	for i = 2, 10 do
		local vtype  = t.vector(ltype,i)
		local vtype2 = t.vector(ltype,i)
		assert(vtype == vtype2)

		-- each of these vector types should be unique from all previously generated types!
		assert(accum[vtype] == nil)
		accum[vtype] = true

		assert(vtype:isVector())

		assert(vtype:baseType()      == ltype)
		assert(vtype:terraBaseType() == ttype)
		assert(vtype:terraType()     == vector(ttype,i))

		-- cannot call type constructor with a terra type
		local function fn()
			local tp = t.vector(ttype,i)
		end
		test.fail_function(fn, 'invalid type')

		-- verify returned runtime type enum, length
		local rt, ln = vtype:runtimeType()
		assert(rt == rtypes[ttype])
		assert(ln == i)

		-- user-exposed type constructor should return the correct liszt type
		assert(L.vector(ltype,i) == t.vector(ltype,i))
	end
end

--------------------------------
--[[ Test topo types:       ]]--
--------------------------------
for ltype, rtp in pairs(rtopo) do
	assert(ltype:isTopo())
	assert(ltype:runtimeType() == rtp)
	assert(tutil.ltype(ptopo[ltype]) == ltype)
end

--------------------------------
--[[ Test scalar types:     ]]--
--------------------------------
local accum = {}
for ttype, ltype in pairs(ptypes) do
	local sc  = t.scalar(ltype)
	local sc2 = t.scalar(ltype)
	assert(sc == sc2)
	assert(accum[sc] == nil)
	accum[sc] = true

	for i = 2, 10 do
		local vtype = t.vector(ltype,i)
		local sc    = t.scalar(vtype)
		local sc2   = t.scalar(vtype)

		assert(sc == sc2)
		assert(accum[sc] == nil)
		accum[sc] = true
		assert(sc:terraType()     == vtype:terraType())
		assert(sc:terraBaseType() == ttype)
		assert(sc:dataType()      == vtype)
	end
end

--------------------------------
--[[ Test field types:      ]]--
--------------------------------
local accum = {}
for ttype, ltype in pairs(ptypes) do
	for ltopo, rtype in pairs(rtopo) do
		local f  = t.field(ltopo, ltype)
		local f2 = t.field(ltopo, ltype)
		assert(f == f2)
		assert(accum[f] == nil)
		accum[f] = true

		assert(f:dataType() == ltype)
		assert(f:topoType() == ltopo)

		for i = 2, 10 do
			local vtype = t.vector(ltype,i)
			local f     = t.field(ltopo, vtype)
			local f2    = t.field(ltopo, vtype)

			assert(f == f2)
			assert(accum[f] == nil)
			accum[f] = true

			assert(f:dataType() == vtype)
			assert(f:topoType() == ltopo)
		end
	end
end

