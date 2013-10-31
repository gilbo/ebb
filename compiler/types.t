local exports = {}
terralib.require('runtime/liszt')

-------------------------------------------------------------------------------
--[[ Liszt type prototype:                                                 ]]--
-------------------------------------------------------------------------------
local Type   = {}
exports.Type = Type
Type.__index = Type
Type.kinds   = {}
Type.kinds.primitive = {string='primitive' }
Type.kinds.vector    = {string='vector'    }
Type.kinds.field     = {string='field'     }
Type.kinds.functype  = {string='functype'  }
Type.kinds.scalar    = {string='scalar'    }
Type.kinds.table     = {string='table'     }
Type.kinds.error     = {string='error'     }

local Scope   = {}
exports.Scope = Scope
Scope.lua     = {}
Scope.liszt   = {}

Type.kinds.int   = {string='int',   terratype=int,   runtimetype=runtime.L_INT}
Type.kinds.float = {string='float', terratype=float, runtimetype=runtime.L_FLOAT}
Type.kinds.bool  = {string='bool',  terratype=bool,  runtimetype=runtime.L_BOOL}
Type.kinds.uint  = {string='uint',  terratype=uint}
Type.kinds.uint8 = {string='uint8', terratype=uint8}

Type.validKinds = {
	[Type.kinds.primitive] = true,
	[Type.kinds.vector]    = true,
	[Type.kinds.field]     = true,
	[Type.kinds.functype]  = true,
	[Type.kinds.scalar]    = true,
	[Type.kinds.error]     = true
}

function Type:new (kind,typ,scope)
	return setmetatable({kind=kind,type=typ,scope=scope}, self)
end

function Type.isLisztType (obj)
	return getmetatable(obj) == Type
end

-------------------------------------------------------------------------------
--[[ These methods can only be called on liszt types                       ]]--
-------------------------------------------------------------------------------
function Type:isPrimitive() return type(self) == 'table' and self.kind == Type.kinds.primitive end
function Type:isVector()    return type(self) == 'table' and self.kind == Type.kinds.vector    end

-- of type integer or vectors of integers
function Type:isIntegral ()
  return (self.kind == Type.kinds.primitive and
          self.type == Type.kinds.int)
      or (self.kind == Type.kinds.vector and self.type:isIntegral())
end

-- ints, floats, or vectors of either

local numeric_kinds = {
	[Type.kinds.int]   = true,
	[Type.kinds.float] = true,
	[Type.kinds.uint]  = true,
	[Type.kinds.uint8] = true,
}

function Type:isNumeric  ()
  return (self.kind == Type.kinds.primitive and numeric_kinds[self.type])
      or (self.kind == Type.kinds.vector and self.type:isNumeric())
end

-- bools or vectors of bools
function Type:isLogical  ()
  return (self.kind == Type.kinds.primitive and self.type == Type.kinds.bool)
      or (self.kind == Type.kinds.vector    and self.type:isLogical())
end

-- any primitive or vector
function Type:isExpressionType() return self:isPrimitive() or self:isVector() end


-------------------------------------------------------------------------------
--[[ These methods can be called on liszt types or liszt objects           ]]--
-------------------------------------------------------------------------------
function Type:isField()        return type(self) == 'table' and self.kind == Type.kinds.field       end
function Type:isScalar()       return type(self) == 'table' and self.kind == Type.kinds.scalar      end
function Type:isFunction()     return type(self) == 'table' and self.kind == Type.kinds.functype    end
function Type:isLuaTable()     return type(self) == 'table' and self.kind == Type.kinds.table       end
function Type:isError()        return type(self) == 'table' and self.kind == Type.kinds.error       end
function Type:isRelation()     return type(self) == 'table' and self.kind == Type.kinds.relation    end
function Type:isRelationRow()  return type(self) == 'table' and self.kind == Type.kinds.relationrow end

-- currently isTable will return true if obj has kind fieldindex
function Type.isTable(obj)   return type(obj)  == 'table' and not Type.validKinds[obj.kind]      end


-------------------------------------------------------------------------------
--[[ Methods for computing terra or runtime types                          ]]--
-------------------------------------------------------------------------------
function Type:baseType()
	if self:isVector()    then return self.type end
	if self:isPrimitive() then return self      end
	error("baseType not implemented for " .. self:toString(),2)
end

function Type:terraType()
	if     self:isPrimitive() then return self.type.terratype
	elseif self:isVector()    then return vector(self.type:terraType(), self.N)
	elseif self:isField()     then return self.type:terraType()
	elseif self:isScalar()    then return self.type:terraType()
	end
	error("terraType method not implemented for type " .. self:toString(), 2)
end

function Type:terraBaseType()
	if     self:isPrimitive() then return self:terraType()
	elseif self:isVector()    then return self.type:terraType()
	elseif self:isField()     then return self.type:terraBaseType()
	elseif self:isScalar()    then return self.type:terraBaseType()
	end
	error("terraBaseType method not implemented for type " .. self:toString(), 2)
end

function Type:dataType()
	if self:isScalar() then return self.type end
	error("dataType not implemented for type " .. self:toString(),2)
end


-------------------------------------------------------------------------------
--[[ Stringify types                                                       ]]--
-------------------------------------------------------------------------------
function Type:toString()
	if     self:isPrimitive()   then return self.type.string
	elseif self:isVector()      then return 'LVector(' .. self.type:toString() .. ',' .. tostring(self.N)     .. ')'
	elseif self:isScalar()      then return 'LScalar(' .. self.type:toString() .. ')'
	elseif self:isFunction()    then return 'LFunction'
	elseif self:isRelation()    then return 'LRelation'
	elseif self:isRelationRow() then return 'LRelationRow'
    elseif self:isField()       then return 'LField'
	elseif self:isLuaTable()    then return "table"
	elseif self:isError()       then return "error"
	end
	error('toString method not implemented for this type!', 2)
end


-------------------------------------------------------------------------------
--[[ Type constructors:                                                    ]]--
-------------------------------------------------------------------------------
-- cache complex type objects for re-use
local complexTypes = {}

local function vectorType (typ, len)
	if not Type.isLisztType(typ) then error("invalid type argument to vector type constructor (is this a terra type?)", 2) end
	local tpn = 'vector(' .. typ:toString() .. ',' .. tostring(len) .. ')'
	if not complexTypes[tpn] then
		local vt = Type:new(Type.kinds.vector,typ)
		vt.N = len
		complexTypes[tpn] = vt
	end
	return complexTypes[tpn]
end

local function scalarType(typ)
	local tpn = 'scalar(' .. typ:toString() .. ')'
	local sct = complexTypes[tpn]
	if not sct then
		sct = Type:new(Type.kinds.scalar,typ)
		complexTypes[tpn] = sct
	end
	return sct
end


-------------------------------------------------------------------------------
--[[ Type interface:                                                       ]]--
-------------------------------------------------------------------------------
local t   = {}
exports.t = t
t.error   = Type:new(Type.kinds.error)

-- Primitives
t.int       = Type:new(Type.kinds.primitive,Type.kinds.int)
t.uint      = Type:new(Type.kinds.primitive,Type.kinds.uint)
t.uint8     = Type:new(Type.kinds.primitive,Type.kinds.uint8)
t.float     = Type:new(Type.kinds.primitive,Type.kinds.float)
t.bool      = Type:new(Type.kinds.primitive,Type.kinds.bool)

-- un-inferred type
t.unknown   = Type:new()

-- Complex type constructors
t.vector    = vectorType
t.scalar    = scalarType

-- Type for macros
t.func = Type:new(Type.kinds.functype)

-- Support for table lookups / select operator
t.table  = Type:new(Type.kinds.table)

-- Relations
t.relation    = Type:new(Type.kinds.relation)
t.relationrow = Type:new(Type.kinds.relationrow)


-------------------------------------------------------------------------------
--[[ Utilities for converting user-exposed terra types to Liszt types      ]]--
-------------------------------------------------------------------------------
local usertypes = {}
exports.usertypes = usertypes
local terraToPrimitiveType = {
   [int]   = t.int,
   [float] = t.float,
   [bool]  = t.bool
}

-- can be a vector base type
function usertypes.isPrimitiveType (tp) 
	return terraToPrimitiveType[tp] ~= nil       
end

-- terra vector type
function usertypes.isTVectorType (tp)
	return usertypes.isPrimitiveType(tp.type) and usertypes.isSize(tp.N)
end

-- true for either a liszt or terra vector type (e.g. vector(float, 4) or t.vector(float, 4))
function usertypes.isVectorType(tp)
	return usertypes.isTVectorType(tp) or Type.isVector(tp)
end

-- can be a scalar base type, or an acceptable expression type in a liszt kernel, generally
function usertypes.isDataType (tp)
	return usertypes.isPrimitiveType(tp) or Type.isVector(tp)
end

function usertypes.isSize (elem)
   return type(elem) == 'number' and elem > 0 and elem % 1 == 0
end

function usertypes.conformsToType (inst, tp)
   if     tp:isNumeric() and tp:isPrimitive() then return type(inst) == 'number'
   elseif tp:isLogical() and tp:isPrimitive() then return type(inst) == 'boolean'

   elseif tp:isVector()  then
      -- accept an instance of an LVector:
      if Vector.isVector(inst) then 
         -- bool vectors can only be initialized with other bool vectors
         if tp:baseType():isLogical() then return inst.type:baseType():isLogical() end

         -- but any numeric vector can be initialized with another numeric vector!
         return inst.type:baseType():isNumeric()

      -- we also accept arrays as instances of vectors
      elseif type(inst) == 'table' then
         -- make sure array is of the correct length
         if #inst ~= tp.N then return false end
         -- make sure each element conforms to the vector data type
         for i = 1, #inst do
            if not usertypes.conformsToType(inst[i], tp:baseType()) then return false end
         end
         return true
      end
   end
   return false
end

function usertypes.ltype (dt)
  	if usertypes.isPrimitiveType(dt) then return terraToPrimitiveType[dt] end
   	if Type.isVector(dt)             then return dt end -- vector type is already a liszt type
   	-- also accept and convert terra vector types to liszt vector types
   	if usertypes.isTVectorType(dt)   then return t.vector(usertypes.ltype(dt.type), dt.N) end
   	return t.error
end


-------------------------------------------------------------------------------
--[[ Type meeting                                                          ]]--
-------------------------------------------------------------------------------
local function type_meet(ltype, rtype)
	if ltype:isVector() and rtype:isVector() and ltype.N ~= rtype.N then 
		return t.error

	elseif ltype:isVector() or rtype:isVector() then
		local btype = type_meet(ltype:baseType(),rtype:baseType())
		if btype == t.error then return t.error end
		return ltype:isVector() and t.vector(btype,ltype.N) or t.vector(btype,rtype.N)

	-- both ltype and rtype are primitives, make sure they are either both numeric or both logical
	elseif ltype:isNumeric() ~= rtype:isNumeric() then
		return t.error

	elseif ltype:isLogical() then
		return t.bool

	elseif ltype == t.float or rtype == t.float then
		return t.float

	else
		return t.int
	end
end
exports.type_meet = type_meet


return exports

