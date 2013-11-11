local JSON = require('lib/JSON')

local DECL = terralib.require('include/decl')

local exports = {}

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
Type.kinds.macro     = {string='macro'     }
Type.kinds.quoteexpr = {string='quoteexpr' }
Type.kinds.scalar    = {string='scalar'    }
Type.kinds.table     = {string='table'     }
Type.kinds.error     = {string='error'     }
Type.kinds.relation  = {string='relation'  }
Type.kinds.reference = {string='reference' }

Type.kinds.int    = {string='int',    terratype=int   }
Type.kinds.float  = {string='float',  terratype=float }
Type.kinds.bool   = {string='bool',   terratype=bool  }
Type.kinds.uint   = {string='uint',   terratype=uint  }
Type.kinds.uint8  = {string='uint8',  terratype=uint8 }
Type.kinds.double = {string='double', terratype=double}

function Type:new (kind,typ,scope)
  return setmetatable({kind=kind,type=typ,scope=scope}, self)
end

function Type.isLisztType (obj)
  return getmetatable(obj) == Type
end

-------------------------------------------------------------------------------
--[[ These methods can only be called on liszt types                       ]]--
-------------------------------------------------------------------------------
function Type:isPrimitive()
  return type(self) == 'table' and self.kind == Type.kinds.primitive
end
function Type:isVector()
  return type(self) == 'table' and self.kind == Type.kinds.vector
end

-- of type integer or vectors of integers
function Type:isIntegral ()
  return (self.kind == Type.kinds.primitive and
          self.type == Type.kinds.int)
      or (self.kind == Type.kinds.vector and self.type:isIntegral())
end

-- ints, floats, or vectors of either

local numeric_kinds = {
  [Type.kinds.int]    = true,
  [Type.kinds.float]  = true,
  [Type.kinds.uint]   = true,
  [Type.kinds.uint8]  = true,
  [Type.kinds.double] = true,
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

function Type:isField()     return self.kind == Type.kinds.field     end
function Type:isScalar()    return self.kind == Type.kinds.scalar    end
function Type:isFunction()  return self.kind == Type.kinds.functype  end
function Type:isMacro()     return self.kind == Type.kinds.macro     end
function Type:isQuoteExpr() return self.kind == Type.kinds.quoteexpr end
function Type:isLuaTable()  return self.kind == Type.kinds.table     end
function Type:isError()     return self.kind == Type.kinds.error     end
function Type:isRelation()  return self.kind == Type.kinds.relation  end
function Type:isTable()     return self.kind == Type.kinds.table     end
function Type:isRef()       return self.kind == Type.kinds.reference end


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
  elseif self:isRef()       then return uint32
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
  if     self:isPrimitive() then return self.type.string
  elseif self:isVector()    then return 'LVector(' .. self.type:toString() .. ',' .. tostring(self.N)     .. ')'
  elseif self:isScalar()    then return 'LScalar(' .. self.type:toString() .. ')'
  elseif self:isFunction()  then return 'LFunction'
    elseif self:isMacro()     then return 'LMacro'
    elseif self:isQuoteExpr() then return 'LQuoteExpr'
  elseif self:isRelation()  then return 'LRelation'
    elseif self:isField()     then return 'LField'
  elseif self:isLuaTable()  then return 'table'
  elseif self:isError()     then return 'error'
  elseif self:isRef()       then return 'Ref('..self.relation:name()..')'
  end
  print(debug.traceback())
  error('toString method not implemented for this type!', 2)
end


-- THIS DOES NOT EMIT A STRING
-- It outputs a LUA table which can be JSON stringified safely
function Type:json_serialize()
  local json = {}
  return json
end
-- given a table output by json_serialize, deserialize reconstructs
-- a proper Type object with correct metatable, etc.
function Type.json_deserialize(json_tbl)
end


-------------------------------------------------------------------------------
--[[ Type constructors:                                                    ]]--
-------------------------------------------------------------------------------
-- cache complex type objects for re-use
local complexTypes = {}

local function vectorType (typ, len)
  if not Type.isLisztType(typ) or not typ:isPrimitive() then error("invalid type argument to vector type constructor (is this a terra type?)", 2) end
  local tpn = 'vector(' .. typ:toString() .. ',' .. tostring(len) .. ')'
  if not complexTypes[tpn] then
    local vt = Type:new(Type.kinds.vector,typ)
    vt.N = len
    complexTypes[tpn] = vt
  end
  return complexTypes[tpn]
end

local refTypes = {}

local function referenceType (relation)
  if not DECL.is_relation(relation) then
    error("invalid argument to reference type constructor. A relation "..
          "must be provided", 2)
  end
  if not refTypes[relation:name()] then
    local rt = Type:new(Type.kinds.reference)
    rt.relation = relation
    refTypes[relation:name()] = rt
  end
  return refTypes[relation:name()]
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
t.double    = Type:new(Type.kinds.primitive,Type.kinds.double)

-- Complex type constructors
t.vector    = vectorType
t.ref       = referenceType

t.macro     = Type:new(Type.kinds.macro)

-- These types are for ast nodes that can show up in expressions,
-- but are not valid expression types.  We keep track of their type
-- so that we can report type errors to the user.
-- HEY QUESTION: these shouldn't be exposed to the user then?
t.scalar    = Type:new(Type.kinds.scalar)
t.field     = Type:new(Type.kinds.field)
t.func      = Type:new(Type.kinds.functype) -- builtin function type
t.quoteexpr = Type:new(Type.kinds.quoteexpr)  -- produced at Lua scope by "liszt `[expr]"
t.table     = Type:new(Type.kinds.table)  -- lua tables
t.relation  = Type:new(Type.kinds.relation)


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

  elseif ltype == t.double or rtype == t.double then return t.double
  elseif ltype == t.float  or rtype == t.float  then return t.float
  else return t.int
  end
end

local function conformsToType (inst, tp)
  if     tp:isNumeric() and tp:isPrimitive() then
    return type(inst) == 'number'
  elseif tp:isLogical() and tp:isPrimitive() then
    return type(inst) == 'boolean'
  elseif tp:isVector()  then
    -- accept an instance of an LVector:
    if Type.isVector(inst) then 
      -- bool vectors can only be initialized with other bool vectors
      if tp:baseType():isLogical() then
        return inst.type:baseType():isLogical()
      end

      -- but any numeric vector can be initialized
      -- with another numeric vector!
      return inst.type:baseType():isNumeric()

    -- we also accept arrays as instances of vectors
    elseif type(inst) == 'table' then
      -- make sure array is of the correct length
      if #inst ~= tp.N then return false end
      -- make sure each element conforms to the vector data type
      for i = 1, #inst do
        if not conformsToType(inst[i], tp:baseType()) then return false end
      end
      return true
    end
  end
  return false
end

local ttol = {
  [int]    = t.int,
  [float]  = t.float,
  [double] = t.double,
  [bool]   = t.bool,
  [uint]   = t.uint,
  [uint8]  = t.uint8
}


-- converts a terra vector or primitive type into a liszt type
local function terra_to_liszt (tp)
  -- return primitive type
  if ttol[tp] then return ttol[tp] end

  -- return vector type
  if tp:isvector() then
    local p = terra_to_liszt(tp.type)
    if p == nil then return nil end
    return t.vector(p,tp.N)
  end
  return nil
end


-------------------------------------------------------------------------------
--[[ Return exports                                                        ]]--
-------------------------------------------------------------------------------
exports.type_meet      = type_meet
exports.conformsToType = conformsToType
exports.terraToLisztType = terra_to_liszt
return exports

