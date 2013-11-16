local JSON = require('lib/JSON')

local DECL = terralib.require('include/decl')

local exports = {}

-- forward declaration of the type interface
local t   = {}
exports.t = t

-------------------------------------------------------------------------------
--[[ Liszt type prototype:                                                 ]]--
-------------------------------------------------------------------------------
local Type   = {}
exports.Type = Type
Type.__index = Type
Type.kinds   = {}
-- There are four basic kinds of types, plus a fifth: ERROR
Type.kinds.primitive    = {string='primitive' }
  Type.kinds.int        = {string='int',    terratype=int   }
  --Type.kinds.int8       = {string='int8',   terratype=int8  }
  --Type.kinds.int16      = {string='int16',  terratype=int16 }
  --Type.kinds.int32      = {string='int32',  terratype=int32 }
  --Type.kinds.int64      = {string='int64',  terratype=int64 }
  Type.kinds.uint       = {string='uint',   terratype=uint  }
  Type.kinds.uint8      = {string='uint8',  terratype=uint8 }
  --Type.kinds.uint16     = {string='uint16', terratype=uint16}
  --Type.kinds.uint32     = {string='uint32', terratype=uint32}
  Type.kinds.uint64     = {string='uint64', terratype=uint64}
  Type.kinds.bool       = {string='bool',   terratype=bool  }
  Type.kinds.float      = {string='float',  terratype=float }
  Type.kinds.double     = {string='double', terratype=double}
Type.kinds.vector       = {string='vector'    }
Type.kinds.row          = {string='row'       }
-- Internal types are singletons and should never face the user
Type.kinds.internal     = {string='internal'  }
  -- an Internal type's TYPE field should resolve to one of the
  -- following kinds, and the internal type object may hold
  -- a reference to the relevant object when appropriate
  Type.kinds.relation   = {string='relation'  }
  Type.kinds.field      = {string='field'     }
  Type.kinds.scalar     = {string='scalar'    }
  Type.kinds.macro      = {string='macro'     }
  Type.kinds.functype   = {string='functype'  }
  Type.kinds.quoteexpr  = {string='quoteexpr' }
  Type.kinds.luatable   = {string='luatable'  }
Type.kinds.error        = {string='error'     }



function Type:new (kind,typ,scope)
  return setmetatable({kind=kind,type=typ,scope=scope}, self)
end

function Type.isLisztType (obj)
  return getmetatable(obj) == Type
end

-------------------------------------------------------------------------------
--[[ Basic Type Methods                                                    ]]--
-------------------------------------------------------------------------------
function Type:isPrimitive()
  return self.kind == Type.kinds.primitive
end
function Type:isVector()
  return self.kind == Type.kinds.vector
end
function Type:isRow()
  return self.kind == Type.kinds.row
end
function Type:isInternal()
  return self.kind == Type.kinds.internal
end
function Type:isError()
  return self.kind == Type.kinds.error
end

-- Can this type of AST represent a Liszt value
function Type:isValueType()
  return self:isPrimitive() or self:isVector()
end
-- eventually this type should support rows...

-------------------------------------------------------------------------------
--[[ Primitive/Vector Type Methods                                         ]]--
-------------------------------------------------------------------------------

local integral_kinds = {
  [Type.kinds.int]    = true,
  --[Type.kinds.int8]   = true,
  --[Type.kinds.int16]  = true,
  --[Type.kinds.int32]  = true,
  --[Type.kinds.int64]  = true,
  [Type.kinds.uint]   = true,
  [Type.kinds.uint8]  = true,
  --[Type.kinds.uint16] = true,
  --[Type.kinds.uint32] = true,
  [Type.kinds.uint64] = true,
}

local numeric_kinds = {
  [Type.kinds.float]  = true,
  [Type.kinds.double] = true,
}
for k,v in pairs(integral_kinds) do numeric_kinds[k] = v end

-- of type integer or vectors of integers
function Type:isIntegral ()
  if   self:isPrimitive() then return integral_kinds[self.type]
  elseif  self:isVector() then return self.type:isIntegral()
  else                         return false end
end

function Type:isNumeric ()
  if   self:isPrimitive() then return numeric_kinds[self.type]
  elseif  self:isVector() then return self.type:isNumeric()
  else                         return false end
end

function Type:isLogical ()
  if   self:isPrimitive() then return self.type == Type.kinds.bool
  elseif  self:isVector() then return self.type:isLogical()
  else                         return false end
end

-------------------------------------------------------------------------------
--[[ Internal Type Methods                                                 ]]--
-------------------------------------------------------------------------------

local function in_check(typ, kind)
  return typ:isInternal() and typ.type == kind
end

function Type:isField()     return in_check(self, Type.kinds.field)     end
function Type:isScalar()    return in_check(self, Type.kinds.scalar)    end
function Type:isFunction()  return in_check(self, Type.kinds.functype)  end
function Type:isMacro()     return in_check(self, Type.kinds.macro)     end
function Type:isQuoteExpr() return in_check(self, Type.kinds.quoteexpr) end
function Type:isLuaTable()  return in_check(self, Type.kinds.luatable)  end
function Type:isRelation()  return in_check(self, Type.kinds.relation)  end


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
  elseif self:isRow()       then return t.addr:terraType()
  end
  error("terraType method not implemented for type " .. self:toString(), 2)
end

function Type:terraBaseType()
  if     self:isPrimitive() then return self:terraType()
  elseif self:isVector()    then return self.type:terraType()
  elseif self:isRow()       then return self:terraType()
  end
  error("terraBaseType method not implemented for type " .. self:toString(), 2)
end

--function Type:dataType()
--  if self:isScalar() then return self.type end
--  error("dataType not implemented for type " .. self:toString(),2)
--end


-------------------------------------------------------------------------------
--[[ Stringify types                                                       ]]--
-------------------------------------------------------------------------------
function Type:toString()
  if     self:isPrimitive() then return self.type.string
  elseif self:isVector()    then return 'LVector('..self.type:toString()..
                                        ','..tostring(self.N)..')'
  elseif self:isRow()       then return 'Row('..self.relation._name..')'
  elseif self:isInternal()  then return 'Internal('..self.type.string..')'
  elseif self:isError()     then return 'error'
  end
  print(debug.traceback())
  error('toString method not implemented for this type!', 2)
end
Type.__tostring = Type.toString


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

local vector_types = {}
local function vectorType (typ, len)
  if not Type.isLisztType(typ) or not typ:isPrimitive() then error("invalid type argument to vector type constructor (is this a terra type?)", 2) end
  local tpn = 'vector(' .. typ:toString() .. ',' .. tostring(len) .. ')'
  if not vector_types[tpn] then
    local vt = Type:new(Type.kinds.vector,typ)
    vt.N = len
    vector_types[tpn] = vt
  end
  return vector_types[tpn]
end

local row_type_table = {}
local function rowType (relation)
  if not DECL.is_relation(relation) then
    error("invalid argument to row type constructor. A relation "..
          "must be provided", 2)
  end
  if not row_type_table[relation] then
    local rt = Type:new(Type.kinds.row)
    rt.relation = relation
    row_type_table[relation:name()] = rt
  end
  return row_type_table[relation:name()]
end


-------------------------------------------------------------------------------
--[[ Type interface:                                                       ]]--
-------------------------------------------------------------------------------

-- Primitives
t.int       = Type:new(Type.kinds.primitive,Type.kinds.int)
--t.int8      = Type:new(Type.kinds.primitive,Type.kinds.int8)
--t.int16     = Type:new(Type.kinds.primitive,Type.kinds.int16)
--t.int32     = Type:new(Type.kinds.primitive,Type.kinds.int32)
--t.int64     = Type:new(Type.kinds.primitive,Type.kinds.int64)
t.uint      = Type:new(Type.kinds.primitive,Type.kinds.uint)
t.uint8     = Type:new(Type.kinds.primitive,Type.kinds.uint8)
--t.uint16    = Type:new(Type.kinds.primitive,Type.kinds.uint16)
--t.uint32    = Type:new(Type.kinds.primitive,Type.kinds.uint32)
t.uint64    = Type:new(Type.kinds.primitive,Type.kinds.uint64)
t.bool      = Type:new(Type.kinds.primitive,Type.kinds.bool)
t.float     = Type:new(Type.kinds.primitive,Type.kinds.float)
t.double    = Type:new(Type.kinds.primitive,Type.kinds.double)

-- Complex type constructors
t.vector    = vectorType
t.row       = rowType

-- Errors
t.error     = Type:new(Type.kinds.error)

-- Internals
t.relation  = Type:new(Type.kinds.internal, Type.kinds.relation)
t.field     = Type:new(Type.kinds.internal, Type.kinds.field)
t.scalar    = Type:new(Type.kinds.internal, Type.kinds.scalar)
t.macro     = Type:new(Type.kinds.internal, Type.kinds.macro)
t.func      = Type:new(Type.kinds.internal, Type.kinds.functype)
t.quoteexpr = Type:new(Type.kinds.internal, Type.kinds.quoteexpr)
t.luatable  = Type:new(Type.kinds.internal, Type.kinds.luatable)

-- terra type for address/index encoding of row identifiers
t.addr      = t.uint64


-------------------------------------------------------------------------------
--[[ Type meeting                                                          ]]--
-------------------------------------------------------------------------------

-- num_meet defines a hierarchy of numeric coercions
local function num_meet(ltype, rtype)
      if ltype == t.double or rtype == t.double then return t.double
  elseif ltype == t.float  or rtype == t.float  then return t.float
  elseif ltype == t.uint64 or rtype == t.uint64 then return t.uint64
  --elseif ltype == t.int64  or rtype == t.int64  then return t.int64
  --elseif ltype == t.uint32 or rtype == t.uint32 then return t.uint32
  --elseif ltype == t.int32  or rtype == t.int32  then return t.int32
  -- note: this is assuming that int and uint get mapped to
  -- 32 bit numbers, which is probably wrong somewhere...
  elseif ltype == t.uint   or rtype == t.uint   then return t.uint
  elseif ltype == t.int    or rtype == t.int    then return t.int
  --elseif ltype == t.uint16 or rtype == t.uint16 then return t.uint16
  --elseif ltype == t.int16  or rtype == t.int16  then return t.int16
  elseif ltype == t.uint8  or rtype == t.uint8  then return t.uint8
  --elseif ltype == t.int8   or rtype == t.int8   then return t.int8
  else
    return t.error
  end
end

local function type_meet(ltype, rtype)
  -- helper
  local function vec_meet(ltype, rtype, N)
    local btype = type_meet(ltype:baseType(), rtype:baseType())
    if btype == t.error then return t.error
    else                     return t.vector(btype, N) end
  end

  -- vector meets
  if ltype:isVector() and rtype:isVector() then
    if ltype.N ~= rtype.N then return t.error
    else                       return vec_meet(ltype, rtype, ltype.N) end

  elseif ltype:isVector() and rtype:isPrimitive() then
    return vec_meet(ltype, rtype, ltype.N)

  elseif ltype:isPrimitive() and rtype:isVector() then
    return vec_meet(ltype, rtype, rtype.N)

  -- primitive meets
  elseif ltype:isPrimitive() and rtype:isPrimitive() then
    if ltype:isNumeric() and rtype:isNumeric() then
      return num_meet(ltype, rtype)

    elseif ltype:isLogical() and rtype:isLogical() then
      return t.bool

    end
  end

  -- default is to error
  return t.error
end


-------------------------------------------------------------------------------
--[[ Lua & Terra Interoperation                                            ]]--
-------------------------------------------------------------------------------

local function luaValConformsToType (luaval, tp)
  -- primitives
  if tp:isPrimitive() then
    return (tp:isNumeric() and type(luaval) == 'number') or
           (tp:isLogical() and type(luaval) == 'boolean')
  -- vectors
  elseif tp:isVector()  then
    -- accept an instance of an LVector:
    if DECL.is_vector(luaval) then 
      return (tp:isLogical() and luaval.type:isLogical()) or
             (tp:isNumeric() and luaval.type:isNumeric())

    -- we also accept arrays as instances of vectors
    elseif type(luaval) == 'table' then
      -- make sure array is of the correct length
      if #luaval ~= tp.N then return false end
      -- make sure each element conforms to the vector data type
      for i = 1, #luaval do
        if not luaValConformsToType(luaval[i], tp:baseType()) then
          return false
        end
      end
      return true
    end
  end
  return false
end

local ttol = {
  [bool]   = t.bool,
  [double] = t.double,
  [float]  = t.float,
  [int]    = t.int,
  --[int8]   = t.int8,
  --[int16]  = t.int16,
  --[int32]  = t.int32,
  --[int64]  = t.int64,
  [uint]   = t.uint,
  [uint8]  = t.uint8,
  --[uint16] = t.uint16,
  --[uint32] = t.uint32,
  [uint64] = t.uint64,
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
exports.type_meet             = type_meet
exports.luaValConformsToType  = luaValConformsToType
exports.terraToLisztType      = terra_to_liszt
return exports

