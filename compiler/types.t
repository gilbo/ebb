
local T = {}
package.loaded["compiler.types"] = T

local L = terralib.require "compiler.lisztlib"

-------------------------------------------------------------------------------
--[[ Liszt type prototype:                                                 ]]--
-------------------------------------------------------------------------------
local Type   = {}
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

function T.isLisztType (obj)
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
  elseif self:isRow()       then return L.addr:terraType()
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
  if not T.isLisztType(typ) or not typ:isPrimitive() then error("invalid type argument to vector type constructor (is this a terra type?)", 2) end
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
  if not L.is_relation(relation) then
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
L.int       = Type:new(Type.kinds.primitive,Type.kinds.int)
--L.int8      = Type:new(Type.kinds.primitive,Type.kinds.int8)
--L.int16     = Type:new(Type.kinds.primitive,Type.kinds.int16)
--L.int32     = Type:new(Type.kinds.primitive,Type.kinds.int32)
--L.int64     = Type:new(Type.kinds.primitive,Type.kinds.int64)
L.uint      = Type:new(Type.kinds.primitive,Type.kinds.uint)
L.uint8     = Type:new(Type.kinds.primitive,Type.kinds.uint8)
--L.uint16    = Type:new(Type.kinds.primitive,Type.kinds.uint16)
--L.uint32    = Type:new(Type.kinds.primitive,Type.kinds.uint32)
L.uint64    = Type:new(Type.kinds.primitive,Type.kinds.uint64)
L.bool      = Type:new(Type.kinds.primitive,Type.kinds.bool)
L.float     = Type:new(Type.kinds.primitive,Type.kinds.float)
L.double    = Type:new(Type.kinds.primitive,Type.kinds.double)

-- Complex type constructors
L.vector    = vectorType
L.row       = rowType

-- Errors
L.error     = Type:new(Type.kinds.error)

-- Internals
L.relation  = Type:new(Type.kinds.internal, Type.kinds.relation)
L.field     = Type:new(Type.kinds.internal, Type.kinds.field)
L.scalar    = Type:new(Type.kinds.internal, Type.kinds.scalar)
L.macro     = Type:new(Type.kinds.internal, Type.kinds.macro)
L.func      = Type:new(Type.kinds.internal, Type.kinds.functype)
L.quoteexpr = Type:new(Type.kinds.internal, Type.kinds.quoteexpr)
L.luatable  = Type:new(Type.kinds.internal, Type.kinds.luatable)

-- terra type for address/index encoding of row identifiers
L.addr      = L.uint64



-------------------------------------------------------------------------------
--[[ Stringify types                                                       ]]--
-------------------------------------------------------------------------------
function Type:toString()
  if     self:isPrimitive() then return self.type.string
  elseif self:isVector()    then return 'Vector('..self.type:toString()..
                                        ','..tostring(self.N)..')'
  elseif self:isRow()       then return 'Row('..self.relation._name..')'
  elseif self:isInternal()  then return 'Internal('..self.type.string..')'
  elseif self:isError()     then return 'error'
  end
  print(debug.traceback())
  error('toString method not implemented for this type!', 2)
end


local str_to_primitive = {
  int    = L.int,
  --int8   = L.int8,
  --int16  = L.int16,
  --int32  = L.int32,
  --int64  = L.int64,
  uint   = L.uint,
  uint8  = L.uint8,
  --uint16 = L.uint16,
  --uint32 = L.uint32,
  uint64 = L.uint64,
  bool   = L.bool,
  float  = L.float,
  double = L.double,
}

-- THIS DOES NOT EMIT A STRING
-- It outputs a LUA table which can be JSON stringified safely
function Type:json_serialize(rel_to_name)
  rel_to_name = rel_to_name or {}
  if     self:isPrimitive() then
    return { basic_kind = 'primitive',
             primitive = self.type.string }

  elseif self:isVector()    then
    return { basic_kind = 'vector',
             base = self.type:json_serialize(),
             n = self.N }

  elseif self:isRow()       then
    local relname = rel_to_name[self.relation]
    if not relname then
      error('Could not find a relation name when attempting to '..
            'JSON serialize a Row type', 2)
    end
    return { basic_kind = 'row',
             relation   = relname }

  else
    error('Cannot serialize type: '..self:toString(), 2)
  end
end

-- given a table output by json_serialize, deserialize reconstructs
-- the correct Type object with correct metatable, etc.
function Type.json_deserialize(json, name_to_rel)
  name_to_rel = name_to_rel or {}
  if not type(json) == 'table' then
    error('Tried to deserialize type, but found a non.', 2)
  end
  if json.basic_kind == 'primitive' then
    local primitive = str_to_primitive[json.primitive]
    if not primitive then
      error('Tried to deserialize primitive but type "'..json.primitive..
            '" is not supported.', 2)
    end
    return primitive

  elseif json.basic_kind == 'vector' then
    local baseType = Type.json_deserialize(json.base)
    return L.vector(baseType, json.n)

  elseif json.basic_kind == 'row' then
    local relation = name_to_rel[json.relation]
    return L.row(relation)

  else
    error('Cannot deserialize type, could not find basic kind of type', 2)
  end
end



-------------------------------------------------------------------------------
--[[ Type meeting                                                          ]]--
-------------------------------------------------------------------------------

-- num_meet defines a hierarchy of numeric coercions
local function num_meet(ltype, rtype)
      if ltype == L.double or rtype == L.double then return L.double
  elseif ltype == L.float  or rtype == L.float  then return L.float
  elseif ltype == L.uint64 or rtype == L.uint64 then return L.uint64
  --elseif ltype == L.int64  or rtype == L.int64  then return L.int64
  --elseif ltype == L.uint32 or rtype == L.uint32 then return L.uint32
  --elseif ltype == L.int32  or rtype == L.int32  then return L.int32
  -- note: this is assuming that int and uint get mapped to
  -- 32 bit numbers, which is probably wrong somewhere...
  elseif ltype == L.uint   or rtype == L.uint   then return L.uint
  elseif ltype == L.int    or rtype == L.int    then return L.int
  --elseif ltype == L.uint16 or rtype == L.uint16 then return L.uint16
  --elseif ltype == L.int16  or rtype == L.int16  then return L.int16
  elseif ltype == L.uint8  or rtype == L.uint8  then return L.uint8
  --elseif ltype == L.int8   or rtype == L.int8   then return L.int8
  else
    return L.error
  end
end

local function type_meet(ltype, rtype)
  -- helper
  local function vec_meet(ltype, rtype, N)
    local btype = type_meet(ltype:baseType(), rtype:baseType())
    if btype == L.error then return L.error
    else                     return L.vector(btype, N) end
  end

  -- vector meets
  if ltype:isVector() and rtype:isVector() then
    if ltype.N ~= rtype.N then return L.error
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
      return L.bool

    end
  end

  -- default is to error
  return L.error
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
    if L.is_vector(luaval) then 
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
  [bool]   = L.bool,
  [double] = L.double,
  [float]  = L.float,
  [int]    = L.int,
  --[int8]   = L.int8,
  --[int16]  = L.int16,
  --[int32]  = L.int32,
  --[int64]  = L.int64,
  [uint]   = L.uint,
  [uint8]  = L.uint8,
  --[uint16] = L.uint16,
  --[uint32] = L.uint32,
  [uint64] = L.uint64,
}


-- converts a terra vector or primitive type into a liszt type
local function terraToLisztType (tp)
  -- return primitive type
  if ttol[tp] then return ttol[tp] end

  -- return vector type
  if tp:isvector() then
    local p = terraToLisztType(tp.type)
    if p == nil then return nil end
    return L.vector(p,tp.N)
  end
  return nil
end


-------------------------------------------------------------------------------
--[[ export type api                                                       ]]--
-------------------------------------------------------------------------------
T.type_meet             = type_meet
T.luaValConformsToType  = luaValConformsToType
T.terraToLisztType      = terraToLisztType
T.Type = Type

