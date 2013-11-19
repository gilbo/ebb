
local T = {}
package.loaded["compiler.types"] = T

local L = terralib.require "compiler.lisztlib"

-------------------------------------------------------------------------------
--[[ Liszt type prototype:                                                 ]]--
-------------------------------------------------------------------------------
local Type   = {}
Type.__index = Type

function Type:new (kind)
  return setmetatable({kind = kind}, self)
end

function T.isLisztType (obj)
  return getmetatable(obj) == Type
end

-------------------------------------------------------------------------------
--[[ Basic Type Methods                                                    ]]--
-------------------------------------------------------------------------------
-- There are 5 basic kinds of types:
function Type:isPrimitive()
  return self.kind == "primitive"
end
function Type:isVector()
  return self.kind == "vector"
end
function Type:isRow()
  return self.kind == "row"
end
function Type:isInternal()
  return self.kind == "internal"
end
function Type:isError()
  return self.kind == "error"
end

-- Can this type of AST represent a Liszt value
function Type:isValueType()
  return self:isPrimitive() or self:isVector()
end
-- eventually this type should support rows...

-------------------------------------------------------------------------------
--[[ Primitive/Vector Type Methods                                         ]]--
-------------------------------------------------------------------------------

-- of type integer or vectors of integers
function Type:isIntegral ()
  return self:isValueType() and self.terratype:isintegral()
end

function Type:isNumeric ()
  return self:isValueType() and self.terratype:isarithmetic()
end

function Type:isLogical ()
  return self:isPrimitive() and self.terratype == bool
end

-------------------------------------------------------------------------------
--[[ Methods for computing terra or runtime types                          ]]--
-------------------------------------------------------------------------------

 
function Type:baseType()
  if self:isVector()    then return self.type end
  if self:isPrimitive() or self:isRow() then return self end
  error("baseType not implemented for " .. self:toString(),2)
end

local struct emptyStruct {}

function Type:terraType()
  if     self:isPrimitive() then return self.terratype
  elseif self:isVector()    then return vector(self.type:terraType(), self.N)
  elseif self:isRow()       then return L.addr:terraType()
  elseif self:isInternal()  then return emptyStruct
  end
  error("terraType method not implemented for type " .. self:toString(), 2)
end

function Type:terraBaseType()
    return self:baseType():terraType()
end


-------------------------------------------------------------------------------
--[[ Type constructors:                                                    ]]--
-------------------------------------------------------------------------------

local vector_types = {}
local function vectorType (typ, len)
  if not T.isLisztType(typ) or not typ:isPrimitive() then error("invalid type argument to vector type constructor (is this a terra type?)", 2) end
  local tpn = 'vector(' .. typ:toString() .. ',' .. tostring(len) .. ')'
  if not vector_types[tpn] then
    local vt = Type:new("vector")
    vt.N = len
    vt.type = typ
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
    local rt = Type:new("row")
    rt.relation = relation
    row_type_table[relation:Name()] = rt
  end
  return row_type_table[relation:Name()]
end
local internal_types = {}
local function internalType(obj)
    local t = internal_types[obj]
    if not t then
        t = Type:new("internal")
        t.value = obj
        internal_types[obj] = t
    end
    return t
end

-------------------------------------------------------------------------------
--[[ Type interface:                                                       ]]--
-------------------------------------------------------------------------------

-- Primitives
local terraprimitive_to_liszt = {}
local primitives = {"int","uint","uint8","uint64","bool","float","double"}
for i,p in ipairs(primitives) do
    local t = Type:new("primitive")
    t.terratype = _G[p] 
    t.name = p
    L[p] = t
    terraprimitive_to_liszt[t.terratype] = t
end

-- Complex type constructors
L.vector    = vectorType
L.row       = rowType
L.internal  = internalType

-- Errors
L.error     = Type:new("error")

-- terra type for address/index encoding of row identifiers
L.addr      = L.uint64



-------------------------------------------------------------------------------
--[[ Stringify types                                                       ]]--
-------------------------------------------------------------------------------
function Type:toString()
  if     self:isPrimitive() then return self.name
  elseif self:isVector()    then return 'Vector('..self.type:toString()..
                                        ','..tostring(self.N)..')'
  elseif self:isRow()       then return 'Row('..self.relation:Name()..')'
  elseif self:isInternal()  then return 'Internal('..tostring(self.value)..')'
  elseif self:isError()     then return 'error'
  end
  print(debug.traceback())
  error('toString method not implemented for this type!', 2)
end
Type.__tostring = Type.toString

-- THIS DOES NOT EMIT A STRING
-- It outputs a LUA table which can be JSON stringified safely
function Type:json_serialize(rel_to_name)
  rel_to_name = rel_to_name or {}
  if     self:isPrimitive() then
    return { basic_kind = 'primitive',
             primitive = self.name }

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
    local primitive = L[json.primitive]
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

-- converts a terra vector or primitive type into a liszt type
local function terraToLisztType (tp)
  -- return primitive type
  local typ = terraprimitive_to_liszt[tp]
  if typ then return typ end
  
  -- return vector type
  if tp:isvector() then
    local p = terraToLisztType(tp.type)
    return p and L.vector(p,tp.N)
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

