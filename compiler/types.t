local T = {}
-- Warning: DO NOT UNCOMMENT this line!
-- Circular dependencies for types.t cannot be resolved this way because Liszt
-- types are defined inline when this file is first executed.
-- packed.loaded["compiler.types"] = T

local L = terralib.require "compiler.lisztlib"

-- From the Lua Documentation
function pairs_sorted(tbl, compare)
  local arr = {}
  for k in pairs(tbl) do table.insert(arr, k) end
  table.sort(arr, compare)

  local i = 0
  local iter = function() -- iterator
    i = i + 1
    if arr[i] == nil then return nil
    else return arr[i], tbl[arr[i]] end
  end
  return iter
end

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
-- There are 6 basic kinds of types:
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
function Type:isQuery()
  return self.kind == "query"
end
function Type:isRecord()
  return self.kind == "record"
end

-- These types represent Liszt values (not row references though)
function Type:isValueType()
  return self:isPrimitive() or self:isVector()
end

-- These are types that are valid to use for a field
function Type:isFieldType()
  return self:isValueType() or self:isRow()
end


-------------------------------------------------------------------------------
--[[ Primitive/Vector Type Methods                                         ]]--
-------------------------------------------------------------------------------

-- of type integer or vectors of integers
function Type:isIntegral ()
  return self:isValueType() and self:terraBaseType():isintegral()
end

function Type:isNumeric ()
  return self:isValueType() and self:terraBaseType():isarithmetic()
end

function Type:isLogical ()
  return self:isPrimitive() and self:terraBaseType() == bool
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
local struct  QueryType {
    start : uint64;
    finish : uint64;
}
function Type:terraType()
  if     self:isPrimitive() then return self.terratype
  elseif self:isVector()    then return self.terratype
  elseif self:isRow()       then return L.addr:terraType()
  elseif self:isQuery()     then return QueryType
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
  if not T.isLisztType(typ) or not typ:isPrimitive() then
    error("invalid type argument to vector type constructor "..
          "(is this a terra type?)", 2)
  end
  local tpn = 'vector(' .. typ:toString() .. ',' .. tostring(len) .. ')'
  if not vector_types[tpn] then
    local vt = Type:new("vector")
    vt.N = len
    vt.type = typ
    local ttype = typ:terraType()
    local struct_name = tostring(ttype) .. "_" .. tostring(vt.N)
    vt.terratype = struct { d : ttype[vt.N]; }
    vt.terratype.metamethods.__typename = function(self)
      return struct_name
    end
    vector_types[tpn] = vt
  end
  return vector_types[tpn]
end

local record_types = {}
local function recordType (rec)
  if type(rec) ~= 'table' then
    error('invalid argument to record type constructor:\n'..
          '  must supply a table', 2)
  end
  -- string for de-duplicating types
  local unique_str = '{'
  -- build the string and check the types
  for name, typ in pairs_sorted(rec) do
    if type(name) ~= 'string' then
      error('invalid argument to record type constructor:\n'..
            '  table keys must be strings', 2)
    end
    if not T.isLisztType(typ) or
       not (typ:isValueType() or typ:isRow())
    then
      error('invalid argument to record type constructor:\n'..
            '  table values must be valid types for fields, not '..
            tostring(typ), 2)
    end
    unique_str = unique_str .. tostring(name) .. '=' .. tostring(typ) .. ','
  end
  unique_str = unique_str .. '}'

  if not record_types[unique_str] then
    local rt = Type:new("record")
    rt.rec = rec
    record_types[unique_str] = rt
  end
  return record_types[unique_str]
end

local function cached(ctor)
    local cache = {}
    return function(param)
        local t = cache[param]
        if not t then
            t = ctor(param)
            cache[param] = t
        end
        return t
    end
end
local function checkrelation(relation)
    if not L.is_relation(relation) then
        error("invalid argument to type constructor."..
              "A relation must be provided", 4)
    end
end
local rowType = cached(function(relation)
    checkrelation(relation)
    local rt = Type:new("row")
    rt.relation = relation
    return rt
end)
local internalType = cached(function(obj)
    local t = Type:new("internal")
    t.value = obj
    return t
end)
--we don't bother to de-duplicate query types
--for simplicity and since since queries are not compared to each other
local function queryType(relation,projections)
    local t = Type:new("query")
    t.relation = relation
    t.projections = projections
    return t
end

-------------------------------------------------------------------------------
--[[ Type interface:                                                       ]]--
-------------------------------------------------------------------------------

-- Primitives
local terraprimitive_to_liszt = {}
local primitives = {"int","uint64","bool","float","double"}
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
L.record    = recordType
L.internal  = internalType
L.query     = queryType
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
  elseif self:isRecord()    then
    local str = 'Record({ '
    local first_pair = true
    for name, typ in pairs_sorted(self.rec) do
      if first_pair then first_pair = false
      else str = str .. ', ' end
      str = str .. name .. '=' .. typ:toString()
    end
    str = str .. ' })'
    return str
  elseif self:isQuery()     then return 'Query('..self.relation:Name()..').'
                                        ..table.concat(self.projections,'.') 
  elseif self:isInternal()  then return 'Internal('..tostring(self.value)..')'
  elseif self:isError()     then return 'error'
  end
  error('toString method not implemented for this type!', 2)
end
Type.__tostring = Type.toString

local primitive_set = {
  ["int"   ] = true,
  ["uint64"] = true,
  ["bool"  ] = true,
  ["float" ] = true,
  ["double"] = true,
}
-- For inverting the toString mapping
-- only supports primitives and Vectors
function Type.fromString(str)
  if str:sub(1,6) == 'Vector' then
    local base, n = str:match('Vector%(([^,]*),([^%)]*)%)')
    n = tonumber(n)
    if n == nil then
      error("When constructing a Vector type from a string, "..
            "no length was found.", 2)
    end
    base = Type.fromString(base)
    return L.vector(base, n)
  else
    local lookup = primitive_set[str]
    if lookup then
      return L[str]
    else
      error("Tried to construct a type from a string which does not "..
            "express a vector or primitive type", 2)
    end
  end
end

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
    error('Tried to deserialize type, but found a non-object.', 2)
  end
  if json.basic_kind == 'primitive' then
    local primitive = L[json.primitive]
    if not primitive then
      error('Tried to deserialize primitive but type "'..json.primitive..
            '" is not supported.', 2)
    end
    return primitive

  elseif json.basic_kind == 'vector' then
    if type(json.base) ~= 'table' then
      error('Tried to deserialize vector but missing base type', 2)
    end
    local baseType = Type.json_deserialize(json.base)
    return L.vector(baseType, json.n)

  elseif json.basic_kind == 'row' then
    local relation = name_to_rel[json.relation]
    if not relation then
      error('Tried to deserialize row type, but couldn\'t find the '..
            'relation "'..json.relation..'"', 2)
    end
    return L.row(relation)

  else
    error('Cannot deserialize type, could not find basic kind of type', 2)
  end
end



-------------------------------------------------------------------------------
--[[ Type coercion                                                         ]]--
-------------------------------------------------------------------------------

-- Coercion defines a partial order.  This function defines that order
function Type:isCoercableTo(target)
  -- make sure we have two types
  if not T.isLisztType(target) then return false end

  -- identity relationship preserved
  if self == target then return true end

  -- Only numeric values are coercable otherwise...
  if self:isNumeric() and target:isNumeric() then
    local source = self

    -- If we have matching dimension vectors, then delegate the
    -- decision to the base types
    if source:isVector() and target:isVector() and source.N == target.N then
      source = source:baseType()
      target = target:baseType()
    end

    if source:isPrimitive() and target:isPrimitive() then
      if target == L.double and
         (source == L.float or source == L.int)
      then
        return true
      end
      -- This should probably be stripped out at some point
      -- However, you should expect trouble with the number literals
      -- if you do so.
      if source == L.int and target == L.float then
        return true
      end
      if source == L.int and target == L.uint64 then
        return true
      end
    end
  end

  -- In all other cases, coercion fails
  return false
end

-------------------------------------------------------------------------------
--[[ Type meeting                                                          ]]--
-------------------------------------------------------------------------------

-- num_meet defines a hierarchy of numeric coercions
local function num_meet(ltype, rtype)
      if ltype == L.double or rtype == L.double then return L.double
  elseif ltype == L.float  or rtype == L.float  then return L.float
  elseif ltype == L.uint64 or rtype == L.uint64 then return L.uint64
  -- note: this is assuming that int and uint get mapped to
  -- 32 bit numbers, which is probably wrong somewhere...
  elseif ltype == L.int    or rtype == L.int    then return L.int
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
--[[ type aliases                                                          ]]--
-------------------------------------------------------------------------------
L.vec2f     = L.vector(L.float, 2)
L.vec3f     = L.vector(L.float, 3)
L.vec4f     = L.vector(L.float, 4)
L.vec2d     = L.vector(L.double, 2)
L.vec3d     = L.vector(L.double, 3)
L.vec4d     = L.vector(L.double, 4)

L.vec2b     = L.vector(L.bool, 2)
L.vec3b     = L.vector(L.bool, 3)
L.vec4b     = L.vector(L.bool, 4)


-------------------------------------------------------------------------------
--[[ export type api                                                       ]]--
-------------------------------------------------------------------------------
T.type_meet             = type_meet
T.luaValConformsToType  = luaValConformsToType
T.terraToLisztType      = terraToLisztType
T.Type = Type

return T
