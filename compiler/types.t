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
function Type:isScalarRow()
  return self.kind == "row"
end
function Type:isScalar()
  return self:isPrimitive() or self:isScalarRow()
end
function Type:isVector()
  return self.kind == "vector"
end
function Type:isSmallMatrix()
  return self.kind == "smallmatrix"
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
  if self:isVector() or self:isSmallMatrix() then
    return self.type:isPrimitive()
  else
    return self:isPrimitive()
  end
end

-- These are types that are valid to use for a field
function Type:isFieldType()
  return self:isScalar() or self:isVector() or self:isSmallMatrix()
end


-------------------------------------------------------------------------------
--[[ Primitive/Vector Type Methods                                         ]]--
-------------------------------------------------------------------------------

-- of type integer or multiple integers
function Type:isIntegral ()
  return self:isValueType() and self:terraBaseType():isintegral()
end

function Type:isNumeric ()
  return self:isValueType() and self:terraBaseType():isarithmetic()
end

function Type:isLogical ()
  return self:isValueType() and self:terraBaseType() == bool
end

function Type:isRow()
  return self:isFieldType() and self:baseType():isScalarRow()
end

-------------------------------------------------------------------------------
--[[ Methods for computing terra or runtime types                          ]]--
-------------------------------------------------------------------------------

 
function Type:baseType()
  if self:isVector()      then return self.type end
  if self:isSmallMatrix() then return self.type end
  if self:isScalar()      then return self end
  error("baseType not implemented for " .. self:toString(),2)
end

local struct emptyStruct {}
local struct  QueryType {
    start : uint64;
    finish : uint64;
}
function Type:terraType()
  if     self:isPrimitive()   then return self.terratype
  elseif self:isVector()      then return self.terratype
  elseif self:isSmallMatrix() then return self.terratype
  elseif self:isRow()         then return self.terratype
  elseif self:isQuery()       then return QueryType
  elseif self:isInternal()    then return emptyStruct
  elseif self:isError()       then return emptyStruct
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
  -- Liszt may not be fully loaded yet, so...
  if L.is_relation and L.is_relation(typ) then typ = L.row(typ) end
  if not T.isLisztType(typ) or not typ:isScalar() then
    error("invalid type argument to vector type constructor "..
          "(is this a terra type?)", 2)
  end
  if not vector_types[typ] then vector_types[typ] = {} end
  if not vector_types[typ][len] then
    local vt = Type:new("vector")
    vt.N = len
    vt.type = typ
    local ttype = typ:terraType()
    local struct_name = "vector_" .. tostring(ttype) .. "_" .. tostring(vt.N)
    vt.terratype = struct { d : ttype[vt.N]; }
    vt.terratype.metamethods.__typename = function(self)
      return struct_name
    end
    vector_types[typ][len] = vt
  end
  return vector_types[typ][len]
end

local smatrix_types = {}
local function smallMatrixType (typ, nrow, ncol)
  -- Liszt may not be fully loaded yet, so...
  if L.is_relation and L.is_relation(typ) then typ = L.row(typ) end
  if not T.isLisztType(typ) or not typ:isScalar() then
    error("invalid type argument to small matrix type constructor "..
          "(is this a terra type?)", 2)
  end
  if not smatrix_types[typ]       then smatrix_types[typ] = {} end
  if not smatrix_types[typ][nrow] then smatrix_types[typ][nrow] = {} end
  if not smatrix_types[typ][nrow][ncol] then
    local smt = Type:new("smallmatrix")
    smt.Nrow = nrow
    smt.Ncol = ncol
    smt.type = typ
    local ttype = typ:terraType()
    local struct_name = "smallmatrix_" .. tostring(ttype) .. "_" ..
                        tostring(smt.Nrow) .. '_' .. tostring(smt.Ncol)
    smt.terratype = struct { d : ttype[smt.Ncol][smt.Nrow] }
    smt.terratype.metamethods.__typename = function(self)
      return struct_name
    end
    smatrix_types[typ][nrow][ncol] = smt
  end
  return smatrix_types[typ][nrow][ncol]
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
    rt.terratype = L.addr:terraType()
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
for i=1,#primitives do
  local p = primitives[i]
  local t = Type:new("primitive")
  t.terratype = _G[p] 
  t.name = p
  L[p] = t
  terraprimitive_to_liszt[t.terratype] = t
  primitives[i] = t
end

-- Complex type constructors
L.vector        = vectorType
L.smallmatrix   = smallMatrixType
L.row           = rowType
L.record        = recordType
L.internal      = internalType
L.query         = queryType
-- Errors
L.error         = Type:new("error")

-- terra type for address/index encoding of row identifiers
L.addr          = L.uint64



-------------------------------------------------------------------------------
--[[ Stringify types                                                       ]]--
-------------------------------------------------------------------------------
function Type:toString()
  if     self:isPrimitive()   then return self.name
  elseif self:isVector()      then return 'Vector('..self.type:toString()..
                                          ','..tostring(self.N)..')'
  elseif self:isSmallMatrix() then return 'SmallMatrix('..
                                          self.type:toString()..','..
                                          tostring(self.Nrow)..','..
                                          tostring(self.Ncol)..')'
  elseif self:isRow()         then return 'Row('..self.relation:Name()..')'
  elseif self:isRecord()      then
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
    local base, n = str:match('^Vector%(([^,]*),([^%)]*)%)$')
    n = tonumber(n)
    if n == nil then
      error("When constructing a Vector type from a string, "..
            "no length was found.", 2)
    end
    base = Type.fromString(base)
    return L.vector(base, n)
  elseif str:sub(1,11) == 'SmallMatrix' then
    local base, n, m = str:match('^SmallMatrix%(([^,]*),([^,]*),([^%)]*)%)$')
    n = tonumber(n)
    m = tonumber(m)
    if n == nil or m == nil then
      error("When constructing a SmallMatrix type from a string, "..
            "did not find both number of rows and number of columns", 2)
    end
    base = Type.fromString(base)
    return L.smallmatrix(base, n, m)
  else
    local lookup = primitive_set[str]
    if lookup then
      return L[str]
    else
      error("Tried to construct a type from a string which does not "..
            "express a vector, smallmatrix, or primitive type", 2)
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

  elseif self:isSmallMatrix() then
    return { basic_kind = 'smallmatrix',
             base = self.type:json_serialize(),
             nrow = self.Nrow,
             ncol = self.Ncol }

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

  elseif json.basic_kind == 'smallmatrix' then
    if type(json.base) ~= 'table' then
      error('Tried to deserialize smallmatrix but missing base type', 2)
    end
    local baseType = Type.json_deserialize(json.base)
    return L.smallmatrix(baseType, json.nrow, json.ncol)

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


-------------------------------------------------------------------------------
--[[ Type meeting                                                          ]]--
-------------------------------------------------------------------------------

-- CURRENT, COMPLETE RULES FOR PRIMITIVES
-- int < float < double
-- int < uint64

local prim_lessthan = {}
for i,p in ipairs(primitives) do
  prim_lessthan[p] = {}
  -- default error
  for j,pp in ipairs(primitives) do prim_lessthan[p][pp] = L.error end
  -- diagonal is always ok
  prim_lessthan[p][p] = p
end
prim_lessthan[L.float][L.double] = L.double
prim_lessthan[L.int][L.double]   = L.double
prim_lessthan[L.int][L.float]    = L.float -- WARNING: LOSES PRECISION
prim_lessthan[L.int][L.uint64]   = L.uint64

-- primitive meet  (construct by symmetrizing the lessthan table)
local prim_meet = {}
for i,p in ipairs(primitives) do
  prim_meet[p] = {}
  for j,q in ipairs(primitives) do
    local pq = prim_lessthan[p][q]
    local qp = prim_lessthan[q][p]
    local val = pq
    if qp ~= L.error then val = qp end
    prim_meet[p][q] = val
  end
end


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

    -- similarly, we can unpack matching dimension smallmatrices
    if source:isSmallMatrix() and target:isSmallMatrix() and
       source.Nrow == target.Nrow and source.Ncol == target.Ncol
    then
      source = source:baseType()
      target = target:baseType()
    end

    if source:isPrimitive() and target:isPrimitive() then
      local top = prim_lessthan[source][target]
      if top ~= L.error then return true end
    end
  end

  -- In all other cases, coercion fails
  return false
end

-- helpers
local function vec_meet(ltype, rtype, N)
    local btype = prim_meet[ltype:baseType()][rtype:baseType()]
    if btype == L.error then return L.error
                        else return L.vector(btype, N) end
end

local function mat_meet(ltype, rtype, N, M)
  local btype = prim_meet[ltype:baseType()][rtype:baseType()]
  if btype == L.error then return L.error
                      else return L.smallmatrix(btype,N,M) end
end

local function type_meet(ltype, rtype)
  -- matching case
  if ltype == rtype then return ltype end

  -- outside the matching case, both values must be numeric
  if ltype:isNumeric() and rtype:isNumeric() then

    -- base-type-meet
    local base_meet = prim_meet[ltype:baseType()][rtype:baseType()]
    if base_meet == L.error then return L.error end

    -- primitive meets
    if ltype:isPrimitive() and rtype:isPrimitive() then
      return base_meet

    -- vector meets
    elseif ltype:isVector() and rtype:isVector() then
      if ltype.N == rtype.N then
        return L.vector(base_meet, ltype.N) end

    elseif ltype:isVector() and rtype:isPrimitive() then
      return L.vector(base_meet, ltype.N)

    elseif ltype:isPrimitive() and rtype:isVector() then
      return L.vector(base_meet, rtype.N)

    -- smallmatrix meets
    elseif ltype:isSmallMatrix() and rtype:isSmallMatrix() then
      if ltype.Nrow == rtype.Nrow or ltype.Ncol == rtype.Ncol then
        return L.smallmatrix(base_meet, ltype.Nrow, ltype.Ncol) end

    elseif ltype:isSmallMatrix() and rtype:isPrimitive() then
      return L.smallmatrix(base_meet, ltype.Nrow, ltype.Ncol)

    elseif ltype:isPrimitive() and rtype:isSmallMatrix() then
      return L.smallmatrix(base_meet, rtype.Nrow, rtype.Ncol)

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
  
  -- return vector type  (WARNING: different backing for types now...)
  if tp:isvector() then
    local p = terraToLisztType(tp.type)
    return p and L.vector(p,tp.N)
  end
  
  return nil
end


-------------------------------------------------------------------------------
--[[ type aliases                                                          ]]--
-------------------------------------------------------------------------------
--L.vec2i     = L.vector(L.int, 2)
--L.vec3i     = L.vector(L.int, 3)
--L.vec4i     = L.vector(L.int, 4)
--
--L.vec2f     = L.vector(L.float, 2)
--L.vec3f     = L.vector(L.float, 3)
--L.vec4f     = L.vector(L.float, 4)
--L.vec2d     = L.vector(L.double, 2)
--L.vec3d     = L.vector(L.double, 3)
--L.vec4d     = L.vector(L.double, 4)
--
--L.vec2b     = L.vector(L.bool, 2)
--L.vec3b     = L.vector(L.bool, 3)
--L.vec4b     = L.vector(L.bool, 4)

for n=2,4 do
  local vecname = 'vec'..tostring(n)
  L[vecname..'i'] = L.vector(L.int, n)
  L[vecname..'f'] = L.vector(L.float, n)
  L[vecname..'d'] = L.vector(L.double, n)
  L[vecname..'b'] = L.vector(L.bool, n)

  for m=2,4 do
    local matname = 'mat'..tostring(n)..'x'..tostring(m)
    L[matname..'i'] = L.smallmatrix(L.int, n, m)
    L[matname..'f'] = L.smallmatrix(L.float, n, m)
    L[matname..'d'] = L.smallmatrix(L.double, n, m)
    L[matname..'b'] = L.smallmatrix(L.bool, n, m)
  end

  -- alias square matrices
  local shortname = 'mat'..tostring(n)
  local fullname  = 'mat'..tostring(n)..'x'..tostring(n)
  L[shortname..'i'] = L[fullname..'i']
  L[shortname..'f'] = L[fullname..'f']
  L[shortname..'d'] = L[fullname..'d']
  L[shortname..'b'] = L[fullname..'b']
end


-------------------------------------------------------------------------------
--[[ export type api                                                       ]]--
-------------------------------------------------------------------------------
T.type_meet             = type_meet
T.luaValConformsToType  = luaValConformsToType
T.terraToLisztType      = terraToLisztType
T.Type = Type

return T
