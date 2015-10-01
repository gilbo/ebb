local T = {}
-- Warning: DO NOT UNCOMMENT this line!
-- Circular dependencies for types.t cannot be resolved this way because Liszt
-- types are defined inline when this file is first executed.
-- packed.loaded["ebb.src.types"] = T

local L   = require "ebb.src.lisztlib"

local use_legion = not not rawget(_G, '_legion_env')
local LW
if use_legion then LW  = require "ebb.src.legionwrap" end

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
L.is_type = T.isLisztType

-------------------------------------------------------------------------------
--[[ Basic Type Methods                                                    ]]--
-------------------------------------------------------------------------------
-- There are 6 basic kinds of types:
function Type:isPrimitive()
  return self.kind == "primitive"
end
function Type:isScalarKey()
  return self.kind == "key"
end
function Type:isScalar()
  return self:isPrimitive() or self:isScalarKey()
end
function Type:isVector()
  return self.kind == "vector"
end
function Type:isMatrix()
  return self.kind == "matrix"
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

-- These types represent Liszt values (not keys though)
function Type:isValueType()
  if self:isVector() or self:isMatrix() then
    return self.type:isPrimitive()
  else
    return self:isPrimitive()
  end
end

-- These are types that are valid to use for a field
function Type:isFieldType()
  return self:isScalar() or self:isVector() or self:isMatrix()
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

function Type:isKey()
  return self:isFieldType() and self:baseType():isScalarKey()
end

-------------------------------------------------------------------------------
--[[ Methods for computing terra or runtime types                          ]]--
-------------------------------------------------------------------------------

 
function Type:baseType()
  if self:isVector()      then return self.type end
  if self:isMatrix()      then return self.type end
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
  elseif self:isMatrix()      then return self.terratype
  elseif self:isKey()         then return self.terratype
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
  if L.is_relation and L.is_relation(typ) then typ = L.key(typ) end
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
local function matrixType (typ, nrow, ncol)
  -- Liszt may not be fully loaded yet, so...
  if L.is_relation and L.is_relation(typ) then typ = L.key(typ) end
  if not T.isLisztType(typ) or not typ:isScalar() then
    error("invalid type argument to small matrix type constructor "..
          "(is this a terra type?)", 2)
  end
  if not smatrix_types[typ]       then smatrix_types[typ] = {} end
  if not smatrix_types[typ][nrow] then smatrix_types[typ][nrow] = {} end
  if not smatrix_types[typ][nrow][ncol] then
    local smt = Type:new("matrix")
    smt.Nrow = nrow
    smt.Ncol = ncol
    smt.type = typ
    local ttype = typ:terraType()
    local struct_name = "matrix_" .. tostring(ttype) .. "_" ..
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
       not (typ:isValueType() or typ:isKey())
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
--[[
L.addr_terra_types = {}
for i=1,3 do
  local struct_name = "addr_"..tostring(i)
  L.addr_terra_types[i] = struct { a : uint64[i]; }
  L.addr_terra_types[i].metamethods.__typename = function(self)
    return struct_name
  end
  L.addr_terra_types[i].metamethods.__eq = macro(function(lhs,rhs)
    local exp = `lhs.a[0] == rhs.a[0]
    for k=2,i do exp = `exp and lhs.a[k] == rhs.a[k] end
    return exp
  end)
  L.addr_terra_types[i].metamethods.__ne = macro(function(lhs,rhs)
    return `not lhs == rhs
  end)
end
function T.linAddrLua(addr, dims)
      if #dims == 1 then return addr.a[0]
  elseif #dims == 2 then return addr.a[0] + dims[1] * addr.a[1]
  elseif #dims == 3 then return addr.a[0] + dims[1] * (addr.a[1] +
                                                       dims[2]*addr.a[2])
  else error('INTERNAL > 3 dimensional address???') end
end
function T.linAddrTerraGen(dims)
  if #dims == 1 then
    return macro(function(addr)
      return `addr.a[0]
    end)
  elseif #dims == 2 then
    return macro(function(addr)
      return quote var a = addr.a in a[0] + [ dims[1] ] * a[1] end
    end)
  elseif #dims == 3 then
    return macro(function(addr)
      return quote var a = addr.a in a[0] +
                           [ dims[1] ] * (a[1] + [ dims[2] ]*a[2]) end
    end)
  else error('INTERNAL > 3 dimensional address???') end
end
]]
local function dims_to_bit_dims(dims)
  local bitdims = {}
  for k=1,#dims do
    if      dims[k] < 256         then  bitdims[k] = 8
    elseif  dims[k] < 65536       then  bitdims[k] = 16
    elseif  dims[k] < 4294967296  then  bitdims[k] = 32
                                  else  bitdims[k] = 64 end
  end
  return bitdims
end
local function bit_dims_to_dim_types(bitdims)
  local dimtyps = {}
  for k=1,#bitdims do
    if      bitdims[k] == 8   then  dimtyps[k] = uint8
    elseif  bitdims[k] == 16  then  dimtyps[k] = uint16
    elseif  bitdims[k] == 32  then  dimtyps[k] = uint32
                              else  dimtyps[k] = uint64 end
  end
  return dimtyps
end
local function dims_to_strides(dims)
  local strides = {1}
  if #dims >= 2 then  strides[2] = dims[1]            end
  if #dims >= 3 then  strides[3] = dims[1] * dims[2]  end
  return strides
end
local function lua_lin_gen(strides)
  local code = 'return function(self) return self.a0'
  for k=2,#strides do
    code = code..' + '..tostring(strides[k])..' * self.a'..tostring(k-1)
  end
  return assert(loadstring(code..' end'))()
end
local function terra_lin_gen(keytyp, strides)
  local key   = symbol(keytyp)
  local exp   = `key.a0
  for k=2,#strides do
    exp = `[exp] + [strides[k]] * key.['a'..tostring(k-1)]
  end
  return terra( [key] ) : uint64  return exp  end
end
local function legion_terra_lin_gen(keytyp)
  local key     = symbol(keytyp)
  local strides = symbol(LW.legion_byte_offset_t[#keytyp.entries])
  local exp     = `key.a0 * strides[0].offset
  for k=2,#keytyp.entries do
    exp = `[exp] + strides[k-1].offset * key.['a'..tostring(k-1)]
  end
  return terra( [key], [strides] ) : uint64   return exp  end
end
local function get_physical_key_type(rel)
  local cached = rawget(rel, '_key_type_cached')
  if cached then return cached end
  -- If not cached, then execute the rest of this function
  -- to build the type

  local dims    = rel:Dims()
  local bitdims = dims_to_bit_dims(dims)
  local dimtyps = bit_dims_to_dim_types(bitdims)
  local strides = dims_to_strides(dims)
  if rel:isElastic() then
    bitdims = {64}
    dimtyps = {uint64}
  end

  local name    = 'key'
  for k=1,#bitdims do name = name .. '_' .. tostring(bitdims[k]) end
  name = name .. '_'..rel:Name()

  local PhysKey = terralib.types.newstruct(name)
  for k=1,#bitdims do name = 
    table.insert(PhysKey.entries,
                 { field = 'a'..tostring(k-1), type = dimtyps[k] })
  end

  -- install specialized methods
  PhysKey.methods.luaLinearize            = lua_lin_gen(strides)
  PhysKey.methods.terraLinearize          = terra_lin_gen(PhysKey, strides)
  if use_legion then
    PhysKey.methods.legionTerraLinearize  = legion_terra_lin_gen(PhysKey)
  end
  -- add equality / inequality tests
  local ndim = #dims
  PhysKey.metamethods.__eq = macro(function(lhs,rhs)
    local exp = `lhs.a0 == rhs.a0
    for k=2,ndim do
      local astr = 'a'..tostring(k-1)
      exp = `exp and lhs.[astr] == rhs.[astr]
    end
    return exp
  end)
  PhysKey.metamethods.__ne = macro(function(lhs,rhs)
    return `not lhs == rhs
  end)

  return PhysKey
end

local keyType = cached(function(relation)
    checkrelation(relation)
    local rt = Type:new("key")
    rt.relation = relation
    rt.ndims    = relation:nDims()
    rt.terratype = get_physical_key_type(relation)
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
local primitives = {"int","uint", "uint64","bool","float","double"}
for i=1,#primitives do
  local p = primitives[i]
  local t = Type:new("primitive")
  t.terratype = _G[p] 
  t.name = p
  L[p] = t
  terraprimitive_to_liszt[t.terratype] = t
  primitives[i] = t
end
L.color_type = L.uint
if use_legion then
  assert(uint == LW.legion_color_t)
end

-- Complex type constructors
L.vector        = vectorType
L.matrix        = matrixType
L.key           = keyType
L.record        = recordType
L.internal      = internalType
L.query         = queryType
-- Errors
L.error         = Type:new("error")



-------------------------------------------------------------------------------
--[[ Stringify types                                                       ]]--
-------------------------------------------------------------------------------
function Type:toString()
  if     self:isPrimitive()   then return self.name
  elseif self:isVector()      then return 'Vector('..self.type:toString()..
                                          ','..tostring(self.N)..')'
  elseif self:isMatrix()      then return 'Matrix('..
                                          self.type:toString()..','..
                                          tostring(self.Nrow)..','..
                                          tostring(self.Ncol)..')'
  elseif self:isKey()         then return 'Key('..self.relation:Name()..')'
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


-------------------------------------------------------------------------------
--[[ Type ordering / joining / coercion                                    ]]--
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

-- primitive join  (construct by symmetrizing the lessthan table)
-- this works because there is currently no primitive type C s.t.
--    A < C and B < C
-- but with A and B being incomparable
local prim_join = {}
for i,p in ipairs(primitives) do
  prim_join[p] = {}
  for j,q in ipairs(primitives) do
    -- get both less than values
    local pq = prim_lessthan[p][q]
    local qp = prim_lessthan[q][p]
    -- choose whichever is not an error
    local val = pq
    if qp ~= L.error then val = qp end
    -- and assign that as the join value
    prim_join[p][q] = val
  end
end


-- Coercion defines a partial order.  This function defines that order
function Type:isCoercableTo(target)
  -- make sure we have two types
  if not T.isLisztType(target) then return false end

  -- identity relationship preserved
  -- ensures reflexivity of the relation
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
    if source:isMatrix() and target:isMatrix() and
       source.Nrow == target.Nrow and source.Ncol == target.Ncol
    then
      source = source:baseType()
      target = target:baseType()
    end

    -- appeal to the lessthan table for primitive types
    if source:isPrimitive() and target:isPrimitive() then
      local top = prim_lessthan[source][target]
      if top ~= L.error then return true end
    end
  end

  -- In all other cases, coercion fails
  return false
end

-- helpers
local function vec_join(ltype, rtype, N)
    local btype = prim_join[ltype:baseType()][rtype:baseType()]
    if btype == L.error then return L.error
                        else return L.vector(btype, N) end
end

local function mat_join(ltype, rtype, N, M)
  local btype = prim_join[ltype:baseType()][rtype:baseType()]
  if btype == L.error then return L.error
                      else return L.matrix(btype,N,M) end
end

local function type_join(ltype, rtype)
  -- matching case
  if ltype == rtype then return ltype end

  -- outside the matching case, both values must be numeric
  -- We allow two cases of vector/matrix dimensioning:
  --    1. both lhs, rhs vectors or matrices of matching dimension
  --    2. one of lhs, rhs vector/matrix and other scalar
  if ltype:isNumeric() and rtype:isNumeric() then

    -- base-type-join
    local base_join = prim_join[ltype:baseType()][rtype:baseType()]
    if base_join == L.error then return L.error end

    -- primitive joins
    if ltype:isPrimitive() and rtype:isPrimitive() then
      return base_join

    -- vector joins
    elseif ltype:isVector() and rtype:isVector() then
      if ltype.N == rtype.N then
        return L.vector(base_join, ltype.N) end

    elseif ltype:isVector() and rtype:isPrimitive() then
      return L.vector(base_join, ltype.N)

    elseif ltype:isPrimitive() and rtype:isVector() then
      return L.vector(base_join, rtype.N)

    -- matrix joins
    elseif ltype:isMatrix() and rtype:isMatrix() then
      if ltype.Nrow == rtype.Nrow or ltype.Ncol == rtype.Ncol then
        return L.matrix(base_join, ltype.Nrow, ltype.Ncol) end

    elseif ltype:isMatrix() and rtype:isPrimitive() then
      return L.matrix(base_join, ltype.Nrow, ltype.Ncol)

    elseif ltype:isPrimitive() and rtype:isMatrix() then
      return L.matrix(base_join, rtype.Nrow, rtype.Ncol)

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

  -- keys
  elseif tp:isScalarKey() then
    if tp.ndims == 1 then
      return type(luaval) == 'number'
    else
      if type(luaval) ~= 'table' or #luaval ~= tp.ndims then return false end
      for i=1,tp.ndims do
        if not type(luaval[i]) == 'number' then return false end
      end
      return true
    end

  -- vectors
  elseif tp:isVector()  then
    if type(luaval) ~= 'table' or #luaval ~= tp.N then return false end
    -- make sure each element conforms to the vector data type
    for i = 1, #luaval do
      if not luaValConformsToType(luaval[i], tp:baseType()) then
        return false
      end
    end
    -- if we made it here, everything checked out
    return true

  elseif tp:isMatrix() then
    if type(luaval) ~= 'table' or #luaval ~= tp.Nrow then return false end
    -- check each row / matrix value
    for r = 1, #luaval do
      local row = luaval[r]
      if type(row) ~= 'table' or #row ~= tp.Ncol then return false end
      for c = 1, #row do
        if not luaValConformsToType(row[c], tp:baseType()) then
          return false
        end
      end
    end
    -- if we made it here, everything checked out
    return true

  end
  return false
end

local function luaToLisztVal (luaval, typ)
  if typ:isPrimitive() then
    return luaval

  elseif typ:isScalarKey() then
    if typ.ndims == 1 then
      return terralib.new(typ:terraType(), { luaval })
    else
      return terralib.new(typ:terraType(), luaval)
    end

  elseif typ:isVector() then
    local btyp      = typ:baseType()
    local terraval = terralib.new(typ:terraType())
    for i=1,typ.N do terraval.d[i-1] = luaToLisztVal(luaval[i], btyp) end
    return terraval

  elseif typ:isMatrix() then
    local btyp      = typ:baseType()
    local terraval = terralib.new(typ:terraType())
    for r=1,typ.Nrow do for c=1,typ.Ncol do
      terraval.d[r-1][c-1] = luaToLisztVal(luaval[r][c], btyp)
    end end
    return terraval

  else
    error('INTERNAL: Should not try to convert lua values to this type: '..
          tostring(typ))
  end
end

local function lisztToLuaVal(lzval, typ)
  if typ:isPrimitive() then
    if typ:isNumeric() then
      return tonumber(lzval)
    elseif typ:isLogical() then
      if type(lzval) == 'cdata' then
        return not (lzval == 0)
      else
        return lzval
      end
    end

  elseif typ:isScalarKey() then
    if typ.ndims == 1 then
      return tonumber(lzval.a0)
    elseif typ.ndims == 2 then
      return { tonumber(lzval.a0), tonumber(lzval.a1) }
    elseif typ.ndims == 3 then
      return { tonumber(lzval.a0), tonumber(lzval.a1), tonumber(lzval.a2) }
    else
      error('INTERNAL: Cannot have > 3 dimensional keys')
    end

  elseif typ:isVector() then
    local vec = {}
    for i=1,typ.N do
      vec[i] = lisztToLuaVal(lzval.d[i-1], typ:baseType())
    end
    return vec

  elseif typ:isMatrix() then
    local mat = {}
    for r=1,typ.Nrow do
      mat[r] = {}
      for c=1,typ.Ncol do
        mat[r][c] = lisztToLuaVal(lzval.d[r-1][c-1], typ:baseType())
      end
    end
    return mat

  else
    error('INTERNAL: Should not try to convert lua values from this type: '..
          tostring(typ))
  end
end

-- converts a terra vector or primitive type into a liszt type
local function terraToLisztType (tp)
  -- return primitive type
  local typ = terraprimitive_to_liszt[tp]
  if typ then return typ end
  
  -- return vector type  (WARNING: different backing for types now...)
  --if tp:isvector() then
  --  local p = terraToLisztType(tp.type)
  --  return p and L.vector(p,tp.N)
  --end
  
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
    L[matname..'i'] = L.matrix(L.int, n, m)
    L[matname..'f'] = L.matrix(L.float, n, m)
    L[matname..'d'] = L.matrix(L.double, n, m)
    L[matname..'b'] = L.matrix(L.bool, n, m)
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
T.type_join             = type_join
T.luaValConformsToType  = luaValConformsToType
T.luaToLisztVal         = luaToLisztVal
T.lisztToLuaVal         = lisztToLuaVal

T.terraToLisztType      = terraToLisztType
T.Type = Type

return T
