-- The MIT License (MIT)
-- 
-- Copyright (c) 2015 Stanford University.
-- All rights reserved.
-- 
-- Permission is hereby granted, free of charge, to any person obtaining a
-- copy of this software and associated documentation files (the "Software"),
-- to deal in the Software without restriction, including without limitation
-- the rights to use, copy, modify, merge, publish, distribute, sublicense,
-- and/or sell copies of the Software, and to permit persons to whom the
-- Software is furnished to do so, subject to the following conditions:
-- 
-- The above copyright notice and this permission notice shall be included
-- in all copies or substantial portions of the Software.
-- 
-- THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
-- IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
-- FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
-- AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
-- LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
-- FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
-- DEALINGS IN THE SOFTWARE.

local T = {}
-- Warning: DO NOT UNCOMMENT this line!
-- Circular dependencies for types.t cannot be resolved this way because Ebb
-- types are defined inline when this file is first executed.
package.loaded["ebb.src.types"] = T

local DLD = require "ebb.lib.dld"

-- SHOULD eliminate Legion from this file if at all possible
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

local function is_pos_int_val(val)
  return  type(val) == 'number' and
          math.floor(val) == val and
          val > 0
end

-------------------------------------------------------------------------------
--[[ Ebb type prototype:                                                   ]]--
-------------------------------------------------------------------------------
local Type   = {}
Type.__index = Type

local function NewType(kind)
  return setmetatable({kind = kind}, Type)
end

local function istype(obj)
  return getmetatable(obj) == Type
end
T.istype = istype


-------------------------------------------------------------------------------
--[[ Type constructors:                                                    ]]--
-------------------------------------------------------------------------------

-- Primitives
local terraprimitive_to_ebb = {}
local primitives = {"int","uint", "uint64","bool","float","double"}
for i=1,#primitives do
  local p         = primitives[i]
  local t         = NewType("primitive")
  t._terra_type   = _G[p] 
  t.name          = p
  T[p]            = t
  terraprimitive_to_ebb[t._terra_type] = t
  primitives[i]   = t
end

-- TODO: Remove this; why is this an Ebb type?  Is it user-visible or...?
T.color_type = T.uint
if use_legion then
  assert(uint == LW.legion_color_t)
end

-------------------------------------------------------------------------------

local vectortype_cache = {}
local function vectorType (typ, len)
  -- Ebb may not be fully loaded yet, so...
  if T.is_relation and T.is_relation(typ) then typ = T.key(typ) end
  -- memoize
  if vectortype_cache[typ] and vectortype_cache[typ][len] then
    return vectortype_cache[typ][len]
  end

  if not istype(typ) or not typ:isscalar() then
    error("invalid type argument to vector type constructor "..
          "(is this a terra type?)", 2)
  end


  local vt                = NewType("vector")
  vt.N                    = len
  vt.valsize              = len
  vt.type                 = typ
  local ttype             = typ:terratype()
  local struct_name       = "vector_" .. tostring(ttype) ..
                                  "_" .. tostring(vt.N)
  vt._terra_type          = terralib.types.newstruct(struct_name)
  vt._terra_type.entries  = {{ field='d', type=ttype[len] }}

  -- cache
  if not vectortype_cache[typ] then vectortype_cache[typ] = {} end
  vectortype_cache[typ][len] = vt
  return vt
end

local matrixtype_cache = {}
local function matrixType (typ, nrow, ncol)
  -- Ebb may not be fully loaded yet, so...
  if T.is_relation and T.is_relation(typ) then typ = T.key(typ) end
  -- memoize
  local lookup  = matrixtype_cache[typ]
  lookup        = lookup and lookup[nrow]
  lookup        = lookup and lookup[ncol]
  if lookup then return lookup end

  if not T.istype(typ) or not typ:isscalar() then
    error("invalid type argument to small matrix type constructor "..
          "(is this a terra type?)", 2)
  end

  local smt               = NewType("matrix")
  smt.Nrow                = nrow
  smt.Ncol                = ncol
  smt.valsize             = nrow*ncol
  smt.type                = typ
  local ttype             = typ:terratype()
  local struct_name       = "matrix_" .. tostring(ttype) ..
                                  "_" .. tostring(nrow) ..
                                  "_" .. tostring(ncol)
  smt._terra_type         = terralib.types.newstruct(struct_name)
  smt._terra_type.entries = {{ field='d', type=ttype[smt.Ncol][smt.Nrow] }}

  if not matrixtype_cache[typ]  then matrixtype_cache[typ] = {} end
  local cache   = matrixtype_cache[typ]
  if not cache[nrow]            then cache[nrow] = {}           end
  cache         = cache[nrow]
  cache[ncol]   = smt
  return smt
end

-------------------------------------------------------------------------------

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
    if not T.istype(typ) or
       not (typ:isvalue() or typ:iskey())
    then
      error('invalid argument to record type constructor:\n'..
            '  table values must be valid types for fields, not '..
            tostring(typ), 2)
    end
    unique_str = unique_str .. tostring(name) .. '=' .. tostring(typ) .. ','
  end
  unique_str = unique_str .. '}'

  if not record_types[unique_str] then
    local rt = NewType("record")
    rt.rec = rec
    record_types[unique_str] = rt
  end
  return record_types[unique_str]
end

local internal_cache = {}
local function internalType(obj)
  if internal_cache[obj] then return internal_cache[obj] end

  local newtyp          = NewType('internal')
  newtyp.value          = obj
  internal_cache[obj]   = newtyp
  return newtyp
end

--we don't bother to de-duplicate query types
--for simplicity and since since queries are not compared to each other
local function queryType(relation,projections)
  local t = NewType("query")
  t.relation = relation
  t.projections = projections
  return t
end

-------------------------------------------------------------------------------

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
  return terra( [key] ) : int64  return exp  end
end
local function legion_terra_lin_gen(keytyp)
  local key     = symbol(keytyp)
  local strides = symbol(LW.legion_byte_offset_t[#keytyp.entries])
  local exp     = `key.a0 * strides[0].offset
  for k=2,#keytyp.entries do
    exp = `[exp] + strides[k-1].offset * key.['a'..tostring(k-1)]
  end
  return terra( [key], [strides] ) : int64  return exp  end
end
local function linearize_strided_gen(keytyp)
  local key     = symbol(keytyp)
  local strides = symbol(int64[#keytyp.entries])
  local exp     = `key.a0 * strides[0]
  for k=2,#keytyp.entries do
    exp = `[exp] + key.['a'..tostring(k-1)] * strides[k-1]
  end
  -- TODO: THIS IS A HACK TO AVOID LLVM IR SHIPPING ERRORS
  --        Might be a good idea to figure out how to simplify the
  --        complexity of generated code here in the future.
  --return terra( [key], [strides] ) : int64  return exp  end
  return macro(function(k, s)
    return quote
      var [key] = k
      var [strides] = s
    in exp end
  end)
end
local function legion_domain_point_gen(keytyp)
  local key  = symbol(keytyp)
  local dims = #keytyp.entries
  if dims == 2 then return terra([key]) : LW.legion_domain_point_t
    return LW.legion_domain_point_t {
      dim = 2,
      point_data = arrayof(LW.coord_t,
        [LW.coord_t](key.a0), [LW.coord_t](key.a1), 0)
    } end
  elseif dims == 3 then return terra([key]) : LW.legion_domain_point_t
    return LW.legion_domain_point_t {
      dim = 3,
      point_data = arrayof(LW.coord_t,
        [LW.coord_t](key.a0), [LW.coord_t](key.a1), [LW.coord_t](key.a2))
    } end
  else return macro(function()
      error('INTERNAL :DomainPoint() undefined for key of dimension '..dims)
    end)
  end
end

local function dim_to_bits(n_dim)
  if      n_dim < 128         then  return 8
  elseif  n_dim < 32768       then  return 16
  elseif  n_dim < 2147483648  then  return 32
                              else  return 64 end
end

local function checkrelation(relation)
  if not T.is_relation(relation) then
    error("invalid argument to type constructor."..
          "A relation must be provided", 3)
  end
end

local keytype_cache = {}
local function keyType(relation)
  if keytype_cache[relation] then return keytype_cache[relation] end

  checkrelation(relation)

  -- collect information
  local dims        = relation:Dims()
  local strides     = relation:_INTERNAL_Strides()
  local dimbits, dimtyps = {}, {}
  local name        = 'key'
  if relation:isElastic() then
    dimbits = {64}
    dimtyps = {int64}
    name    = 'key_64'
  else
    for i,d in ipairs(dims) do
      dimbits[i]  = dim_to_bits(d)
      dimtyps[i]  = assert(_G['int'..tostring(dimbits[i])])
      name        = name .. '_' .. tostring(dimbits[i])
    end
  end
  name              = name..relation:Name()

  -- create the type and the corresponding struct
  local ktyp        = NewType("key")
  ktyp.relation     = relation
  ktyp.ndims        = #dims
  local tstruct     = terralib.types.newstruct(name)
  ktyp._terra_type  = tstruct --get_physical_key_type(relation)
  for i,typ in ipairs(dimtyps) do
    table.insert(tstruct.entries, { field='a'..tostring(i-1), type=typ })
  end
  tstruct:complete()

  -- Install methods
  tstruct.methods.luaLinearize            = lua_lin_gen(strides)
  tstruct.methods.terraLinearize          = terra_lin_gen(tstruct, strides)
  tstruct.methods.stridedLinearize        = linearize_strided_gen(tstruct)
  if use_legion then
    tstruct.methods.legionTerraLinearize  = legion_terra_lin_gen(tstruct)
    tstruct.methods.domainPoint           = legion_domain_point_gen(tstruct)
  end
  -- add equality / inequality tests
  tstruct.metamethods.__eq = macro(function(lhs,rhs)
    local exp = `lhs.a0 == rhs.a0
    for k=2,#dims do
      local astr = 'a'..tostring(k-1)
      exp = `exp and lhs.[astr] == rhs.[astr]
    end
    return exp
  end)
  tstruct.metamethods.__ne = macro(function(lhs,rhs)
    return `not lhs == rhs
  end)


  keytype_cache[relation] = ktyp
  return ktyp
end

-------------------------------------------------------------------------------
-- In Summary of the Constructors

-- Complex type constructors
T.vector        = vectorType
T.matrix        = matrixType
T.key           = keyType
T.record        = recordType
T.internal      = internalType
T.query         = queryType
T.error         = NewType("error")


-------------------------------------------------------------------------------
--[[ Basic Type Methods                                                    ]]--
-------------------------------------------------------------------------------
-- There are 6 basic kinds of types:
function Type:isprimitive()
  return self.kind == "primitive"
end
function Type:isscalarkey()
  return self.kind == "key"
end
function Type:isscalar()
  return self:isprimitive() or self:isscalarkey()
end
function Type:isvector()
  return self.kind == "vector"
end
function Type:ismatrix()
  return self.kind == "matrix"
end
function Type:isinternal()
  return self.kind == "internal"
end
function Type:iserror()
  return self.kind == "error"
end
function Type:isquery()
  return self.kind == "query"
end
function Type:isrecord()
  return self.kind == "record"
end

-- These types represent Ebb values (not keys though)
function Type:isvalue()
  if self:isvector() or self:ismatrix() then
    return self.type:isprimitive()
  else
    return self:isprimitive()
  end
end

-- These are types that are valid to use for a field
function Type:isfieldvalue()
  return self:isscalar() or self:isvector() or self:ismatrix()
end


-------------------------------------------------------------------------------
--[[ Primitive/Vector Type Methods                                         ]]--
-------------------------------------------------------------------------------

-- of type integer or multiple integers
function Type:isintegral()
  return self:isvalue() and self:terrabasetype():isintegral()
end

function Type:isnumeric()
  return self:isvalue() and self:terrabasetype():isarithmetic()
end

function Type:islogical()
  return self:isvalue() and self:terrabasetype() == bool
end

function Type:iskey()
  return self:isfieldvalue() and self:basetype():isscalarkey()
end

-------------------------------------------------------------------------------
--[[ Methods for computing terra or runtime types                          ]]--
-------------------------------------------------------------------------------

 
function Type:basetype()
  if self:isvector()      then return self.type end
  if self:ismatrix()      then return self.type end
  if self:isscalar()      then return self end
  error("baseType not implemented for " .. tostring(self), 2)
end

local struct emptyStruct {}
local struct  QueryType {
    start : int64;
    finish : int64;
}
function Type:terratype()
  if     self:isprimitive()   then return self._terra_type
  elseif self:isvector()      then return self._terra_type
  elseif self:ismatrix()      then return self._terra_type
  elseif self:iskey()         then return self._terra_type
  elseif self:isquery()       then return QueryType
  elseif self:isinternal()    then return emptyStruct
  elseif self:iserror()       then return emptyStruct
  end
  error("terraType method not implemented for type " .. tostring(self), 2)
end

function Type:terrabasetype()
    return self:basetype():terratype()
end


-------------------------------------------------------------------------------
--[[ Stringify types                                                       ]]--
-------------------------------------------------------------------------------
function Type:__tostring()
  if     self:isprimitive()   then return self.name
  elseif self:isvector()      then return 'Vector('..tostring(self.type)..
                                          ','..tostring(self.N)..')'
  elseif self:ismatrix()      then return 'Matrix('..
                                          tostring(self.type)..','..
                                          tostring(self.Nrow)..','..
                                          tostring(self.Ncol)..')'
  elseif self:isscalarkey()   then return 'Key('..self.relation:Name()..')'
  elseif self:isrecord()      then
    local str = 'Record({ '
    local first_pair = true
    for name, typ in pairs_sorted(self.rec) do
      if first_pair then first_pair = false
      else str = str .. ', ' end
      str = str .. name .. '=' .. tostring(typ)
    end
    str = str .. ' })'
    return str
  elseif self:isquery()     then return 'Query('..self.relation:Name()..').'
                                        ..table.concat(self.projections,'.') 
  elseif self:isinternal()  then return 'Internal('..tostring(self.value)..')'
  elseif self:iserror()     then return 'error'
  end
  error('toString method not implemented for this type!', 2)
end

-------------------------------------------------------------------------------
--[[ DLD types                                                             ]]--
-------------------------------------------------------------------------------

local DLD_prim_translate = {
  [int]     = DLD.SINT_32,
  [uint]    = DLD.UINT_32,
  [uint64]  = DLD.UINT_64,
  [bool]    = DLD.UINT_8,
  [float]   = DLD.FLOAT,
  [double]  = DLD.DOUBLE,
}
-- sanity check sizes
assert(sizeof(int) == 4)
assert(sizeof(uint) == 4)
assert(sizeof(uint64) == 8)
assert(sizeof(bool) == 1)
function Type:DLDEnum()
  if self:isprimitive() then
    local lookup = assert( DLD_prim_translate[self._terra_type] )
    return lookup
  elseif self:isscalarkey() then
    local dims    = self.relation:Dims()
    local name    = 'KEY'
    for _,d in ipairs(dims) do name = name..'_'..tostring(dim_to_bits(d)) end
    return DLD[name]
  else
    error('cannot convert more complex type '..tostring(self)..
          ' to a DLD Enum type', 2)
  end
end

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
  for j,pp in ipairs(primitives) do prim_lessthan[p][pp] = T.error end
  -- diagonal is always ok
  prim_lessthan[p][p] = p
end
prim_lessthan[T.float][T.double] = T.double
prim_lessthan[T.int][T.double]   = T.double
prim_lessthan[T.int][T.float]    = T.float -- WARNING: LOSES PRECISION
prim_lessthan[T.int][T.uint64]   = T.uint64

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
    if qp ~= T.error then val = qp end
    -- and assign that as the join value
    prim_join[p][q] = val
  end
end


-- Coercion defines a partial order.  This function defines that order
function Type:isCoercableTo(target)
  -- make sure we have two types
  if not T.istype(target) then return false end

  -- identity relationship preserved
  -- ensures reflexivity of the relation
  if self == target then return true end

  -- Only numeric values are coercable otherwise...
  if self:isnumeric() and target:isnumeric() then
    local source = self

    -- If we have matching dimension vectors, then delegate the
    -- decision to the base types
    if source:isvector() and target:isvector() and source.N == target.N then
      source = source:basetype()
      target = target:basetype()
    end

    -- similarly, we can unpack matching dimension smallmatrices
    if source:ismatrix() and target:ismatrix() and
       source.Nrow == target.Nrow and source.Ncol == target.Ncol
    then
      source = source:basetype()
      target = target:basetype()
    end

    -- appeal to the lessthan table for primitive types
    if source:isprimitive() and target:isprimitive() then
      local top = prim_lessthan[source][target]
      if top ~= T.error then return true end
    end
  end

  -- In all other cases, coercion fails
  return false
end

-- helpers
local function vec_join(ltype, rtype, N)
    local btype = prim_join[ltype:basetype()][rtype:basetype()]
    if btype == T.error then return T.error
                        else return T.vector(btype, N) end
end

local function mat_join(ltype, rtype, N, M)
  local btype = prim_join[ltype:basetype()][rtype:basetype()]
  if btype == T.error then return T.error
                      else return T.matrix(btype,N,M) end
end

local function type_join(ltype, rtype)
  -- matching case
  if ltype == rtype then return ltype end

  -- outside the matching case, both values must be numeric
  -- We allow two cases of vector/matrix dimensioning:
  --    1. both lhs, rhs vectors or matrices of matching dimension
  --    2. one of lhs, rhs vector/matrix and other scalar
  if ltype:isnumeric() and rtype:isnumeric() then

    -- base-type-join
    local base_join = prim_join[ltype:basetype()][rtype:basetype()]
    if base_join == T.error then return T.error end

    -- primitive joins
    if ltype:isprimitive() and rtype:isprimitive() then
      return base_join

    -- vector joins
    elseif ltype:isvector() and rtype:isvector() then
      if ltype.N == rtype.N then
        return T.vector(base_join, ltype.N) end

    elseif ltype:isvector() and rtype:isprimitive() then
      return T.vector(base_join, ltype.N)

    elseif ltype:isprimitive() and rtype:isvector() then
      return T.vector(base_join, rtype.N)

    -- matrix joins
    elseif ltype:ismatrix() and rtype:ismatrix() then
      if ltype.Nrow == rtype.Nrow or ltype.Ncol == rtype.Ncol then
        return T.matrix(base_join, ltype.Nrow, ltype.Ncol) end

    elseif ltype:ismatrix() and rtype:isprimitive() then
      return T.matrix(base_join, ltype.Nrow, ltype.Ncol)

    elseif ltype:isprimitive() and rtype:ismatrix() then
      return T.matrix(base_join, rtype.Nrow, rtype.Ncol)

    end
  end
  
  -- default is to error
  return T.error
end


-------------------------------------------------------------------------------
--[[ Lua & Terra Interoperation                                            ]]--
-------------------------------------------------------------------------------

local function luaValConformsToType (luaval, tp)
  -- primitives
  if tp:isprimitive() then
    return (tp:isnumeric() and type(luaval) == 'number') or
           (tp:islogical() and type(luaval) == 'boolean')

  -- keys
  elseif tp:isscalarkey() then
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
  elseif tp:isvector()  then
    if type(luaval) ~= 'table' or #luaval ~= tp.N then return false end
    -- make sure each element conforms to the vector data type
    for i = 1, #luaval do
      if not luaValConformsToType(luaval[i], tp:basetype()) then
        return false
      end
    end
    -- if we made it here, everything checked out
    return true

  elseif tp:ismatrix() then
    if type(luaval) ~= 'table' or #luaval ~= tp.Nrow then return false end
    -- check each row / matrix value
    for r = 1, #luaval do
      local row = luaval[r]
      if type(row) ~= 'table' or #row ~= tp.Ncol then return false end
      for c = 1, #row do
        if not luaValConformsToType(row[c], tp:basetype()) then
          return false
        end
      end
    end
    -- if we made it here, everything checked out
    return true

  end
  return false
end

local function luaToEbbVal (luaval, typ)
  if typ:isprimitive() then
    return luaval

  elseif typ:isscalarkey() then
    if typ.ndims == 1 then
      return terralib.new(typ:terratype(), { luaval })
    else
      return terralib.new(typ:terratype(), luaval)
    end

  elseif typ:isvector() then
    local btyp      = typ:basetype()
    local terraval = terralib.new(typ:terratype())
    for i=1,typ.N do terraval.d[i-1] = luaToEbbVal(luaval[i], btyp) end
    return terraval

  elseif typ:ismatrix() then
    local btyp      = typ:basetype()
    local terraval = terralib.new(typ:terratype())
    for r=1,typ.Nrow do for c=1,typ.Ncol do
      terraval.d[r-1][c-1] = luaToEbbVal(luaval[r][c], btyp)
    end end
    return terraval

  else
    error('INTERNAL: Should not try to convert lua values to this type: '..
          tostring(typ))
  end
end

local function ebbToLuaVal(lzval, typ)
  if typ:isprimitive() then
    if typ:isnumeric() then
      return tonumber(lzval)
    elseif typ:islogical() then
      if type(lzval) == 'cdata' then
        return not (lzval == 0)
      else
        return lzval
      end
    end

  elseif typ:isscalarkey() then
    if typ.ndims == 1 then
      return tonumber(lzval.a0)
    elseif typ.ndims == 2 then
      return { tonumber(lzval.a0), tonumber(lzval.a1) }
    elseif typ.ndims == 3 then
      return { tonumber(lzval.a0), tonumber(lzval.a1), tonumber(lzval.a2) }
    else
      error('INTERNAL: Cannot have > 3 dimensional keys')
    end

  elseif typ:isvector() then
    local vec = {}
    for i=1,typ.N do
      vec[i] = ebbToLuaVal(lzval.d[i-1], typ:basetype())
    end
    return vec

  elseif typ:ismatrix() then
    local mat = {}
    for r=1,typ.Nrow do
      mat[r] = {}
      for c=1,typ.Ncol do
        mat[r][c] = ebbToLuaVal(lzval.d[r-1][c-1], typ:basetype())
      end
    end
    return mat

  else
    error('INTERNAL: Should not try to convert lua values from this type: '..
          tostring(typ))
  end
end

-- converts a terra vector or primitive type into an ebb type
local function terraToEbbType (tp)
  -- return primitive type
  local typ = terraprimitive_to_ebb[tp]
  if typ then return typ end
  
  -- return vector type  (WARNING: different backing for types now...)
  --if tp:isvector() then
  --  local p = terraToEbbType(tp.type)
  --  return p and T.vector(p,tp.N)
  --end
  
  return nil
end


-------------------------------------------------------------------------------
--[[ type aliases                                                          ]]--
-------------------------------------------------------------------------------
for n=2,4 do
  local vecname = 'vec'..tostring(n)
  T[vecname..'i'] = T.vector(T.int, n)
  T[vecname..'f'] = T.vector(T.float, n)
  T[vecname..'d'] = T.vector(T.double, n)
  T[vecname..'b'] = T.vector(T.bool, n)

  for m=2,4 do
    local matname = 'mat'..tostring(n)..'x'..tostring(m)
    T[matname..'i'] = T.matrix(T.int, n, m)
    T[matname..'f'] = T.matrix(T.float, n, m)
    T[matname..'d'] = T.matrix(T.double, n, m)
    T[matname..'b'] = T.matrix(T.bool, n, m)
  end

  -- alias square matrices
  local shortname = 'mat'..tostring(n)
  local fullname  = 'mat'..tostring(n)..'x'..tostring(n)
  T[shortname..'i'] = T[fullname..'i']
  T[shortname..'f'] = T[fullname..'f']
  T[shortname..'d'] = T[fullname..'d']
  T[shortname..'b'] = T[fullname..'b']
end

-------------------------------------------------------------------------------
--[[ export type api                                                       ]]--
-------------------------------------------------------------------------------
T.type_join             = type_join
T.luaValConformsToType  = luaValConformsToType
T.luaToEbbVal           = luaToEbbVal
T.ebbToLuaVal           = ebbToLuaVal

T.terraToEbbType        = terraToEbbType
T.Type = Type

--return T
