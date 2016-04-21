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

-- file/module namespace table
local R = {}
package.loaded["ebb.src.relations"] = R

local use_legion = not not rawget(_G, '_legion_env')
local use_single = not use_legion

local Pre   = require "ebb.src.prelude"
local T     = require "ebb.src.types"
local C     = require "ebb.src.c"
local F     = require "ebb.src.functions"

local DLD = require "ebb.lib.dld"
local DLDiter = require 'ebb.src.dlditer'

local uint64T   = T.uint64
local boolT     = T.bool

local keyT      = T.key
local recordT   = T.record

local CPU       = Pre.CPU
local GPU       = Pre.GPU

local is_macro      = Pre.is_macro
local is_function   = F.is_function


local PN = require "ebb.lib.pathname"

local rawdata = require('ebb.src.rawdata')
local DynamicArray = use_single and rawdata.DynamicArray
local DataArray    = use_single and rawdata.DataArray
local LW = use_legion and require "ebb.src.legionwrap"

local P
local use_partitioning
if use_legion then
  run_config = rawget(_G, '_run_config')
  use_partitioning = run_config.use_partitioning
  if use_partitioning then
    P     = require "ebb.src.partitions"
  end
end

local valid_name_err_msg_base =
  "must be valid Lua Identifiers: a letter or underscore,"..
  " followed by zero or more underscores, letters, and/or numbers"
local valid_name_err_msg = {
  relation = "Relation names "  .. valid_name_err_msg_base,
  field    = "Field names "     .. valid_name_err_msg_base,
  subset   = "Subset names "    .. valid_name_err_msg_base
}
local function is_valid_lua_identifier(name)
  if type(name) ~= 'string' then return false end

  -- regex for valid LUA identifiers
  if not name:match('^[_%a][_%w]*$') then return false end

  return true
end

local function linid(ids,dims)
  if #dims == 1 then
    if type(ids) == 'number' then return ids
                             else return ids[1] end
  elseif #dims == 2 then return ids[1] + dims[1] * ids[2]
  elseif #dims == 3 then return ids[1] + dims[1] * (ids[2] + dims[2]*ids[3])
  else error('INTERNAL > 3 dimensional address???') end
end




-------------------------------------------------------------------------------
-------------------------------------------------------------------------------


local Relation    = {}
Relation.__index  = Relation
R.Relation        = Relation
local function is_relation(obj) return getmetatable(obj) == Relation end
R.is_relation     = is_relation


local Field       = {}
Field.__index     = Field
R.Field           = Field
local function is_field(obj) return getmetatable(obj) == Field end
R.is_field        = is_field

local CreateField


local Subset      = {}
Subset.__index    = Subset
R.Subset          = Subset
local function is_subset(obj) return getmetatable(obj) == Subset end
R.is_subset       = is_subset


local Index       = {}
Index.__index     = Index
local function is_index(obj) return getmetatable(obj) == Index end


-------------------------------------------------------------------------------
-------------------------------------------------------------------------------
--[[  Relation methods                                                     ]]--
-------------------------------------------------------------------------------
-------------------------------------------------------------------------------


-- A Relation can be in at most one of the following MODES
--    PLAIN
--    GROUPED (has been sorted for reference)
--    GRID
--    ELASTIC (can insert/delete)
function Relation:isPlain()       return self._mode == 'PLAIN'      end
function Relation:isGrouped()     return self._mode == 'GROUPED'    end
function Relation:isGrid()        return self._mode == 'GRID'       end
function Relation:isElastic()     return self._mode == 'ELASTIC'    end

function Relation:isFragmented()  return self._is_fragmented end

-- Create a generic relation
-- local myrel = L.NewRelation {
--   name = 'myrel',
--   mode = 'PLAIN',
--  [size = 35,]        -- IF mode ~= 'GRID'
--  [dims = {45,90}, ]  -- IF mode == 'GRID'
-- }
local relation_uid = 0
function R.NewRelation(params)
  -- CHECK the parameters coming in
  if type(params) ~= 'table' then
    error("NewRelation() expects a table of named arguments", 2)
  elseif type(params.name) ~= 'string' then
    error("NewRelation() expects 'name' string argument", 2)
  end
  if not is_valid_lua_identifier(params.name) then
    error(valid_name_err_msg.relation, 2)
  end
  local mode = params.mode or 'PLAIN'
  if not params.mode and params.dims then mode = 'GRID' end
  if mode ~= 'PLAIN' and mode ~= 'GRID'  and mode ~= 'ELASTIC' then
    error("NewRelation(): Bad 'mode' argument.  Was expecting\n"..
          "  PLAIN, GRID, or ELASTIC", 2)
  end
  if mode == 'GRID' then
    if type(params.dims) ~= 'table' or
       (#params.dims ~= 2 and #params.dims ~= 3)
    then
      error("NewRelation(): Grids must specify 'dim' argument; "..
            "a table of 2 to 3 numbers specifying grid size", 2)
    end
    if params.periodic then
      if type(params.periodic) ~= 'table' then
        error("NewRelation(): 'periodic' argument must be a list", 2)
      elseif #params.periodic ~= #params.dims then
        error("NewRelation(): periodicity is specified for "..
              tostring(#params.periodic).." dimensions; does not match "..
              tostring(#params.dims).." dimensions specified", 2)
      end
    end
  else
    if type(params.size) ~= 'number' then
      error("NewRelation() expects 'size' numeric argument", 2)
    end
  end

  -- CONSTRUCT and return the relation
  local rel = setmetatable( {
    _name      = params.name,
    _mode      = mode,
    _uid       = relation_uid,

    _fields    = terralib.newlist(),
    _subsets   = terralib.newlist(),
    _macros    = terralib.newlist(),
    _functions = terralib.newlist(),

    _incoming_refs = {}, -- used for walking reference graph
    _disjoint_partition = nil
  },
  Relation)
  relation_uid = relation_uid + 1 -- increment unique id counter

  -- store mode dependent values
  local size = params.size
  if mode == 'GRID' then
    size = 1
    rawset(rel, '_dims', {})
    rawset(rel, '_strides', {})
    rawset(rel, '_periodic', {})
    for i,n in ipairs(params.dims) do
      rel._dims[i]    = n
      rel._strides[i] = size
      size            = size * n
      if params.periodic and params.periodic[i] then rel._periodic = true
                                                else rel._periodic = false end
    end
  end
  rawset(rel, '_concrete_size', size)
  rawset(rel, '_logical_size',  size)
  if rel:isElastic() then
    rawset(rel, '_is_fragmented', false)
  end

  -- SINGLE vs. LEGION
  if use_single then
    -- TODO: Remove the _is_live_mask for inelastic relations
    -- create a mask to track which rows are live.
    rawset(rel, '_is_live_mask', CreateField(rel, '_is_live_mask', boolT))
    rel._is_live_mask:Load(true)

  elseif use_legion then
    -- create a logical region.
    if mode == 'GRID' then
      rawset(rel, '_logical_region_wrapper', LW.NewGridLogicalRegion {
        relation = rel,
        dims     = rel._dims,
      })
    else
      rawset(rel, '_logical_region_wrapper', LW.NewLogicalRegion {
        relation = rel,
        n_rows   = size,
      })
    end
  end

  return rel
end

function Relation:_INTERNAL_UID() -- Why is this even necessary?
  return self._uid
end
function Relation:Size()
  return self._logical_size
end
function Relation:ConcreteSize()
  return self._concrete_size
end
function Relation:Name()
  return self._name
end
function Relation:Dims()
  if not self:isGrid() then
    return { self:Size() }
  end

  local dimret = {}
  for i,n in ipairs(self._dims) do dimret[i] = n end
  return dimret
end
function Relation:_INTERNAL_Strides()
  if not self:isGrid() then
    return { 1 }
  end
  local dimstrides = {}
  for i,n in ipairs(self._strides) do dimstrides[i] = n end
  return dimstrides
end
function Relation:GroupedKeyField()
  if not self:isGrouped() then return nil
                          else return self._grouped_field end
end
function Relation:_INTERNAL_GroupedOffset()
  if not self:isGrouped() then return nil
                          else return self._grouped_offset end
end
function Relation:_INTERNAL_GroupedLength()
  if not self:isGrouped() then return nil
                          else return self._grouped_length end
end
function Relation:Periodic()
  if not self:isGrid() then return { false } end
  local wraps = {}
  for i,p in ipairs(self._dims) do wraps[i] = p end
  return wraps
end

function Relation:foreach(user_func, ...)
  if not is_function(user_func) then
    error('foreach(): expects an ebb function as the first argument', 2)
  end
  user_func:_doForEach(self, ...)
end

function Relation:hasSubsets()
  return #self._subsets ~= 0
end

-- returns a record type
function Relation:StructuralType()
  local rec = {}
  for _, field in ipairs(self._fields) do
    rec[field._name] = field._type
  end
  local typ = recordT(rec)
  return typ
end

-- prevent user from modifying the lua table
function Relation:__newindex(fieldname,value)
  error("Cannot assign members to Relation object "..
      "(did you mean to call relation:New...?)", 2)
end

local FieldDispatcher     = {}
FieldDispatcher.__index   = FieldDispatcher
R.FieldDispatcher         = FieldDispatcher
local function NewFieldDispatcher()
  return setmetatable({
    _reader   = nil,
    _writer   = nil,
    _reducers = {},
  }, FieldDispatcher)
end
local function isfielddispatcher(obj)
  return getmetatable(obj) == FieldDispatcher
end
R.isfielddispatcher = isfielddispatcher

function Relation:NewFieldMacro (name, macro)
  if not name or type(name) ~= "string" then
    error("NewFieldMacro() expects a string as the first argument", 2)
  end
  if not is_valid_lua_identifier(name) then
    error(valid_name_err_msg.field, 2)
  end
  if self[name] then
    error("Cannot create a new field-macro with name '"..name.."'  "..
          "That name is already being used.", 2)
  end

  if not is_macro(macro) then
    error("NewFieldMacro() expects a Macro as the 2nd argument", 2)
  end

  rawset(self, name, macro)
  self._macros:insert(macro)
  return macro
end

local function getFieldDispatcher(rel, fname, ufunc)
  if not fname or type(fname) ~= "string" then
    error("NewField*Function() expects a string as the first argument", 3)
  end
  if not is_valid_lua_identifier(fname) then
    error(valid_name_err_msg.field, 3)
  end
  if not is_function(ufunc) then
    error("NewField*Function() expects an Ebb Function "..
          "as the last argument", 3)
  end

  local lookup = rel[fname]
  if lookup and isfielddispatcher(lookup) then return lookup
  elseif lookup then
    error("Cannot create a new field-function with name '"..fname.."'  "..
          "That name is already being used.", 3)
  end

  rawset(rel, fname, NewFieldDispatcher())
  return rel[fname]
end

function Relation:NewFieldReadFunction(fname, userfunc)
  local dispatch = getFieldDispatcher(self, fname, userfunc)
  if dispatch._reader then
    error("NewFieldReadFunction() error: function already assigned.", 2)
  end
  dispatch._reader = userfunc
  self._functions:insert(userfunc)
  return userfunc
end

function Relation:NewFieldWriteFunction(fname, userfunc)
  local dispatch = getFieldDispatcher(self, fname, userfunc)
  if dispatch._writer then
    error("NewFieldWriteFunction() error: function already assigned.", 2)
  end
  dispatch._writer = userfunc
  self._functions:insert(userfunc)
  return userfunc
end

local redops = {
  ['+'] = true,
  ['-'] = true,
  ['*'] = true,
  ['max'] = true,
  ['min'] = true,
}
function Relation:NewFieldReduceFunction(fname, op, userfunc)
  local dispatch = getFieldDispatcher(self, fname, userfunc)
  if not redops[op] then
    error("NewFieldReduceFunction() expects a reduction operator as the "..
          "second argument.", 2)
  end
  if dispatch._reducers[op] then
    error("NewFieldReduceFunction() error: '"..op.."' "..
          "function already assigned.", 2)
  end
  dispatch._reducers[op] = userfunc
  self._functions:insert(userfunc)
  return userfunc
end


local terra initShuffleArray( a : &uint64, n : uint64 )
  for i=0,n do a[i] = i end
end

local keysort_cache = {}
local function gen_keysort( keytyp )
  if keysort_cache[keytyp] then return keysort_cache[keytyp] end

  local ttyp = keytyp:terratype()
  local terra checksorted( keys : &ttyp, n : uint64 )
    for k=0,n-1 do
      if keys[k]:terraLinearize() > keys[k+1]:terraLinearize() then
        return false
      end
    end
    return true
  end
  local terra selectsort( keys:&ttyp, shuffle:&uint64, lo:uint64, hi:uint64 )
    for i=lo,hi do
      var least, least_i = keys[i]:terraLinearize(), i
      for j=i+1,hi+1 do
        var j_lin = keys[j]:terraLinearize()
        if j_lin < least then
          least, least_i = j_lin, j
        end
      end
      -- now swap
      var tmp_k, tmp_s = keys[i], shuffle[i]
      keys[i], shuffle[i] = keys[least_i], shuffle[least_i]
      keys[least_i], shuffle[least_i] = tmp_k, tmp_s
    end
  end
  local terra choosepivot( keys:&ttyp, lo:uint64, hi:uint64 )
    -- median of three
    var mid = (lo + hi) / 2
    var a,b,c = keys[lo]:terraLinearize(),
                keys[hi]:terraLinearize(),
                keys[mid]:terraLinearize()
    var pivot = a
    if a < b then if a < c then if b < c then pivot = b
                                         else pivot = c end
                           else pivot = a end
             else if b < c then if a < c then pivot = a
                                         else pivot = c end
                           else pivot = b end
    end
    return pivot
  end
  local terra partition( keys:&ttyp, shuffle:&uint64, lo:uint64, hi:uint64 )
    var pivot = choosepivot(keys, lo, hi)
    -- do pivoting
    lo, hi = lo - 1, hi + 1
    while true do
      repeat lo = lo + 1 until keys[lo]:terraLinearize() >= pivot
      repeat hi = hi - 1 until keys[hi]:terraLinearize() <= pivot
      if lo < hi then
        var tempkey, tempidx  = keys[lo], shuffle[lo]
        keys[lo], shuffle[lo] = keys[hi], shuffle[hi]
        keys[hi], shuffle[hi] = tempkey,  tempidx
      else return hi,hi+1 end
    end
  end
  local terra quicksort( keys:&ttyp, shuffle:&uint64, lo:uint64, hi:uint64 ):{}
    --if stop - start < 8 then selectsort(keys, shuffle, lo, hi) end
    if lo >= hi then return end
    var mid_lo, mid_hi = partition(keys, shuffle, lo, hi)
    quicksort(keys, shuffle, lo, mid_lo)
    quicksort(keys, shuffle, mid_hi, hi)
  end
  local terra keysort( keys : &ttyp, shuffle : &uint64, n : uint64 )
    if checksorted(keys, n) then return true end
    quicksort(keys, shuffle, 0, n-1)
    return false
  end

  keysort_cache[keytyp] = keysort
  return keysort
end

local shuffle_cache = {}
local function gen_shuffle( ftyp )
  if shuffle_cache[ftyp] then return shuffle_cache[ftyp] end

  local ttyp = ftyp:terratype()
  local terra shufflefunc( vals : &ttyp, shuffle : &uint64, n : uint64 )
    var tmp = [&ttyp](C.malloc(sizeof(ttyp) * n))

    for k=0,n do tmp[k] = vals[ shuffle[k] ] end
    for k=0,n do vals[k] = tmp[k] end

    C.free(tmp)
  end

  shuffle_cache[ftyp] = shufflefunc
  return shufflefunc
end

function Relation:_INTERNAL_SortBy(keyfield)
  assert( is_field(keyfield) and keyfield._owner == self and
          keyfield:Type():isscalarkey(), 'expecting key field' )
  assert( not self:isFragmented(), 'cannot sort fragmented relation')
  assert( #self._subsets == 0, 'TODO: Support sorting relations w/subsets')

  -- First we'll sort the keys and produce the following shuffle index
  local shuffle_array = terralib.cast( &uint64, C.malloc(8 * self:Size()) )
  initShuffleArray(shuffle_array, self:Size())

  -- get access to the key data to sort on
  local keydld = nil
  local keytyp = keyfield:Type()
  local legion_region = nil
  if use_single then
    keydld          = keyfield:GetDLD()
    keydld.address  = keyfield._array:open_readwrite_ptr()
    keydld:setlocation(DLD.CPU)
  elseif use_legion then
    legion_region = LW.NewInlinePhysicalRegion {
      relation  = self,
      fields    = { keyfield },
      privilege = LW.READ_WRITE,
    }
    keydld = legion_region:GetLuaDLDs()[1]
  end

  -- verify that the layout is acceptable
  assert(keydld.version[1] == 1 and keydld.version[2] == 0)
  assert(keydld.type_stride == terralib.sizeof(keytyp:terratype()))
  local ndim, size = #self:Dims(), 1
  while ndim > 0 do -- check for tight packing of grids
    assert(keydld.dim_stride[ndim] == size)
    size = size * keydld.dim_size[ndim]
    ndim = ndim-1
  end

  -- sort keys
  local keysort = gen_keysort(keytyp)
  local addr = terralib.cast( &(keytyp:terratype()), keydld.address )
  local already_sorted = keysort( addr, shuffle_array, self:Size() )

  -- release lock on the key data
  if use_single then
    keyfield._array:close_readwrite_ptr()
  elseif use_legion then
    legion_region:Destroy()
  end

  if already_sorted then return end

  ------------

  -- Now we'll shuffle the other fields

  local function shuffle_field(f)
    local dld = nil
    if use_single then
      dld         = f:GetDLD()
      dld.address = f._array:open_readwrite_ptr()
      dld:setlocation(DLD.CPU)
    elseif use_legion then
      legion_region = LW.NewInlinePhysicalRegion {
        relation  = self,
        fields    = {f},
        privilege = LW.READ_WRITE,
      }
      dld = legion_region:GetLuaDLDs()[1]
    end

    local ftyp = f:Type()
    assert(dld.type_stride == terralib.sizeof(ftyp:terratype()))

    local shufflefunc = gen_shuffle(ftyp)
    local addr = terralib.cast( &(ftyp:terratype()), dld.address )
    shufflefunc( dld.address, shuffle_array, self:Size() )

    if use_single then
      f._array:close_readwrite_ptr()
    elseif use_legion then
      legion_region:Destroy()
    end
  end

  if self._is_live_mask then shuffle_field(self._is_live_mask) end
  for _,f in ipairs(self._fields) do
    if f ~= keyfield then shuffle_field(f) end
  end

  -- and release the shuffle array data
  C.free(shuffle_array)
end

function Relation:GroupBy(keyf_name)
  if self:isGrouped() then
    error("GroupBy(): Relation is already grouped", 2)
  elseif not self:isPlain() then
    error("GroupBy(): Cannot group a relation "..
          "unless it's a PLAIN relation", 2)
  end

  local key_field = type(keyf_name) == 'string' and self[keyf_name]
                                                 or keyf_name
  if not is_field(key_field) or key_field._owner ~= self then
    error("GroupBy(): Could not find a field named '"..
          tostring(keyf_name).."'", 2)
  elseif not key_field._type:isscalarkey() then
    error("GroupBy(): Grouping by non-scalar-key fields is "..
          "prohibited.", 2)
  end

  if key_field._owner:hasSubsets() then
    error("GroupBy(): TODO support grouping a relation that has subsets. "..
          "Please notify the developers if this is important.", 2)
  end

  self:_INTERNAL_SortBy(key_field)

  -- In the below, we use the following convention
  --  SRC is the relation referred to by the key field
  --  DST is 'self' here, the relation which is actively being grouped
  --    In a Where query, a key into the SRC relation is translated
  --    into a sequence of keys into the DST relation
  local srcrel = key_field._type.relation
  local dstrel = self
  local n_src  = srcrel:Size()
  local n_dst  = dstrel:Size()
  local dstname = dstrel:Name()
  local offset_f = CreateField(srcrel, dstname..'_grouped_offset', uint64T)
  local length_f = CreateField(srcrel, dstname..'_grouped_length', uint64T)

  rawset(self,'_grouped_field', key_field)
  rawset(self,'_grouped_offset', offset_f)
  rawset(self,'_grouped_length', length_f)

  if use_single then
    -- NOTE: THIS IMPLEMENTATION HEAVILY ASSUMES THAT A GRID IS LINEARIZED
    -- IN ROW-MAJOR ORDER
    local offptr = offset_f._array:open_write_ptr()
    local lenptr = length_f._array:open_write_ptr()
    local keyptr = key_field._array:open_read_ptr()
      local dims = srcrel:Dims()

      local dst_i, prev_src = 0,0
      for src_i=0,n_src-1 do -- linear scan assumption here
        offptr[src_i] = dst_i -- where to find the first row
        local count = 0
        while dst_i < n_dst do
          local lin_src = keyptr[dst_i]:luaLinearize()
          if lin_src ~= src_i then break end
          if lin_src < prev_src then
            error("GroupBy(): Key field '"..key_field:Name().."' "..
                  "is not sorted.")
          end
          count     = count + 1
          dst_i     = dst_i + 1
          prev_src  = lin_src
        end
        lenptr[src_i] = count -- # of rows
      end
      assert(dst_i == n_dst)
    key_field._array:close_read_ptr()
    length_f._array:close_write_ptr()
    offset_f._array:close_write_ptr()
  elseif use_legion then

    local keyf_list = key_field:Dump({})
    local dims      = srcrel:Dims()

    local src_scanner = LW.NewControlScanner {
      relation       = srcrel,
      fields         = { offset_f, length_f },
      privilege      = LW.WRITE_ONLY
    }
    local dst_i, prev_src = 0,0
    for ids, ptrs in src_scanner:ScanThenClose() do
      local src_i   = linid(ids,dims)
      local offptr  = terralib.cast(&uint64,ptrs[1])
      local lenptr  = terralib.cast(&uint64,ptrs[2])

      offptr[0] = dst_i
      local count   = 0
      while dst_i < n_dst do
        local lin_src = linid(keyf_list[dst_i+1],dims)
        if lin_src ~= src_i then break end
        if lin_src < prev_src then
          error("GroupBy(): Key field '"..key_field:Name().."' "..
                "is not sorted.")
        end
        count     = count + 1
        dst_i     = dst_i + 1
        prev_src  = lin_src
      end
      lenptr[0] = count
    end
    assert(dst_i == n_dst)
    assert(dst_i == n_dst)
  else
    error("INTERNAL: must use either single or legion...")
  end

  
  self._mode = 'GROUPED'
  -- record reference from this relation to the relation it's grouped by
  srcrel._incoming_refs[self] = 'group'
end

function Relation:MoveTo( proc )
  if use_legion then error("MoveTo() unsupported using Legion", 2) end
  if proc ~= CPU and proc ~= GPU then
    error('must specify valid processor to move to', 2)
  end

  self._is_live_mask:MoveTo(proc)
  for _,f in ipairs(self._fields) do f:MoveTo(proc) end
  for _,s in ipairs(self._subsets) do s:MoveTo(proc) end
  if self:isGrouped() then
    self._grouped_offset:MoveTo(proc)
    self._grouped_length:MoveTo(proc)
  end
end


function Relation:Print()
  if use_legion then
    error("print() currently unsupported using Legion", 2)
  end
  print(self._name, "size: ".. tostring(self:Size()),
                    "concrete size: "..tostring(self:ConcreteSize()))
  for i,f in ipairs(self._fields) do
    f:Print()
  end
end


-------------------------------------------------------------------------------
-------------------------------------------------------------------------------
--[[  Indices:                                                             ]]--
-------------------------------------------------------------------------------
-------------------------------------------------------------------------------


function Index.New(params)
  if not is_relation(params.owner) or
     type(params.name) ~= 'string' or
     not (params.size or params.terra_type or params.data)
  then
    error('bad parameters')
  end

  local index = setmetatable({
    _owner = params.owner,
    _name  = params.name,
  }, Index)

  index._array = DynamicArray.New {
    size = params.size or (#params.data),
    type = params.terra_type,
    processor = params.processor or Pre.default_processor,
  }

  if params.data then
    local ptr = index._array:open_write_ptr()
    --index._array:write_ptr(function(ptr)
      for i=1,#params.data do
        for k=1,params.ndims do
          ptr[i-1]['a'..tostring(k-1)] = params.data[i][k]
        end
      end
    --end) -- write_ptr
    index._array:close_write_ptr()
  end

  return index
end

function Index:_Raw_DataPtr()
  return self._array:_raw_ptr()
end
function Index:Size()
  return self._array:size()
end

function Index:Relation()
  return self._owner
end

function Index:ReAllocate(size)
  self._array:resize(size)
end

function Index:MoveTo(proc)
  self._array:moveto(proc)
end

function Index:Release()
  if self._array then
    self._array:free()
    self._array = nil
  end
end


-------------------------------------------------------------------------------
-------------------------------------------------------------------------------
--[[  Subsets:                                                             ]]--
-------------------------------------------------------------------------------
-------------------------------------------------------------------------------


function Subset:foreach(user_func, ...)
  if not is_function(user_func) then
    error('map(): expects an Ebb function as the argument', 2)
  end
  user_func:_doForEach(self, ...)
end

function Subset:Relation()
  return self._owner
end

function Subset:Name()
  return self._name
end

function Subset:FullName()
  return self._owner._name .. '.' .. self._name
end

-- prevent user from modifying the lua table
function Subset:__newindex(name,value)
  error("Cannot assign members to Subset object", 2)
end

function Subset:MoveTo( proc )
  if proc ~= CPU and proc ~= GPU then
    error('must specify valid processor to move to', 2)
  end

  if self._boolmask   then self._boolmask:MoveTo(proc)    end
  if self._index      then self._index:MoveTo(proc)       end
end

function Relation:_INTERNAL_NewSubsetFromLuaFunction (name, predicate)
  if not name or type(name) ~= "string" then
    error("NewSubsetFromFunction() "..
          "expects a string as the first argument", 2)
  end
  if not is_valid_lua_identifier(name) then
    error(valid_name_err_msg.subset, 2)
  end
  if self[name] then
    error("Cannot create a new subset with name '"..name.."'  "..
          "That name is already being used.", 2)
  end

  if type(predicate) ~= 'function' then
    error("NewSubsetFromFunction() expects a predicate "..
          "for determining membership as the second argument", 2)
  end

  -- SIMPLIFYING HACK FOR NOW
  if self:isElastic() then
    error("NewSubsetFromFunction(): "..
          "Subsets of elastic relations are currently unsupported", 2)
  end

  -- setup and install the subset object
  local subset = setmetatable({
    _owner    = self,
    _name     = name,
  }, Subset)
  rawset(self, name, subset)
  self._subsets:insert(subset)

  -- NOW WE DECIDE how to encode the subset
  -- we'll try building a mask and decide between using a mask or index
  local SUBSET_CUTOFF_FRAC = 0.1
  local SUBSET_CUTOFF = SUBSET_CUTOFF_FRAC * self:Size()

  local boolmask  = CreateField(self, name..'_subset_boolmask', boolT)
  local index_tbl = {}
  local subset_size = 0
  local dims = self:Dims()
  boolmask:_INTERNAL_LoadLuaPerElemFunction(function(xi,yi,zi)
    local val = predicate(xi,yi,zi)
    local ids = {xi,yi,zi}
    if val then
      table.insert(index_tbl, ids)
      subset_size = subset_size + 1
    end
    return val
  end)

  if use_legion or subset_size > SUBSET_CUTOFF or self:isGrid() then
  -- USE MASK
    rawset(subset, '_boolmask', boolmask)
  else
  -- USE INDEX
    rawset(subset, '_index', Index.New{
      owner=self,
      terra_type = keyT(self):terratype(),
      ndims=#self:Dims(),
      name=name..'_subset_index',
      data=index_tbl
    })
    boolmask:_INTERNAL_ClearData() -- free memory
  end

  return subset
end

local function is_int(obj)
  return type(obj) == 'number' and obj % 1 == 0
end
local function is_subrectangle(rel, obj)
  local dims = rel:Dims()
  if not terralib.israwlist(obj) or #obj ~= #dims then return false end
  for i,r in ipairs(obj) do
    if not terralib.israwlist(r) or #r ~= 2 then return false end
    if not is_int(r[1]) or not is_int(r[2]) then return false end
    if r[1] < 0 or r[2] < 0 or r[1] >= dims[i] or r[2] >= dims[i] then
      return false
    end
  end
  return true
end

function Relation:_INTERNAL_NewSubsetFromRectangles(name, rectangles)
  if #self:Dims() == 2 then
    return self:_INTERNAL_NewSubsetFromLuaFunction(name, function(xi, yi)
      for _,r in ipairs(rectangles) do
        local xlo, xhi, ylo, yhi = r[1][1], r[1][2], r[2][1], r[2][2]
        if xlo <= xi and xi <= xhi and ylo <= yi and yi <= yhi then
          return true -- found cell inside some rectangle
        end
      end
      return false -- couldn't find cell inside any rectangle
    end)
  else
    assert(#self:Dims() == 3, "grids must be 2 or 3 dimensional")
    return self:_INTERNAL_NewSubsetFromLuaFunction(name, function(xi, yi, zi)
      for _,r in ipairs(rectangles) do
        local xlo, xhi, ylo, yhi, zlo, zhi =
          r[1][1], r[1][2],  r[2][1], r[2][2], r[3][1], r[3][2]
        if xlo <= xi and xi <= xhi and
           ylo <= yi and yi <= yhi and
           zlo <= zi and zi <= zhi
        then
          return true -- found cell inside some rectangle
        end
      end
      return false -- couldn't find cell inside any rectangle
    end)
  end
end

function Relation:NewSubset( name, arg )
  if not name or type(name) ~= "string" then
    error("NewSubset() expects a string as the first argument", 2) end
  if not is_valid_lua_identifier(name) then
    error(valid_name_err_msg.subset, 2) end
  if self[name] then
    error("Cannot create a new subset with name '"..name.."'  "..
          "That name is already being used.", 2)
  end

  if self:isElastic() then
    error("NewSubset(): "..
          "Subsets of elastic relations are currently unsupported", 2)
  end
  if self:isGrouped() then
    error("NewSubset(): TODO support grouping relations with subsets. "..
          "Notify the developers if this is important to you.", 2)
  end

  if type(arg) == 'table' then
    if self:isGrid() then
      if arg.rectangles then
        if not terralib.israwlist(arg.rectangles) then
          error("NewSubset(): Was expecting 'rectangles' to be a list", 2)
        end
        for i,r in ipairs(arg.rectangles) do
          if not is_subrectangle(self, r) then
            error("NewSubset(): Entry #"..i.." in 'rectangles' list was "..
                  "not a rectangle, specified as a list of "..(#self:Dims())..
                  " range pairs lying inside the grid", 2)
          end
        end
        return self:_INTERNAL_NewSubsetFromRectangles(name, arg.rectangles)
      else -- assume a single rectangle
        if not is_subrectangle(self, arg) then
          error('NewSubset(): Was expecting a rectangle specified as a '..
                'list of '..(#self:Dims())..' range pairs lying inside '..
                'the grid', 2)
        end
        return self:_INTERNAL_NewSubsetFromRectangles(name, { arg })
      end
    end
  end

  -- catch-all
  error('NewSubset(): unrecognized argument type', 2)
end

-------------------------------------------------------------------------------
-------------------------------------------------------------------------------
--[[  Fields:                                                              ]]--
-------------------------------------------------------------------------------
-------------------------------------------------------------------------------


-- Client code should never call this constructor
-- For internal use only.  Does not install on relation...
function CreateField(rel, name, typ)
  local field   = setmetatable({
    _type   = typ,
    _name   = name,
    _owner  = rel,
  }, Field)
  if use_single then
    field:_INTERNAL_Allocate()
  elseif use_legion then
    rawset( field, '_fid',
            rel._logical_region_wrapper:AllocateField(typ:terratype()) )
    rel._logical_region_wrapper:AttachNameToField(field._fid, name)
  end
  return field
end

-- prevent user from modifying the lua table
function Field:__newindex(name,value)
  error("Cannot assign members to Field object", 2)
end

function Field:Name()
  return self._name
end
function Field:FullName()
  return self._owner._name .. '.' .. self._name
end
function Field:Size()
  return self._owner:Size()
end
function Field:ConcreteSize()
  return self._owner:ConcreteSize()
end
function Field:Type()
  return self._type
end
function Field:_Raw_DataPtr()
  if use_legion then error('DataPtr() unsupported using legion') end
  return self._array:_raw_ptr()
end
function Field:Relation()
  return self._owner
end

function Relation:NewField (name, typ)  
  if not name or type(name) ~= "string" then
    error("NewField() expects a string as the first argument", 2)
  end
  if not is_valid_lua_identifier(name) then
    error(valid_name_err_msg.field, 2)
  end
  if self[name] then
    error("Cannot create a new field with name '"..name.."'  "..
          "That name is already being used.", 2)
  end
  
  if is_relation(typ) then
    typ = keyT(typ)
  end
  if not T.istype(typ) or not typ:isfieldvalue() then
    error("NewField() expects an Ebb type or "..
          "relation as the 2nd argument", 2)
  end

  -- prevent the creation of key fields pointing into elastic relations
  if typ:iskey() then
    local rel = typ:basetype().relation
    if rel:isElastic() then
      error("NewField(): Cannot create a key-type field referring to "..
            "an elastic relation", 2)
    end
  end
  if self:isFragmented() then
    error("NewField() cannot be called on a fragmented relation.", 2)
  end

  -- create the field
  local field = CreateField(self, name, typ)
  rawset(self, name, field)
  self._fields:insert(field)

  -- record reverse pointers for key-field references
  if typ:iskey() then
    typ:basetype().relation._incoming_refs[field] = 'key_field'
  end

  return field
end

-- TODO: Hide this function so it's not public
function Field:_INTERNAL_Allocate()
  if use_legion then error('No Allocate() using legion') end
  if not self._array then
    --if self._owner:isElastic() then
      rawset(self, '_array', DynamicArray.New{
        size = self:ConcreteSize(),
        type = self:Type():terratype()
      })
    --else
    --  self._array = DataArray.New {
    --    size = self:ConcreteSize(),
    --    type = self:Type():terratype()
    --  }
    --end
  end
end

-- TODO: Hide this function so it's not public
-- remove allocated data and clear any depedent data, such as indices
function Field:_INTERNAL_ClearData()
  if use_legion then error('No ClearData() using legion') end
  if self._array then
    self._array:free()
    rawset(self, '_array', nil)
  end
  -- clear grouping data if set on this field
  if self._owner:isGrouped() and
     self._owner:GroupedKeyField() == self
  then
    error('UNGROUPING CURRENTLY UNIMPLEMENTED')
  end
end

function Field:MoveTo( proc )
  if use_legion then error('No MoveTo() using legion') end
  if proc ~= CPU and proc ~= GPU then
    error('must specify valid processor to move to', 2)
  end

  self._array:moveto(proc)
end

function Relation:Swap( f1_name, f2_name )
  local f1 = self[f1_name]
  local f2 = self[f2_name]
  if not is_field(f1) then
    error('Could not find a field named "'..f1_name..'"', 2) end
  if not is_field(f2) then
    error('Could not find a field named "'..f2_name..'"', 2) end
  if f1._type ~= f2._type then
    error('Cannot Swap() fields of different type', 2)
  end

  if use_single then
    local tmp = f1._array
    f1._array = f2._array
    f2._array = tmp
  elseif use_legion then
    local region  = self._logical_region_wrapper
    local rhandle = region:get_handle()
    local fid_1   = f1._fid
    local fid_2   = f2._fid
    -- create a temporary Legion field
    local fid_tmp = region:AllocateField(f1._type:terratype())

    LW.CopyField { region = rhandle,  src_fid = fid_1,    dst_fid = fid_tmp }
    LW.CopyField { region = rhandle,  src_fid = fid_2,    dst_fid = fid_1   }
    LW.CopyField { region = rhandle,  src_fid = fid_tmp,  dst_fid = fid_2   }

    -- destroy temporary field
    region:FreeField(fid_tmp)
  end
end

function Relation:Copy( p )
  if type(p) ~= 'table' or not p.from or not p.to then
    error("relation:Copy() should be called using the form\n"..
          "  relation:Copy{from='f1',to='f2'}", 2)
  end
  local from = p.from
  local to   = p.to
  if type(from) == 'string' then from = self[from] end
  if type(to)   == 'string' then to   = self[to]   end
  if not is_field(from) then
    error('Could not find a field named "'..p.from..'"', 2) end
  if not is_field(to) then
    error('Could not find a field named "'..p.to..'"', 2) end
  if not from:Relation() == self then
    error('Field '..from:FullName()..' is not a field of '..
          'Relation '..self:Name(), 2) end
  if not to:Relation() == self then
    error('Field '..to:FullName()..' is not a field of '..
          'Relation '..self:Name(), 2) end
  if from._type ~= to._type then
    error('Cannot Copy() fields of different type', 2)
  end

  if use_single then
    if not from._array then
      error('Cannot Copy() from field with no data', 2) end
    if not to._array then
      to:_INTERNAL_Allocate()
    end
    to._array:copy(from._array)

  elseif use_legion then
    LW.CopyField {
      region  = self._logical_region_wrapper:get_handle(),
      src_fid = from._fid,
      dst_fid = to._fid,
    }
  end
end


-------------------------------------------------------------------------------
-------------------------------------------------------------------------------
--[[  Loading and I/O                                                      ]]--
-------------------------------------------------------------------------------
-------------------------------------------------------------------------------


-------------------------------------------------------------------------------
--[[  Base-Level Relation Memory Access Operations  (Lua and Terra)        ]]--

local mapcount = 0
function Relation:_INTERNAL_MapJointFunction(
  iswrite, isterra, clbk, fields, ...
)
  assert(type(iswrite) == 'boolean')
  assert(type(isterra) == 'boolean')
  if isterra then
    assert(terralib.isfunction(clbk), 'arg should be terra function')
  else
    assert(type(clbk) == 'function', 'arg should be lua function')
  end
  assert(terralib.israwlist(fields), 'arg should be list of fields')
  for _,f in ipairs(fields) do
    assert(is_field(f), 'arg should be list of fields')
    assert(f._owner == self, 'fields should belong to this relation')
  end
  assert(not self:isFragmented(), 'cannot expose a fragmented relation')

  local dld_list = isterra and C.safemalloc(DLD.C_DLD, #fields) or {}
  if use_single then
    for i = 1, #fields do
      local dld     = fields[i]:GetDLD()
      if iswrite then   dld.address   = fields[i]._array:open_write_ptr()
                 else   dld.address   = fields[i]._array:open_read_ptr() end
      dld:setlocation(DLD.CPU)
      if isterra then   dld_list[i-1] = dld:toTerra()
                 else   dld_list[i]   = dld end
    end

    local retvals = { clbk(dld_list, ...) }

    for i = 1, #fields do
      if iswrite then   fields[i]._array:close_write_ptr()
                 else   fields[i]._array:close_read_ptr() end
    end
    return unpack(retvals)

  elseif use_legion then
    -- create a physical mapping and get the DLDs from these
    local region = LW.NewInlinePhysicalRegion {
      relation  = self,
      fields    = fields,
      privilege = iswrite and LW.WRITE_ONLY or LW.READ_ONLY,
    }
    local dld_list = isterra and region:GetTerraDLDs() or region:GetLuaDLDs()

    local retvals = { clbk(dld_list, ...) }

    region:Destroy()
    return unpack(retvals)
  end
end


-------------------------------------------------------------------------------
--[[  Mid-Level Loading Operations (Lua and Terra)                         ]]--

function Field:_INTERNAL_LoadLuaPerElemFunction(clbk)
  local typ       = self:Type()
  local ttyp      = typ:terratype()

  self._owner:_INTERNAL_MapJointFunction(true, false, function(dldlist)
    local dld         = dldlist[1]
    local ptr         = terralib.cast(&ttyp, dld.address)

    DLDiter.luaiter(dld, function(lin, ...)
      local tval  = T.luaToEbbVal( clbk(...), typ )
      ptr[lin]    = tval
    end)
  end, {self})
end

function Field:_INTERNAL_LoadTerraBulkFunction(clbk, ...)
  return self._owner:_INTERNAL_MapJointFunction(true, true, clbk, {self}, ...)
end

function Relation:_INTERNAL_LoadTerraBulkFunction(fields, clbk, ...)
  return self:_INTERNAL_MapJointFunction(true, true, clbk, fields, ...)
end

-- this is broken for unstructured relations with legion
function Field:_INTERNAL_LoadList(tbl)
  local ndims = #self._owner:Dims()
  if ndims == 1 then
    self:_INTERNAL_LoadLuaPerElemFunction(function(i)
      return tbl[i+1]
    end)
  elseif ndims == 2 then
    self:_INTERNAL_LoadLuaPerElemFunction(function(i,j)
      return tbl[j+1][i+1]
    end)
  elseif ndims == 3 then
    self:_INTERNAL_LoadLuaPerElemFunction(function(i,j,k)
      return tbl[k+1][j+1][i+1]
    end)
  else assert(false, 'INTERNAL: bad # dims '..ndims) end
end

function Field:_INTERNAL_LoadConstant(c)
  -- TODO: Convert implementation to Terra
  self:_INTERNAL_LoadLuaPerElemFunction(function()
    return c
  end)
end


-------------------------------------------------------------------------------
--[[  Mid-Level Dumping Operations (Lua and Terra)                         ]]--

function Relation:_INTERNAL_DumpLuaPerElemFunction(fields, clbk)
  --local typ       = self:Type()
  --local ttyp      = typ:terratype()

  self:_INTERNAL_MapJointFunction(false, false, function(dldlist)
    --local dld         = dldlist[1]
    local typs, ptrs = {}, {}
    for i,dld in ipairs(dldlist) do
      typs[i]     = fields[i]:Type()
      local ttyp  = typs[i]:terratype()
      ptrs[i]     = terralib.cast(&ttyp, dld.address)
    end

    -- any of the dlds will do
    DLDiter.luaiter(dldlist[1], function(lin, ...)
      local vals  = {}
      for i,t in ipairs(typs) do
        vals[i] = T.ebbToLuaVal(ptrs[i][lin], t)
      end

      -- ... is ids
      clbk({...}, unpack(vals))
    end)
  end, fields)
end

function Field:_INTERNAL_DumpLuaPerElemFunction(clbk)
  self._owner:_INTERNAL_DumpLuaPerElemFunction({self}, function(ids, val)
    clbk(val, unpack(ids))
  end)
end

function Relation:_INTERNAL_DumpTerraBulkFunction(fields, clbk, ...)
  return self:_INTERNAL_MapJointFunction(false, true, clbk, fields, ...)
end

function Field:_INTERNAL_DumpTerraBulkFunction(clbk, ...)
  return self._owner:_INTERNAL_MapJointFunction(false, true, clbk, {self}, ...)
end

function Field:_INTERNAL_DumpList()
  local result = {}
  local dims = self._owner:Dims()

  if #dims == 1 then
    self:_INTERNAL_DumpLuaPerElemFunction(function(val, i)
      result[i+1] = val
    end)
  elseif #dims == 2 then
    for j=1,dims[2] do result[j] = {} end
    self:_INTERNAL_DumpLuaPerElemFunction(function(val, i,j)
      result[j+1][i+1] = val
    end)
  elseif #dims == 3 then
    for k=1,dims[3] do
      result[k] = {}
      for j=1,dims[2] do result[k][j] = {} end
    end
    self:_INTERNAL_DumpLuaPerElemFunction(function(val, i,j,k)
      result[k+1][j+1][i+1] = val
    end)
  else assert(false, 'INTERNAL: bad # dims '..ndims) end

  return result
end


-------------------------------------------------------------------------------
--[[  Error Checking subroutines                                           ]]--

-- modular error checking
local function ferrprefix(level)
  local blob = debug.getinfo(level)
  local name = type(blob.name) == 'string' and blob.name..': ' or ''
  return name
end
local function argcheck_loadval_type(obj,typ,lvl)
  if not T.luaValConformsToType(obj,typ) then
    lvl = (lvl or 1) + 1
    error(ferrprefix(lvl).."lua value does not conform to type "..
                           tostring(typ), lvl)
  end
end
--local function argcheck_luafunction(obj,lvl)
--  if type(obj) ~= 'function' then
--    lvl = (lvl or 1) + 1
--    error(ferrprefix(lvl)..'Expected Lua function as argument', lvl)
--  end
--end
local function argcheck_list(obj,lvl)
  if not terralib.israwlist(obj) then
    lvl = (lvl or 1) + 1
    error(ferrprefix(lvl)..'Expected list as argument', lvl)
  end
end
local function argcheck_rel_fields(obj,rel,lvl)
  lvl = (lvl or 1)+1
  argcheck_list(obj,lvl+1)
  for i,f in ipairs(obj) do
    if not is_field(f) then
      error(ferrprefix(lvl)..'Expected field at list entry '..i, lvl)
    end
    if not f._owner == rel then
      error(ferrprefix(lvl)..'Expected field to be a member of '..
            rel:Name(),lvl)
    end
  end
end
local function _helper_argcheck_list_dims_err(dims,lvl)
  local dimstr = tostring(dims[1])
  if dims[2] then dimstr = tostring(dims[2])..','..dimstr end
  if dims[3] then dimstr = tostring(dims[3])..','..dimstr end
  dimstr = '{'..dimstr..'}'
  local errmsg = 'Expected argument list to have dimensions '..dimstr
  assert(lvl)
  error(ferrprefix(lvl)..errmsg, lvl)
end
local function argcheck_list_dims_and_type(obj,dims,typ,lvl)
  lvl = (lvl or 1)
  argcheck_list(obj,lvl+1)

  if #dims == 1 then
    if #obj ~= dims[1] then _helper_argcheck_list_dims_err(dims,lvl+1) end
    for i,val in ipairs(obj) do argcheck_loadval_type(val, typ, lvl+1) end

  elseif #dims == 2 then
    if #obj ~= dims[2] then _helper_argcheck_list_dims_err(dims,lvl+1) end
    for k,sub in ipairs(obj) do
      if #sub ~= dims[1] then _helper_argcheck_list_dims_err(dims,lvl+1) end
      for i,val in ipairs(sub) do argcheck_loadval_type(val, typ, lvl+1) end
    end

  elseif #dims == 3 then
    if #obj ~= dims[3] then _helper_argcheck_list_dims_err(dims,lvl+1) end
    for k,sub in ipairs(obj) do
      if #sub ~= dims[2] then _helper_argcheck_list_dims_err(dims,lvl+1) end
      for j,sub2 in ipairs(sub) do
        if #sub2 ~= dims[1] then _helper_argcheck_list_dims_err(dims,lvl+1) end
        for i,val in ipairs(sub2) do argcheck_loadval_type(val, typ, lvl+1) end
      end
    end
  else assert(false, 'INTERNAL BAD DIM') end
end


-------------------------------------------------------------------------------
--[[  Custom Loader/Dumper Interface                                       ]]--

local CustomLoader = {}
CustomLoader.__index = CustomLoader
local CustomDumper = {}
CustomDumper.__index = CustomDumper

local function isloader(obj) return getmetatable(obj) == CustomLoader end
local function isdumper(obj) return getmetatable(obj) == CustomDumper end
R.is_loader = isloader
R.is_dumper = isdumper

function R.NewLoader(loadfunc)
  if type(loadfunc) ~= 'function' then
    error('NewLoader() expects a lua function as the argument', 2)
  end
  return setmetatable({
    _func = loadfunc,
  }, CustomLoader)
end

function R.NewDumper(dumpfunc)
  if type(dumpfunc) ~= 'function' then
    error('NewDumper() expects a lua function as the argument', 2)
  end
  return setmetatable({
    _func = dumpfunc,
  }, CustomDumper)
end

-------------------------------------------------------------------------------
--[[  High-Level Loading and Dumping Operations (Lua and Terra)            ]]--

function Field:Load(arg, ...)
  if self._owner:isFragmented() then
    error('cannot load into fragmented relation', 2)
  end

  -- TODO(Chinmayee): deprecate this
  if type(arg) == 'function' then
    return self:_INTERNAL_LoadLuaPerElemFunction(arg)

  elseif isloader(arg) then
    return arg._func(self, ...)

  elseif isdumper(arg) then
    error("tried to use a dumper object to load.  "..
          "Did you mean to use the corresponding dumper object?", 2)

  elseif  type(arg) == 'cdata' then
    local typ = terralib.typeof(arg)
    if typ:ispointer() then
      error('Auto-Load from memory is no longer supported', 2)
    end
    -- otherwise fall-through to constant loading

  elseif  type(arg) == 'table' then
    -- terra function
    if (terralib.isfunction(arg)) then
      return self:_INTERNAL_LoadTerraBulkFunction(arg, ...)

    -- field
    elseif is_field(arg) then
      if arg._owner ~= self._owner then
        error('Can only load from another field on the same relation', 2)
      end
      if arg:Type() ~= self:Type() then
        error('Can only load from another field with identical type', 2)
      end
      self._owner:Copy { from=arg, to=self }
      return

    -- scalars, vectors and matrices
    elseif (self._type:isscalarkey() and #arg == self._type.ndims) or
       (self._type:isvector() and #arg == self._type.N) or
       (self._type:ismatrix() and #arg == self._type.Nrow)
    then
      -- fall-through to constant loading

    else
      -- default tables to try loading as Lua lists
      -- TODO: TYPECHECKING HERE
      argcheck_list_dims_and_type(arg, self._owner:Dims(), self._type, 2)
      return self:_INTERNAL_LoadList(arg)
    end
  end
  -- default to try loading as a constant value
  -- TODO: TYPECHECKING HERE
  argcheck_loadval_type(arg, self._type, 2)
  return self:_INTERNAL_LoadConstant(arg)
end

-- joint loading has to be done with a function
function Relation:Load(fieldargs, arg, ...)
  if self:isFragmented() then
    error('cannot load into fragmented relation', 2)
  end

  local fields = {}
  for k,f in ipairs(fieldargs) do
    fields[k] = (type(f) == 'string') and self[f] or f
  end
  argcheck_rel_fields(fields, self, 2)

  -- No support for Lua functions here
  -- Currently no support for loaders, but that should change
  if terralib.isfunction(arg) then
    return self:_INTERNAL_LoadTerraBulkFunction(fields, arg, ...)
  end

  if isdumper(arg) then
    error("tried to use a dumper object to load.  "..
          "Did you mean to use the corresponding dumper object?", 2)
  end

  -- catch-all fall-through
  error('unrecognized argument for Relation:Load(...)', 2)
end

-- pass an empty table to signify that you want the data dumped as a Lua list
function Field:Dump(arg, ...)
  if self._owner:isFragmented() then
    error('cannot dump from a fragmented relation', 2)
  end

  -- TODO: deprecate dumping via a Lua function?
  if type(arg) == 'function' then
    return self:_INTERNAL_DumpLuaPerElemFunction(arg)

  elseif isdumper(arg) then
    return arg._func(self, ...)

  elseif type(arg) == 'table' then
    if terralib.isfunction(arg) then
      return self:_INTERNAL_DumpTerraBulkFunction(arg, ...)
    end
    if #arg == 0 and terralib.israwlist(arg) then -- empty list
      return self:_INTERNAL_DumpList()
    end
  end

  if isloader(arg) then
    error("tried to use a loader object to dump.  "..
          "Did you mean to use the corresponding loader object?", 2)
  end

  -- catch all fall-through
  error('unrecognized argument for Field:Dump(...)', 2)
end

-- joint dumping also has to be done with a function
function Relation:Dump(fieldargs, arg, ...)
  if self:isFragmented() then
    error('cannot dump from a fragmented relation', 2)
  end

  local fields = {}
  for k,f in ipairs(fieldargs) do
    fields[k] = (type(f) == 'string') and self[f] or f
  end
  argcheck_rel_fields(fields, self, 2)

  -- Currently no support for loaders, but that should change
  if terralib.isfunction(arg) then
    return self:_INTERNAL_DumpTerraBulkFunction(fields, arg, ...)

  elseif type(arg) == 'function' then
    return self:_INTERNAL_DumpLuaPerElemFunction(fields, arg)
  end

  if isloader(arg) then
    error("tried to use a loader object to dump.  "..
          "Did you mean to use the corresponding loader object?", 2)
  end

  -- catch-all fall-through
  error('unrecognized argument for Relation:Dump(...)', 2)
end




--[[  I/O: Load from/ save to files, print to stdout                       ]]--

function Field:Print()
  print(self._name..": <" .. tostring(self._type:terratype()) .. '>')
  if use_single and not self._array then
    print("...not initialized")
    return
  end
  local is_elastic = self._owner:isElastic()
  if is_elastic then
    print("  . == live  x == dead")
  end

  local function flattenkey(keytbl)
    if type(keytbl) ~= 'table' then
      return keytbl
    else
      if #keytbl == 2 then
        return '{ '..keytbl[1]..', '..keytbl[2]..' }'
      elseif #keytbl == 3 then
        return '{ '..keytbl[1]..', '..keytbl[2]..', '..keytbl[3]..' }'
      else
        error("INTERNAL: Can only have 2d/3d grid keys, printing what???")
      end
    end
  end

  local fields     = { self }
  if is_elastic then fields[2] = self._owner._is_live_mask end
  self._owner:_INTERNAL_DumpLuaPerElemFunction(fields,
  function(ids, datum, islive)
    local alive = ''
    if is_elastic then
      if islive then alive = ' .'
                else alive = ' x' end
    end

    local idstr = tostring(ids[1])
    if ids[2] then idstr = idstr..' '..tostring(ids[2]) end
    if ids[3] then idstr = idstr..' '..tostring(ids[3]) end

    if self._type:ismatrix() then
      local s = ''
      for c=1,self._type.Ncol do s = s .. flattenkey(datum[1][c]) .. ' ' end
      print("  "..idstr .. alive.."  "..s)

      for r=2,self._type.Nrow do
        local s = ''
        for c=1,self._type.Ncol do s = s .. flattenkey(datum[r][c]) .. ' ' end
        print("    "..s)
      end

    elseif self._type:isvector() then
      local s = ''
      for k=1,self._type.N do s = s .. flattenkey(datum[k]) .. ' ' end
      print("    "..idstr .. alive .."  ".. s)

    else
      print("    "..idstr .. alive .."  ".. flattenkey(datum))
    end
  end)
end


-------------------------------------------------------------------------------
-------------------------------------------------------------------------------
--[[  Data Sharing Hooks                                                   ]]--
-------------------------------------------------------------------------------
-------------------------------------------------------------------------------

function Field:GetDLD()
  if self._owner:isFragmented() then
    error('Cannot get DLD from fragmented relation', 2)
  end

  -- Because the relation is not fragmented, we're guaranteed that
  -- the data is contiguously stored and can be safely exposed to the
  -- external code

  local typ         = self._type
  local type_dims   = ( typ:ismatrix() and {typ.Nrow,typ.Ncol} ) or
                      ( typ:isvector() and {typ.N,1} ) or
                      {1,1}

  if use_single then
    local location  = self._array:location()
    location        = assert( (location == CPU and DLD.CPU) or
                              (location == GPU and DLD.GPU) )
    local dims      = self._owner:Dims()

    local dim_size, dim_stride = {}, {}
    local prod = 1
    for k = 1,3 do
      dim_stride[k] = prod
      dim_size[k]   = dims[k] or 1
      prod          = prod * dim_size[k]
    end
    return DLD.NewDLD {
      base_type       = typ:basetype():DLDEnum(),
      location        = location,
      type_stride     = sizeof(typ:terratype()),
      type_dims       = type_dims,

      address         = self:_Raw_DataPtr(),
      dim_size        = dim_size,
      dim_stride      = dim_stride,
    }
  elseif use_legion then
    error('DLD TO BE REVISED for LEGION')
  else
    assert(use_single or use_legion)
  end

end


-------------------------------------------------------------------------------
-------------------------------------------------------------------------------
--[[  ELASTIC RELATIONS                                                    ]]--
-------------------------------------------------------------------------------
-------------------------------------------------------------------------------


function Relation:_INTERNAL_Resize(new_concrete_size, new_logical)
  if not self:isElastic() then
    error('Can only resize ELASTIC relations', 2)
  end
  if use_legion then error("Can't resize while using Legion", 2) end

  self._is_live_mask._array:resize(new_concrete_size)
  for _,field in ipairs(self._fields) do
    field._array:resize(new_concrete_size)
  end
  self._concrete_size = new_concrete_size
  if new_logical then self._logical_size = new_logical end
end

-------------------------------------------------------------------------------
--[[  Insert / Delete                                                      ]]--
-------------------------------------------------------------------------------

-- returns a useful error message 
function Relation:UnsafeToDelete()
  if not self:isElastic() then
    return "Cannot delete from relation "..self:Name()..
           " because it's not ELASTIC"
  end
  if self:hasSubsets() then
    return 'Cannot delete from relation '..self:Name()..
           ' because it has subsets'
  end
end

function Relation:UnsafeToInsert(record_type)
  -- duplicate above checks
  local msg = self:UnsafeToDelete()
  if msg then
    return msg:gsub('delete from','insert into')
  end

  if record_type ~= self:StructuralType() then
    return 'inserted record type does not match relation'
  end
end


-------------------------------------------------------------------------------
--[[  Defrag                                                               ]]--
-------------------------------------------------------------------------------

function Relation:_INTERNAL_MarkFragmented()
  if not self:isElastic() then
    error("INTERNAL: Cannot Fragment a non-elastic relation")
  end
  rawset(self, '_is_fragmented', true)
end

TOTAL_DEFRAG_TIME = 0
function Relation:Defrag()
  local start_time = terralib.currenttimeinseconds()
  if not self:isElastic() then
    error("Defrag(): Cannot Defrag a non-elastic relation")
  end
  -- TODO: MAKE IDEMPOTENT FOR EFFICIENCY  (huh?)

  -- handle GPU resident fields
  local any_on_gpu  = false
  local on_gpu      = {}
  local live_gpu    = false
  for i,field in ipairs(self._fields) do
    on_gpu[i]   = field._array:location() == GPU
    any_on_gpu  = true
  end
  if self._is_live_mask._array:location() == GPU then
    live_gpu    = true
    any_on_gpu  = true
  end
  -- disallow logic
  --if any_on_gpu then
  --  error('Defrag on GPU unimplemented')
  --end
  -- slow workaround logic
  if any_on_gpu then
    for i,field in ipairs(self._fields) do
      if on_gpu[i] then field:MoveTo(CPU) end
    end
    if live_gpu then self._is_live_mask:MoveTo(CPU) end
  end

  -- ok, build a terra function that we can execute to compact
  -- we can cache it!
  local defrag_func = self._cpu_defrag_func
  local type_sig    = self._cpu_defrag_struct_signature
  if not defrag_func or (type_sig and type_sig ~= self:StructuralType()) then
    -- read and write heads for copy
    local dst = symbol(uint64, 'dst')
    local src = symbol(uint64, 'src')

    -- also need symbols for pointers to all the arrays
    -- They will be passed in as arguments to allow for arrays to move
    local args        = {}
    local liveptrtype = &( self._is_live_mask:Type():terratype() )
    local liveptr     = symbol(liveptrtype)
    args[#self._fields + 1] = liveptr

    -- fill out the rest of the arguments and build a code
    -- snippet that will allow us to copy all of them together
    local do_copy = quote end
    for i,field in ipairs(self._fields) do
      local fptrtype = &( field:Type():terratype() )
      local ptrarg   = symbol( fptrtype )
      args[i]        = ptrarg

      do_copy = quote
        do_copy
        ptrarg[dst] = ptrarg[src]
      end
    end

    defrag_func = terra ( concrete_size : uint64, [args] )
      -- scan the write-head forward from start
      -- and the read head backward from end
      var [dst] = 0
      var [src] = concrete_size - 1
      while dst < src do
        -- scan the src backwards looking for something
        while (src < concrete_size) and -- underflow guard
              not liveptr[src] -- haven't found something to copy yet
        do
          src = src - 1
        end
        -- exit on underflow
        if (src >= concrete_size) then return end

        -- scan the dst forward looking for space to copy into
        while (dst < src) and liveptr[dst] do
          dst = dst + 1
        end

        if dst < src then
          -- do copy
          [do_copy]
          -- flip live bits
          liveptr[dst] = true
          liveptr[src] = false
        end
      end
    end
    rawset(self, '_cpu_defrag_func', defrag_func)
    rawset(self, '_cpu_defrag_struct_signature', self:StructuralType())
  end

  -- assemble the arguments
  local ptrargs = {}
  for i,field in ipairs(self._fields) do
    ptrargs[i] = field:_Raw_DataPtr()
  end
  ptrargs[#self._fields+1] = self._is_live_mask:_Raw_DataPtr()

  -- run the defrag func
  defrag_func(self:ConcreteSize(), unpack(ptrargs))

  -- move back to GPU if necessary
  if any_on_gpu then
    for i,field in ipairs(self._fields) do
      if on_gpu[i] then field:MoveTo(GPU) end
    end
    if live_gpu then self._is_live_mask:MoveTo(GPU) end
  end

  -- now cleanup by resizing the relation
  local logical_size = self:Size()
  -- since the data is now compact, we can shrink down the size
  self:_INTERNAL_Resize(logical_size, logical_size)

  -- mark as compact
  rawset(self, '_is_fragmented', false)
  TOTAL_DEFRAG_TIME = TOTAL_DEFRAG_TIME +
                      (terralib.currenttimeinseconds() - start_time)
end


-------------------------------------------------------------------------------
-------------------------------------------------------------------------------
--[[  Partitioning relations                                               ]]--
-------------------------------------------------------------------------------
-------------------------------------------------------------------------------


function Relation:SetPartitions(num_partitions)
  if self._total_partitions ~= nil then
    error("Partitioning for " .. self._name .. " is already set", 2)
  end
  local ndims = #self:Dims()
  local num_partitions_table = num_partitions
  if type(num_partitions) == 'number' then
    num_partitions_table = { num_partitions }
  end
  if ndims ~= #num_partitions_table then
    error("Need number of partitions for " .. tostring(ndims) ..
          " dimensions " .. " but got " .. tostring(#num_partitions_table) ..
          " dimensions", 2)
  end
  local total_partitions = 1
  for d = 1, ndims do
    total_partitions = total_partitions * num_partitions_table[d]
  end
  rawset(self, '_total_partitions', total_partitions)
  rawset(self, '_num_partitions', num_partitions_table)
  if use_partitioning then
    rawset(self, '_rel_global_partition',
                 P.RelGlobalPartition(self, unpack(num_partitions)))
  end
end

function Relation:_GetGlobalPartition()
  if not self._rel_global_partition and use_partitioning then
    error("If running on Legion, you need to call SetPartitions() on all of "..
          "your relations.  Relation '"..self:Name().."' did not have any "..
          "partition set.")
  end
  return self._rel_global_partition
end

function Relation:TotalPartitions()
  return self._total_partitions
end

function Relation:NumPartitions()
  return self._num_partitions
end

function Relation:IsPartitioningSet()
  return (self._total_partitions ~= nil)
end

-- ghost_width specifies ghost width on each side of a grid.
-- example for a 2d grid, ghost_width is {xl, xh, yl, yh}
-- (l = lower side, h = higher side).
function Relation:SetGhostWidth(ghost_width)
  if not self:isGrid() then
    error("SetGhostWidth supported for only structured relations (grids).", 2)
  else
    local ndims = #self:Dims()
    local num_elems = #ghost_width
    if num_elems ~= 2 * ndims then
      error("Expected a table of " .. tostring(2 * ndims) .. " elements for ghost width." ..
            "Got " .. tostring(num_elems) .. "instead.")
    end
    rawset(self, '_ghost_width_default', ghost_width)
  end
end

function Relation:InvalidateGhostWidth()
  if not self._ghost_width_default then
    error("Attempt to invalidate ghost width which has never been set", 2)
  end
  self._ghost_width_default = nil
end

function Relation:GhostWidth()
  return self._ghost_width_default
end

function Relation:IsGhostWidthValid()
  return self._ghost_width_default ~= nil
end

local ColorPlainIndexSpaceDisjoint = nil
if use_legion then
  ColorPlainIndexSpaceDisjoint = terra(darray : &DLD.C_DLD, num_colors : uint)
    var d = darray[0]
    var b = d.dim_size[0]
    var s = d.dim_stride[0]
    var partn_size = b / num_colors
    if num_colors * partn_size < b then partn_size = partn_size + 1 end
    for i = 0, b do
      var ptr = [&LW.legion_color_t](d.address) + i*s
      @ptr = i / partn_size
    end
  end
end


-------------------------------------------------------------------------------
-------------------------------------------------------------------------------
--[[  Temporary Legion hacks                                               ]]--
-------------------------------------------------------------------------------
-------------------------------------------------------------------------------

-- temporary hack
-- to make it work with legion without blowing up memory
function Relation:TEMPORARY_PrepareForSimulation()
  if use_legion then
    LW._TEMPORARY_LaunchEmptySingleTaskOnRelation(self)
  end
end
