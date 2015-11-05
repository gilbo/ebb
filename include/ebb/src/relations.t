-- file/module namespace table
local R = {}
package.loaded["ebb.src.relations"] = R

local use_legion = not not rawget(_G, '_legion_env')
local use_single = not use_legion

local L = require "ebblib"
local T = require "ebb.src.types"
local C = require "ebb.src.c"
local DLD = require "ebb.src.dld"
local DLDiter = require 'ebb.src.dlditer'

local PN = require "ebb.lib.pathname"

local rawdata = require('ebb.src.rawdata')
local DynamicArray = use_single and rawdata.DynamicArray
local DataArray    = use_single and rawdata.DataArray
local LW = use_legion and require "ebb.src.legionwrap"

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
--[[  LRelation methods                                                    ]]--
-------------------------------------------------------------------------------
-------------------------------------------------------------------------------


-- A Relation can be in at most one of the following MODES
--    PLAIN
--    GROUPED (has been sorted for reference)
--    GRID
--    ELASTIC (can insert/delete)
function L.LRelation:isPlain()      return self._mode == 'PLAIN'      end
function L.LRelation:isGrouped()    return self._mode == 'GROUPED'    end
function L.LRelation:isGrid()       return self._mode == 'GRID'       end
function L.LRelation:isElastic()    return self._mode == 'ELASTIC'    end

function L.LRelation:isFragmented() return self._is_fragmented end

-- Create a generic relation
-- local myrel = L.NewRelation {
--   name = 'myrel',
--   mode = 'PLAIN',
--  [size = 35,]        -- IF mode ~= 'GRID'
--  [dims = {45,90}, ]  -- IF mode == 'GRID'
-- }
local relation_uid = 0
function L.NewRelation(params)
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
  L.LRelation)
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
    rawset(rel, '_is_live_mask', L.LField.New(rel, '_is_live_mask', L.bool))
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

function L.LRelation:_INTERNAL_UID() -- Why is this even necessary?
  return self._uid
end
function L.LRelation:Size()
  return self._logical_size
end
function L.LRelation:ConcreteSize()
  return self._concrete_size
end
function L.LRelation:Name()
  return self._name
end
function L.LRelation:nDims()
  if self:isGrid() then
    return #self._dims
  else
    return 1
  end
end
function L.LRelation:Dims()
  if not self:isGrid() then
    return { self:Size() }
  end

  local dimret = {}
  for i,n in ipairs(self._dims) do dimret[i] = n end
  return dimret
end
function L.LRelation:Strides()
  if not self:isGrid() then
    return { 1 }
  end
  local dimstrides = {}
  for i,n in ipairs(self._strides) do dimstrides[i] = n end
  return dimstrides
end
function L.LRelation:GroupedKeyField()
  if not self:isGrouped() then return nil
                          else return self._grouped_field end
end
function L.LRelation:_INTERNAL_GroupedOffset()
  if not self:isGrouped() then return nil
                          else return self._grouped_offset end
end
function L.LRelation:_INTERNAL_GroupedLength()
  if not self:isGrouped() then return nil
                          else return self._grouped_length end
end
function L.LRelation:Periodicity()
  if not self:isGrid() then return { false } end
  local wraps = {}
  for i,p in ipairs(self._dims) do wraps[i] = p end
  return wraps
end

function L.LRelation:foreach(user_func, ...)
  if not L.is_function(user_func) then
    error('foreach(): expects an ebb function as the first argument', 2)
  end
  user_func:_doForEach(self, ...)
end

function L.LRelation:hasSubsets()
  return #self._subsets ~= 0
end

-- returns a record type
function L.LRelation:StructuralType()
  local rec = {}
  for _, field in ipairs(self._fields) do
    rec[field.name] = field.type
  end
  local typ = L.record(rec)
  return typ
end

-- prevent user from modifying the lua table
function L.LRelation:__newindex(fieldname,value)
  error("Cannot assign members to LRelation object "..
      "(did you mean to call relation:New...?)", 2)
end


function L.LRelation:NewFieldMacro (name, macro)
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

  if not L.is_macro(macro) then
    error("NewFieldMacro() expects a Macro as the 2nd argument", 2)
  end

  rawset(self, name, macro)
  self._macros:insert(macro)
  return macro
end

function L.LRelation:NewFieldFunction (name, userfunc)
  if not name or type(name) ~= "string" then
    error("NewFieldFunction() expects a string as the first argument", 2)
  end
  if not is_valid_lua_identifier(name) then
    error(valid_name_err_msg.field, 2)
  end
  if self[name] then
    error("Cannot create a new field-function with name '"..name.."'  "..
          "That name is already being used.", 2)
  end

  if not L.is_function(userfunc) then
    error("NewFieldFunction() expects an Ebb Function "..
          "as the 2nd argument", 2)
  end

  rawset(self, name, userfunc)
  self._functions:insert(userfunc)
  return userfunc
end

function L.LRelation:GroupBy(keyf_name)
  if self:isGrouped() then
    error("GroupBy(): Relation is already grouped", 2)
  elseif not self:isPlain() then
    error("GroupBy(): Cannot group a relation "..
          "unless it's a PLAIN relation", 2)
  end

  local key_field = keyf_name
  if type(key_field) == 'string' then key_field = self[key_field] end
  if not L.is_field(key_field) then
    error("GroupBy(): Could not find a field named '"..keyf_name.."'", 2)
  elseif not key_field.type:isscalarkey() then
    error("GroupBy(): Grouping by non-scalar-key fields is "..
          "prohibited.", 2)
  end

  -- In the below, we use the following convention
  --  SRC is the relation referred to by the key field
  --  DST is 'self' here, the relation which is actively being grouped
  --    In a Where query, a key into the SRC relation is translated
  --    into a sequence of keys into the DST relation
  local srcrel = key_field.type.relation
  local dstrel = self
  local n_src  = srcrel:Size()
  local n_dst  = dstrel:Size()
  local dstname = dstrel:Name()
  local offset_f = L.LField.New(srcrel, dstname..'_grouped_offset', L.uint64)
  local length_f = L.LField.New(srcrel, dstname..'_grouped_length', L.uint64)

  rawset(self,'_grouped_field', key_field)
  rawset(self,'_grouped_offset', offset_f)
  rawset(self,'_grouped_length', length_f)

  if use_single then
    -- NOTE: THIS IMPLEMENTATION HEAVILY ASSUMES THAT A GRID IS LINEARIZED
    -- IN ROW-MAJOR ORDER
    local offptr = offset_f.array:open_write_ptr()
    local lenptr = length_f.array:open_write_ptr()
    local keyptr = key_field.array:open_read_ptr()
    --offset_f.array:write_ptr(function(offptr)
    --length_f.array:write_ptr(function(lenptr)
    --key_field.array:read_ptr(function(keyptr)
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
    --end) -- key_field read
    --end) -- length_f write
    --end) -- offset_f write
    key_field.array:close_read_ptr()
    length_f.array:close_write_ptr()
    offset_f.array:close_write_ptr()
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

function L.LRelation:MoveTo( proc )
  if use_legion then error("MoveTo() unsupported using Legion", 2) end
  if proc ~= L.CPU and proc ~= L.GPU then
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


function L.LRelation:print()
  if use_legion then
    error("print() currently unsupported using Legion", 2)
  end
  print(self._name, "size: ".. tostring(self:Size()),
                    "concrete size: "..tostring(self:ConcreteSize()))
  for i,f in ipairs(self._fields) do
    f:print()
  end
end


-------------------------------------------------------------------------------
-------------------------------------------------------------------------------
--[[  Indices:                                                             ]]--
-------------------------------------------------------------------------------
-------------------------------------------------------------------------------


function L.LIndex.New(params)
  if not L.is_relation(params.owner) or
     type(params.name) ~= 'string' or
     not (params.size or params.terra_type or params.data)
  then
    error('bad parameters')
  end

  local index = setmetatable({
    _owner = params.owner,
    _name  = params.name,
  }, L.LIndex)

  index._array = DynamicArray.New {
    size = params.size or (#params.data),
    type = params.terra_type,
    processor = params.processor or L.default_processor,
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

function L.LIndex:DataPtr()
  return self._array:_raw_ptr()
end
function L.LIndex:Size()
  return self._array:size()
end

function L.LIndex:Relation()
  return self._owner
end

function L.LIndex:ReAllocate(size)
  self._array:resize(size)
end

function L.LIndex:MoveTo(proc)
  self._array:moveto(proc)
end

function L.LIndex:Release()
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


function L.LSubset:foreach(user_func, ...)
  if not L.is_function(user_func) then
    error('map(): expects an Ebb function as the argument', 2)
  end
  user_func:_doForEach(self, ...)
end

function L.LSubset:Relation()
  return self._owner
end

function L.LSubset:Name()
  return self._name
end

function L.LSubset:FullName()
  return self._owner._name .. '.' .. self._name
end

function L.LSubset:MoveTo( proc )
  if proc ~= L.CPU and proc ~= L.GPU then
    error('must specify valid processor to move to', 2)
  end

  if self._boolmask   then self._boolmask:MoveTo(proc)    end
  if self._index      then self._index:MoveTo(proc)       end
end

function L.LRelation:NewSubsetFromFunction (name, predicate)
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
  }, L.LSubset)
  rawset(self, name, subset)
  self._subsets:insert(subset)

  -- NOW WE DECIDE how to encode the subset
  -- we'll try building a mask and decide between using a mask or index
  local SUBSET_CUTOFF_FRAC = 0.1
  local SUBSET_CUTOFF = SUBSET_CUTOFF_FRAC * self:Size()

  local boolmask  = L.LField.New(self, name..'_subset_boolmask', L.bool)
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
    subset._boolmask = boolmask
  else
  -- USE INDEX
    subset._index = L.LIndex.New{
      owner=self,
      terra_type = L.key(self):terratype(),
      ndims=self:nDims(),
      name=name..'_subset_index',
      data=index_tbl
    }
    boolmask:ClearData() -- free memory
  end

  return subset
end


-------------------------------------------------------------------------------
-------------------------------------------------------------------------------
--[[  Fields:                                                              ]]--
-------------------------------------------------------------------------------
-------------------------------------------------------------------------------


-- Client code should never call this constructor
-- For internal use only.  Does not install on relation...
function L.LField.New(rel, name, typ)
  local field   = setmetatable({}, L.LField)
  field.type    = typ
  field.name    = name
  field.owner   = rel
  if use_single then
    field.array   = nil
    field:Allocate()
  elseif use_legion then
    field.fid = rel._logical_region_wrapper:AllocateField(typ:terratype())
  end
  return field
end

function L.LField:Name()
  return self.name
end
function L.LField:FullName()
  return self.owner._name .. '.' .. self.name
end
function L.LField:Size()
  return self.owner:Size()
end
function L.LField:ConcreteSize()
  return self.owner:ConcreteSize()
end
function L.LField:Type()
  return self.type
end
function L.LField:DataPtr()
  if use_legion then error('DataPtr() unsupported using legion') end
  return self.array:_raw_ptr()
end
function L.LField:Relation()
  return self.owner
end

function L.LRelation:NewField (name, typ)  
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
  
  if L.is_relation(typ) then
    typ = L.key(typ)
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
  local field = L.LField.New(self, name, typ)
  rawset(self, name, field)
  self._fields:insert(field)

  -- record reverse pointers for key-field references
  if typ:iskey() then
    typ:basetype().relation._incoming_refs[field] = 'key_field'
  end

  return field
end

-- TODO: Hide this function so it's not public
function L.LField:Allocate()
  if use_legion then error('No Allocate() using legion') end
  if not self.array then
    --if self.owner:isElastic() then
      self.array = DynamicArray.New{
        size = self:ConcreteSize(),
        type = self:Type():terratype()
      }
    --else
    --  self.array = DataArray.New {
    --    size = self:ConcreteSize(),
    --    type = self:Type():terratype()
    --  }
    --end
  end
end

-- TODO: Hide this function so it's not public
-- remove allocated data and clear any depedent data, such as indices
function L.LField:ClearData ()
  if use_legion then error('No ClearData() using legion') end
  if self.array then
    self.array:free()
    self.array = nil
  end
  -- clear grouping data if set on this field
  if self.owner:isGrouped() and
     self.owner:GroupedKeyField() == self
  then
    error('UNGROUPING CURRENTLY UNIMPLEMENTED')
  end
end

function L.LField:MoveTo( proc )
  if use_legion then error('No MoveTo() using legion') end
  if proc ~= L.CPU and proc ~= L.GPU then
    error('must specify valid processor to move to', 2)
  end

  self.array:moveto(proc)
end

function L.LRelation:Swap( f1_name, f2_name )
  local f1 = self[f1_name]
  local f2 = self[f2_name]
  if not L.is_field(f1) then
    error('Could not find a field named "'..f1_name..'"', 2) end
  if not L.is_field(f2) then
    error('Could not find a field named "'..f2_name..'"', 2) end
  if f1.type ~= f2.type then
    error('Cannot Swap() fields of different type', 2)
  end

  if use_single then
    local tmp = f1.array
    f1.array = f2.array
    f2.array = tmp
  elseif use_legion then
    local region  = self._logical_region_wrapper
    local rhandle = region.handle
    local fid_1   = f1.fid
    local fid_2   = f2.fid
    -- create a temporary Legion field
    local fid_tmp = region:AllocateField(f1.type:terratype())

    LW.CopyField { region = rhandle,  src_fid = fid_1,    dst_fid = fid_tmp }
    LW.CopyField { region = rhandle,  src_fid = fid_2,    dst_fid = fid_1   }
    LW.CopyField { region = rhandle,  src_fid = fid_tmp,  dst_fid = fid_2   }

    -- destroy temporary field
    region:FreeField(fid_tmp)
  end
end

function L.LRelation:Copy( p )
  if type(p) ~= 'table' or not p.from or not p.to then
    error("relation:Copy() should be called using the form\n"..
          "  relation:Copy{from='f1',to='f2'}", 2)
  end
  local from = p.from
  local to   = p.to
  if type(from) == 'string' then from = self[from] end
  if type(to)   == 'string' then to   = self[to]   end
  if not L.is_field(from) then
    error('Could not find a field named "'..p.from..'"', 2) end
  if not L.is_field(to) then
    error('Could not find a field named "'..p.to..'"', 2) end
  if not from:Relation() == self then
    error('Field '..from:FullName()..' is not a field of '..
          'Relation '..self:Name(), 2) end
  if not to:Relation() == self then
    error('Field '..to:FullName()..' is not a field of '..
          'Relation '..self:Name(), 2) end
  if from.type ~= to.type then
    error('Cannot Copy() fields of different type', 2)
  end

  if use_single then
    if not from.array then
      error('Cannot Copy() from field with no data', 2) end
    if not to.array then
      to:Allocate()
    end
    to.array:copy(from.array)

  elseif use_legion then
    LW.CopyField {
      region  = self._logical_region_wrapper.handle,
      src_fid = from.fid,
      dst_fid = to.fid,
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
function L.LRelation:_INTERNAL_MapJointFunction(
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
    assert(L.is_field(f), 'arg should be list of fields')
    assert(f.owner == self, 'fields should belong to this relation')
  end
  assert(not self:isFragmented(), 'cannot expose a fragmented relation')

  local dld_list = isterra and C.safemalloc(DLD.C_DLD, #fields) or {}
  if use_single then
    for i = 1, #fields do
      local dld     = fields[i]:GetDLD()
      if iswrite then   dld.address   = fields[i].array:open_write_ptr()
                 else   dld.address   = fields[i].array:open_read_ptr() end
      dld:setlocation(DLD.CPU)
      if isterra then   dld_list[i-1] = dld:toTerra()
                 else   dld_list[i]   = dld end
    end

    local retvals = { clbk(dld_list, ...) }

    for i = 1, #fields do
      if iswrite then   fields[i].array:close_write_ptr()
                 else   fields[i].array:close_read_ptr() end
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

function L.LField:_INTERNAL_LoadLuaPerElemFunction(clbk)
  local typ       = self:Type()
  local ttyp      = typ:terratype()

  self.owner:_INTERNAL_MapJointFunction(true, false, function(dldlist)
    local dld         = dldlist[1]
    local ptr         = terralib.cast(&ttyp, dld.address)

    DLDiter.luaiter(dld, function(lin, ...)
      local tval  = T.luaToEbbVal( clbk(...), typ )
      ptr[lin]    = tval
    end)
  end, {self})
end

function L.LField:_INTERNAL_LoadTerraBulkFunction(clbk, ...)
  return self.owner:_INTERNAL_MapJointFunction(true, true, clbk, {self}, ...)
end

function L.LRelation:_INTERNAL_LoadTerraBulkFunction(fields, clbk, ...)
  return self:_INTERNAL_MapJointFunction(true, true, clbk, fields, ...)
end

-- this is broken for unstructured relations with legion
function L.LField:_INTERNAL_LoadList(tbl)
  local ndims = self.owner:nDims()
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

function L.LField:_INTERNAL_LoadConstant(c)
  -- TODO: Convert implementation to Terra
  self:_INTERNAL_LoadLuaPerElemFunction(function()
    return c
  end)
end


-------------------------------------------------------------------------------
--[[  Mid-Level Dumping Operations (Lua and Terra)                         ]]--

function L.LRelation:_INTERNAL_DumpLuaPerElemFunction(fields, clbk)
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

function L.LField:_INTERNAL_DumpLuaPerElemFunction(clbk)
  self.owner:_INTERNAL_DumpLuaPerElemFunction({self}, function(ids, val)
    clbk(val, unpack(ids))
  end)
end

function L.LRelation:_INTERNAL_DumpTerraBulkFunction(fields, clbk, ...)
  return self:_INTERNAL_MapJointFunction(false, true, clbk, fields, ...)
end

function L.LField:_INTERNAL_DumpTerraBulkFunction(clbk, ...)
  return self.owner:_INTERNAL_MapJointFunction(false, true, clbk, {self}, ...)
end

function L.LField:_INTERNAL_DumpList()
  local result = {}
  local dims = self.owner:Dims()

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
  return blob.name..': '
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
    if not L.is_field(f) then
      error(ferrprefix(lvl)..'Expected field at list entry '..i, lvl)
    end
    if not f.owner == rel then
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
  lvl = (lvl or 1)+1
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
L.isloader = isloader
L.isdumper = isdumper

function L.NewLoader(loadfunc)
  if type(loadfunc) ~= 'function' then
    error('NewLoader() expects a lua function as the argument', 2)
  end
  return setmetatable({
    _func = loadfunc,
  }, CustomLoader)
end

function L.NewDumper(dumpfunc)
  if type(dumpfunc) ~= 'function' then
    error('NewDumper() expects a lua function as the argument', 2)
  end
  return setmetatable({
    _func = dumpfunc,
  }, CustomDumper)
end

-------------------------------------------------------------------------------
--[[  High-Level Loading and Dumping Operations (Lua and Terra)            ]]--

function L.LField:Load(arg, ...)
  if self.owner:isFragmented() then
    error('cannot load into fragmented relation', 2)
  end

  -- TODO(Chinmayee): deprecate this
  if type(arg) == 'function' then
    return self:_INTERNAL_LoadLuaPerElemFunction(arg)

  elseif isloader(arg) then
    return arg._func(self, ...)

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

    -- scalars, vectors and matrices
    elseif (self.type:isscalarkey() and #arg == self.type.ndims) or
       (self.type:isvector() and #arg == self.type.N) or
       (self.type:ismatrix() and #arg == self.type.Nrow)
    then
      -- fall-through to constant loading

    else
      -- default tables to try loading as Lua lists
      -- TODO: TYPECHECKING HERE
      argcheck_list_dims_and_type(arg, self.owner:Dims(), self.type, 2)
      return self:_INTERNAL_LoadList(arg)
    end
  end
  -- default to try loading as a constant value
  -- TODO: TYPECHECKING HERE
  argcheck_loadval_type(arg, self.type, 2)
  return self:_INTERNAL_LoadConstant(arg)
end

-- joint loading has to be done with a function
function L.LRelation:Load(fieldargs, arg, ...)
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

  -- catch-all fall-through
  error('unrecognized argument for Relation:Load(...)', 2)
end

-- pass an empty table to signify that you want the data dumped as a Lua list
function L.LField:Dump(arg, ...)
  if self.owner:isFragmented() then
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

  -- catch all fall-through
  error('unrecognized argument for Field:Dump(...)', 2)
end

-- joint dumping also has to be done with a function
function L.LRelation:Dump(fieldargs, arg, ...)
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

  -- catch-all fall-through
  error('unrecognized argument for Relation:Dump(...)', 2)
end


--[[
function L.LField:LoadFunction(lua_callback)
  if self.owner:isFragmented() then
    error('cannot load into fragmented relation', 2)
  end

  if use_legion then
    if self.owner:isPlain() then
      -- error("LoadList for unstructured relations is broken with legion")
    end
    -- Ok, we need to map some stuff down here
    local scanner = LW.NewControlScanner {
      relation       = self.owner,
      fields         = { self },
      privilege      = LW.WRITE_ONLY
    }
    for ids, ptrs in scanner:ScanThenClose() do
      local lval = lua_callback(unpack(ids))
      if not T.luaValConformsToType(lval, self.type) then
        error("lua value does not conform to field type "..
              tostring(self.type), 3)
      end
      local tval = T.luaToEbbVal(lval, self.type)
      terralib.cast(&(self.type:terratype()), ptrs[1])[0] = tval
    end
  elseif use_single then
    self:Allocate()

    local dims = self.owner:Dims()
    local dataptr = self.array:open_write_ptr()
    --self.array:write_ptr(function(dataptr)
      for lin,ids in self.owner:_INTERNAL_iter_gen() do
        local val = lua_callback(unpack(ids))
        if not T.luaValConformsToType(val, self.type) then
          error("lua value does not conform to field type "..
                tostring(self.type), 5)
        end
        dataptr[lin] = T.luaToEbbVal(val, self.type)
      end
    --end) -- write_ptr
    self.array:close_write_ptr()
  end
end


-- To load fields using terra callback. Terra callback gets a list of dlds.
--   callback([dlds])
function L.LRelation:LoadJointTerraFunction( terra_callback, fields_arg, ...)
  if not terralib.isfunction(terra_callback) then
    error('LoadJointTerraFunction.. should be used with terra callback')
  end
  if self:isFragmented() then
    error('cannot load to fragmented relation', 2)
  elseif type(fields_arg) ~= 'table' or #fields_arg == 0 then
    error('LoadJointTerraFunction(): Expects a list of fields as its first argument', 2)
  end
  local fields = {}
  for i,f in ipairs(fields_arg) do
    if type(f) == 'string' then f = self[f] end
    if not L.is_field(f) then
      error('LoadJointTerraFunction(): list entry '..tostring(i)..' was either '..
            'not a field or not the name of a field in '..
            'relation '..self:Name(),2)
    end
    if f.owner ~= self then
      error('LoadJointTerraFunction(): list entry '..tostring(i)..', field '..
            f:FullName()..' is not a field of relation '..self:Name(), 2)
    end
    fields[i] = f
  end
  local nfields = #fields

  local dld_array = terralib.new(DLD.C_DLD[nfields])
  if use_single then
    --local cpu_buf = {}
    for i = 1, nfields do
      local dld = fields[i]:GetDLD()
      dld.address = fields[i].array:open_write_ptr()
      dld:setlocation(DLD.CPU)
      dld_array[i-1] = dld:toTerra()
    end

    terra_callback(dld_array, ...)

    for i = 1, nfields do
      fields[i].array:close_write_ptr()
    end
  elseif use_legion then
    -- TODO(Chinmayee): check if it is better to do a separate physical region
    -- for each field
    local params = { relation = self, fields = fields, privilege = LW.WRITE_ONLY }
    local region = LW.NewInlinePhysicalRegion(params)
    local data_ptrs = region:GetDataPointers()
    local dims      = self:Dims()
    local strides   = region:GetStrides()
    local offsets   = region:GetOffsets()
    for i = 1, nfields do
      local dld = fields[i]:GetDLD()
      dld:SetDataPointer(data_ptrs[i])
      dld:SetDims(dims)
      dld:SetStride(strides[i])
      dld:SetOffset(offsets[i])
      dld_array[i-1] = dld:Compile()
    end

    terra_callback(dld_array, ...)

    region:Destroy()
  end
end

-- Load a single field using a terra callback
-- callback accepts argument dld
--   callback(dld)
function L.LField:LoadTerraFunction(terra_callback, ...)
  if not terralib.isfunction(terra_callback) then
    error('LoadTerraFunction should be used with terra callback')
  end
  self.owner:LoadJointTerraFunction(terra_callback, {self}, ...)
end

--[[
-- this is broken for unstructured relations with legion
function L.LField:LoadList(tbl)
  if self.owner:isFragmented() then
    error('cannot load into fragmented relation', 2)
  end
  if type(tbl) ~= 'table' then
    error('bad type passed to LoadList().  Expecting a table', 2)
  end
  if self.owner:isGrid() then
    local dims = self.owner:Dims()
    local dimstr = '{'..tostring(dims[1])..','..tostring(dims[2])
    if dims[3] then dimstr = dimstr..','..tostring(dims[3]) end
    dimstr = dimstr..'}'
    local errmsg = 'argument list should have dimensions '..dimstr

    if dims[3] then
      if dims[3] ~= #tbl then error(errmsg, 2) end
      for zi=1,dims[3] do
        if dims[2] ~= #tbl[zi] then error(errmsg,2) end
        for yi=1,dims[2] do
          if dims[1] ~= #tbl[zi][yi] then error(errmsg,2) end
        end
      end
    else
      if dims[2] ~= #tbl then error(errmsg,2) end
      for yi=1,dims[2] do
        if dims[1] ~= #tbl[yi] then error(errmsg,2) end
      end
    end
  else
    if #tbl ~= self:Size() then
      error('argument list has the wrong number of elements: '..
            tostring(#tbl)..
            ' (was expecting '..tostring(self:Size())..')', 2)
    end
  end

  if self.owner:nDims() == 1 then
    self:LoadFunction(function(i) return tbl[i+1] end)
  elseif self.owner:nDims() == 2 then
    self:LoadFunction(function(xi,yi) return tbl[yi+1][xi+1] end)
  elseif self.owner:nDims() == 3 then
    self:LoadFunction(function(xi,yi,zi) return tbl[zi+1][yi+1][xi+1] end)
  else
    error('INTERNAL > 3 dimensions')
  end
end
--]]

--[[
-- TODO: DEPRECATED FORM.  (USE DLD?)
function L.LField:LoadFromMemory(mem)
  if self.owner:isFragmented() then
    error('cannot load into fragmented relation', 2)
  end
  if use_legion then
    error('Load from memory while using Legion is unimplemented', 2)
  end
  self:Allocate()

  if self.type:iskey() then
    if not self.type:isscalar() then
      error('no support for loading non-scalar keys from memory', 2)
    end
    if self.type.ndims ~= 1 then
      error('no support for loading non-1d keys from memory', 2)
    end
    -- read out a list and then load that
    local data      = {}
    local n_array   = self:Size()
    local arr       = terralib.cast(&uint64, mem)
    for k=0,n_array-1 do
      data[k+1] = tonumber(arr[k])
    end
    self:LoadList(data)
  else
  -- avoid extra copies by wrapping and using the standard copy
    local wrapped = DynamicArray.Wrap{
      size = self:ConcreteSize(),
      type = self.type:terratype(),
      data = mem,
      processor = L.CPU,
    }
    self.array:copy(wrapped)
  end
end

--[[
function L.LField:LoadConstant(constant)
  if self.owner:isFragmented() then
    error('cannot load into fragmented relation', 2)
  end

  local ttype = self.type:terratype()

  -- TODO: We haven't guaranteed that size of ttype equals
  --       the type stride
  local terra LoadConstantFunction(darray : &DLD.C_DLD)
    var d = darray[0]
    var c : ttype = [T.luaToEbbVal(constant, self.type)]
    var b = d.dim_size
    var s = d.dim_stride
    var addr = [&ttype](d.address)
    for i = 0, b[0] do
      for j = 0, b[1] do
        for k = 0, b[2] do
          var ptr = addr + i*s[0] + j*s[1] + k*s[2]
          C.memcpy(ptr, &c, sizeof(ttype))
        end
      end
    end
  end

  self:LoadTerraFunction(LoadConstantFunction)
end
--]]

--[[
-- generic dispatch function for loads
function L.LField:Load(arg)
  if self.owner:isFragmented() then
    error('cannot load into fragmented relation', 2)
  end
  -- load from lua callback
  -- TODO(Chinmayee): deprecate this
  if type(arg) == 'function' then
    return self:LoadFunction(arg)
  elseif  type(arg) == 'cdata' then
    if use_legion then
      error('Load from memory while using Legion is unimplemented', 2)
    end
    local typ = terralib.typeof(arg)

    if typ and typ:ispointer() then
      return self:LoadFromMemory(arg)
    end
  elseif  type(arg) == 'table' then
    -- terra function
    if (terralib.isfunction(arg)) then
      return self:LoadTerraFunction(arg)
    -- scalars, vectors and matrices
    elseif (self.type:isscalarkey() and #arg == self.type.ndims) or
       (self.type:isvector() and #arg == self.type.N) or
       (self.type:ismatrix() and #arg == self.type.Nrow)
    then
      return self:LoadConstant(arg)
    else
      -- default tables to try loading as Lua lists
      return self:LoadList(arg)
    end
  end
  -- default to try loading as constants
  return self:LoadConstant(arg)
end
--]]


--[[  Dumping                                                               ]]--

--[[
-- helper to dump multiple fields jointly
function L.LRelation:DumpJoint(fields_arg, lua_callback)
  if self:isFragmented() then
    error('cannot dump from fragmented relation', 2)
  end
  if type(fields_arg) ~= 'table' or #fields_arg == 0 then
    error('DumpJoint(): Expects a list of fields as its first argument', 2)
  end
  local fields = {}
  for i,f in ipairs(fields_arg) do
    if type(f) == 'string' then f = self[f] end
    if not L.is_field(f) then
      error('DumpJoint(): list entry '..tostring(i)..' was either '..
            'not a field or not the name of a field in '..
            'relation '..self:Name(),2)
    end
    if f.owner ~= self then
      error('DumpJoint(): list entry '..tostring(i)..', field '..
            f:FullName()..' is not a field of relation '..self:Name(), 2)
    end
    fields[i] = f
  end

  if use_legion then
    local typs = {}
    for k=1,#fields do
      local f = fields[k]
      typs[k] = f.type
    end

    local scanner = LW.NewControlScanner {
      relation  = self,
      fields    = fields,
      privilege = LW.READ_ONLY
    }
    for ids, ptrs in scanner:ScanThenClose() do
      local vals = {}
      for k=1,#fields do
        local tval = terralib.cast(&(typs[k]:terratype()), ptrs[k])[0]
        vals[k] = T.ebbToLuaVal(tval, typs[k])
      end
      lua_callback(ids, unpack(vals))
    end
  else
    local dims = self:Dims()
    local nfields = #fields
    local typs = {}
    local ptrs = {}

    -- main loop part
    local loop = function()
      for lin,ids in self:_INTERNAL_iter_gen() do
        local vals = {}
        for k=1,nfields do
          vals[k] = T.ebbToLuaVal(ptrs[k][lin], typs[k])
        end
        lua_callback(ids, unpack(vals))
      end
    end

    for k=1,nfields do
      local f     = fields[k]
      typs[k]     = f.type
      local loopcapture = loop -- THIS IS NEEDED TO STOP INF. RECURSION
      local outerloop = function()
        local dataptr = f.array:open_read_ptr()
        --f.array:read_ptr(function(dataptr)
          ptrs[k] = dataptr
          loopcapture()
        --end)
        f.array:close_read_ptr()
      end
      loop = outerloop
    end

    loop()
  end
end
--]]

--[[
-- callback(i, val)
--      i:      which row we're outputting (starting at 0)
--      val:    the value of this field for the ith row
function L.LField:DumpFunction(lua_callback)
  if self.owner:isFragmented() then
    error('cannot dump from fragmented relation', 2)
  end
  self.owner:DumpJoint({self}, function(ids, val)
    lua_callback(val, unpack(ids))
  end)
end

function L.LField:DumpToList()
  if self.owner:isFragmented() then
    error('cannot dump from fragmented relation', 2)
  end
  local arr = {}
  local dims = self.owner:Dims()
  if #dims == 1 then
    self:DumpFunction(function(val, i)
      arr[i+1] = val
    end)
  elseif #dims == 2 then
    for yi=1,dims[2] do arr[yi] = {} end
    self:DumpFunction(function(val, xi,yi)
      arr[yi+1][xi+1] = val
    end)
  elseif #dims == 3 then
    for zi=1,dims[3] do
      arr[zi] = {}
      for yi=1,dims[2] do arr[zi][yi] = {} end
    end
    self:DumpFunction(function(val, xi,yi,zi)
      arr[zi+1][yi+1][xi+1] = val
    end)
  else
    error('INTERNAL: > 3 dims')
  end
  return arr
end



-- To dump fields using terra callback. Terra callback gets a list of dlds.
--   callback([dlds])
function L.LRelation:DumpJointTerraFunction(terra_callback, fields_arg, ...)
  if not terralib.isfunction(terra_callback) then
    error('DumpJointTerraFunction.. should be used with terra callback')
  end
  if self:isFragmented() then
    error('cannot dump from fragmented relation', 2)
  elseif type(fields_arg) ~= 'table' or #fields_arg == 0 then
    error('DumpJointTerraFunction(): Expects a list of fields as its first argument', 2)
  end
  local fields = {}
  for i,f in ipairs(fields_arg) do
    if type(f) == 'string' then f = self[f] end
    if not L.is_field(f) then
      error('DumpJointTerraFunction(): list entry '..tostring(i)..' was either '..
            'not a field or not the name of a field in '..
            'relation '..self:Name(),2)
    end
    if f.owner ~= self then
      error('DumpJointFunction(): list entry '..tostring(i)..', field '..
            f:FullName()..' is not a field of relation '..self:Name(), 2)
    end
    fields[i] = f
  end
  local nfields = #fields

  local dld_array = terralib.new(DLD.C_DLD[nfields])
  if use_single then
    for i = 1, nfields do
      local dld   = fields[i]:GetDLD()
      dld.address = fields[i].array:open_read_ptr()
      dld:setlocation(DLD.CPU)
      dld_array[i-1] = dld:toTerra()
    end

    terra_callback(dld_array, ...)

    for i = 1, nfields do
      fields[i].array:close_read_ptr()
    end
  elseif use_legion then
    -- TODO(Chinmayee): check if it is better to do a separate physical region
    -- for each field
    local params = { relation = self, fields = fields, privilege = LW.READ_ONLY }
    local region = LW.NewInlinePhysicalRegion(params)
    local data_ptrs = region:GetDataPointers()
    local dims      = self:Dims()
    local strides   = region:GetStrides()
    local offsets   = region:GetOffsets()
    for i = 1, nfields do
      local dld = fields[i]:GetDLD()
      dld:SetDataPointer(data_ptrs[i])
      dld:SetDims(dims)
      dld:SetStride(strides[i])
      dld:SetOffset(offsets[i])
      dld_array[i-1] = dld:Compile()
    end
    if opt_args then
      terra_callback(dld_array, unpack(opt_args))
    else
      terra_callback(dld_array)
    end
    region:Destroy()
  end
end

-- Dump a single field using a terra callback
-- callback accepts argument dld
--   callback(dld)
function L.LField:DumpTerraFunction(terra_callback, ...)
  if not terralib.isfunction(terra_callback) then
    error('DumpTerraFunction should be used with terra callback')
  end
  self.owner:DumpJointTerraFunction(terra_callback, {self}, ...)
end


--[[  I/O: Load from/ save to files, print to stdout                       ]]--

function L.LField:print()
  print(self.name..": <" .. tostring(self.type:terratype()) .. '>')
  if use_single and not self.array then
    print("...not initialized")
    return
  end
  local is_elastic = self.owner:isElastic()
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
  if is_elastic then fields[2] = self.owner._is_live_mask end
  self.owner:_INTERNAL_DumpLuaPerElemFunction(fields,
  function(ids, datum, islive)
    local alive = ''
    if is_elastic then
      if islive then alive = ' .'
                else alive = ' x' end
    end

    local idstr = tostring(ids[1])
    if ids[2] then idstr = idstr..' '..tostring(ids[2]) end
    if ids[3] then idstr = idstr..' '..tostring(ids[3]) end

    if self.type:ismatrix() then
      local s = ''
      for c=1,self.type.Ncol do s = s .. flattenkey(datum[1][c]) .. ' ' end
      print("", idstr .. alive, s)

      for r=2,self.type.Nrow do
        local s = ''
        for c=1,self.type.Ncol do s = s .. flattenkey(datum[r][c]) .. ' ' end
        print("", "", s)
      end

    elseif self.type:isvector() then
      local s = ''
      for k=1,self.type.N do s = s .. flattenkey(datum[k]) .. ' ' end
      print("", idstr .. alive, s)

    else
      print("", idstr .. alive, flattenkey(datum))
    end
  end)
end

--[[
-- load/ save field from file (very basic error handling right now)

function L.LField:LoadFromCSV(filename)
  if self.owner:isFragmented() then
    error('cannot load into fragmented relation', 2)
  end
  if type(filename) ~= 'string' then
    error('LoadFromCSV expected a string argument')
  end
  local fp = C.fopen(filename, 'r')
  if fp == nil then
    error('Cannot read file ' .. filename)
  end

  local btype = self.type:terrabasetype()
  local typeformat = ""
  if btype == int then
    typeformat = "%d"
  elseif btype == uint64 then
    typeformat = "%u"
  elseif btype == bool then
    typeformat = "%d"
  elseif btype == float then
    typeformat = "%f"
  elseif btype == double then
    typeformat = "%lf"
  else
    assert(false, "unrecognized base type in loadCSV")
  end

  local terra LoadCSVFunction(darray : &DLD.C_DLD)
    var d    = darray[0]
    var s    = d.dim_stride
    var dim  = d.dim_size
    var st   = d.type_stride
    var dimt = d.type_dims
    var nt   = dimt[0] * dimt[1]

    var bt  : btype    -- base type
    var c   : int8     -- delimiter in csv, comma

    var ptr = [&btype](d.address)

    for i = 0, dim[0] do
      for j = 0, dim[1] do
        for k = 0, dim[2] do
          var linidx = i*s[0] + j*s[1] + k*s[2]
          for it = 0, dimt[0] do
            for jt = 0, dimt[1] do
              var offset = it*dimt[1] + jt
              C.assert(C.fscanf(fp, typeformat, &bt) == 1)
              C.assert(C.ferror(fp) == 0 and C.feof(fp) == 0)
              ptr[ nt*linidx + offset ] = bt
              --C.memcpy(ptr, &bt, d.type.base_bytes)
              if offset ~= nt-1 then
                c = 0
                while (c ~= 44) do
                  c = C.fgetc(fp)
                  C.assert (  (c == 32 or c == 44) and
                                C.ferror(fp) == 0 and C.feof(fp) == 0,
                              "Expected a comma or a space in CSV file" )
                end
              end
            end
          end
        end
      end
    end
    c = 0
    while (C.feof(fp) == 0) do
      c = C.fgetc(fp)
      if (c > 0 and (c < 9 or (c > 13 and c ~= 32))) then
        C.printf("CSV file %s longer than expected. Expected space or end of file.\n", filename)
        C.exit(-1)
      end
    end
  end
  self:LoadTerraFunction(LoadCSVFunction)
  C.fclose(fp)
end

function L.LField:SaveToCSV(filename, args)
  if self.owner:isFragmented() then
    error('cannot save a fragmented relation', 2)
  end
  if type(filename) ~= 'string' then
    error('SaveToCSV expected a string argument')
  end
  local fp = C.fopen(filename, 'w')
  if fp == nil then
    error('Cannot write to file ' .. filename)
  end
  local precision_str = ""
  if args and args.precision then precision_str = "." .. tostring(args.precision) end

  local btype = self.type:terrabasetype()
  local btype = self.type:terrabasetype()
  local typeformat = ""
  if btype == int then
    typeformat = "%d"
  elseif btype == uint64 then
    typeformat = "%u"
  elseif btype == bool then
    typeformat = "%d"
  elseif btype == float then
    typeformat = "%" .. precision_str .. "f"
  elseif btype == double then
    typeformat = "%" .. precision_str .. "lf"
  end
  local terra SaveCSVFunction(darray : &DLD.C_DLD)
    var d    = darray[0]
    var s    = d.dim_stride
    var dim  = d.dim_size
    var st   = d.type_stride
    var dimt = d.type_dims
    var nt   = dimt[0] * dimt[1]

    var bt  : btype    -- base type
    var ptr = [&btype](d.address)

    for i = 0, dim[0] do
      for j = 0, dim[1] do
        for k = 0, dim[2] do
          var linidx = i*s[0] + j*s[1] + k*s[2]
          for it = 0, dimt[0] do
            for jt = 0, dimt[1] do
              var offset = it*dimt[1] + jt
              --ptr = [&uint8](d.address) + i*s[0] + j*s[1] + k*s[2]
              --ptr = [&uint8](ptr) + it*st[0] + jt*st[1]
              --C.memcpy(&bt, ptr, d.type.base_bytes)
              bt = ptr[ nt*linidx + offset ]
              C.assert(C.fprintf(fp, typeformat, bt) > 0)
              if offset ~= nt-1 then
                C.assert(C.fprintf(fp, ", ") > 0)
              end
            end
          end
          C.assert(C.fprintf(fp, "\n") > 0)
        end
      end
    end
  end
  self:DumpTerraFunction(SaveCSVFunction)
  C.fclose(fp)
end
--]]

-------------------------------------------------------------------------------
-------------------------------------------------------------------------------
--[[  Data Sharing Hooks                                                   ]]--
-------------------------------------------------------------------------------
-------------------------------------------------------------------------------

function L.LField:GetDLD()
  if self.owner:isFragmented() then
    error('Cannot get DLD from fragmented relation', 2)
  end

  -- Because the relation is not fragmented, we're guaranteed that
  -- the data is contiguously stored and can be safely exposed to the
  -- external code

  local typ         = self.type
  local type_dims   = ( typ:ismatrix() and {typ.Nrow,typ.Ncol} ) or
                      ( typ:isvector() and {typ.N,1} ) or
                      {1,1}

  if use_single then
    local location  = self.array:location()
    location        = assert( (location == L.CPU and DLD.CPU) or
                              (location == L.GPU and DLD.GPU) )
    local dims      = self.owner:Dims()

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

      address         = self:DataPtr(),
      dim_size        = dim_size,
      dim_stride      = dim_stride,
    }
  elseif use_legion then
    error('DLD TO BE REVISED for LEGION')
  else
    assert(use_single or use_legion)
  end

  --[[
  -- TODO(Chinmayee): use concrete size here?
  if use_single then
    local dld = DLD.new({
      address         = self:DataPtr(),
      location        = tostring(self.array:location()),
      type            = self.type,
      dims            = self.owner:Dims(),
      compact         = true,
    })
    return dld
  elseif use_legion then
    local dld = DLD.new({
      type = self.type,
    })
    return dld
  end]]
end


-------------------------------------------------------------------------------
-------------------------------------------------------------------------------
--[[  ELASTIC RELATIONS                                                    ]]--
-------------------------------------------------------------------------------
-------------------------------------------------------------------------------


function L.LRelation:_INTERNAL_Resize(new_concrete_size, new_logical)
  if not self:isElastic() then
    error('Can only resize ELASTIC relations', 2)
  end
  if use_legion then error("Can't resize while using Legion", 2) end

  self._is_live_mask.array:resize(new_concrete_size)
  for _,field in ipairs(self._fields) do
    field.array:resize(new_concrete_size)
  end
  self._concrete_size = new_concrete_size
  if new_logical then self._logical_size = new_logical end
end

-------------------------------------------------------------------------------
--[[  Insert / Delete                                                      ]]--
-------------------------------------------------------------------------------

-- returns a useful error message 
function L.LRelation:UnsafeToDelete()
  if not self:isElastic() then
    return "Cannot delete from relation "..self:Name()..
           " because it's not ELASTIC"
  end
  if self:hasSubsets() then
    return 'Cannot delete from relation '..self:Name()..
           ' because it has subsets'
  end
end

function L.LRelation:UnsafeToInsert(record_type)
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

function L.LRelation:_INTERNAL_MarkFragmented()
  if not self:isElastic() then
    error("INTERNAL: Cannot Fragment a non-elastic relation")
  end
  rawset(self, '_is_fragmented', true)
end

TOTAL_DEFRAG_TIME = 0
function L.LRelation:Defrag()
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
    on_gpu[i]   = field.array:location() == L.GPU
    any_on_gpu  = true
  end
  if self._is_live_mask.array:location() == L.GPU then
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
      if on_gpu[i] then field:MoveTo(L.CPU) end
    end
    if live_gpu then self._is_live_mask:MoveTo(L.CPU) end
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
    ptrargs[i] = field:DataPtr()
  end
  ptrargs[#self._fields+1] = self._is_live_mask:DataPtr()

  -- run the defrag func
  defrag_func(self:ConcreteSize(), unpack(ptrargs))

  -- move back to GPU if necessary
  if any_on_gpu then
    for i,field in ipairs(self._fields) do
      if on_gpu[i] then field:MoveTo(L.GPU) end
    end
    if live_gpu then self._is_live_mask:MoveTo(L.GPU) end
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


function L.LRelation:SetPartitions(num_partitions)
  if self._total_partitions ~= nil then
    error("Partitioning for " .. self._name .. " is already set", 2)
  end
  local ndims = self:nDims()
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
end

function L.LRelation:TotalPartitions()
  return self._total_partitions
end

function L.LRelation:NumPartitions()
  return self._num_partitions
end

function L.LRelation:IsPartitioningSet()
  return (self._total_partitions ~= nil)
end

local ColorPlainIndexSpaceDisjoint = nil
if use_legion then
  ColorPlainIndexSpaceDisjoint = terra(darray : &DLD.C_DLD, num_colors : uint)
    var d = darray[0]
    var b = d.dims[0]
    var s = d.stride[0]
    var partn_size = b / num_colors
    if num_colors * partn_size < b then partn_size = partn_size + 1 end
    for i = 0, b do
      var ptr = [&LW.legion_color_t]([&uint8](d.address) + i*s)
      @ptr = i / partn_size
    end
  end
end

-- creates a disjoint partitioning on the relation
function L.LRelation:GetOrCreateDisjointPartitioning()
  -- check if there is a disjoint partition
  if self._disjoint_partitioning then
    return self._disjoint_partitioning
  end
  -- else creaate disjoint partition
  local partn = nil
  if self:isGrid() then
    -- STRUCTURED GRIDS
    -- create block partition
    partn = self._logical_region_wrapper:CreateBlockPartitions()
  else
    -- PLAIN/ GROUPED/ ELASTIC
    -- add a coloring field to logical region
    rawset(self, '_disjoint_coloring',
           L.LField.New(self, '_disjoint_coloring', L.color_type))
    -- set the coloring field
    self._disjoint_coloring:Load(ColorPlainIndexSpaceDisjoint,
                                 self:TotalPartitions())
    -- create index partition using the coloring field and save it
    partn = 
      self._logical_region_wrapper:CreatePartitionsByField(self._disjoint_coloring)
  end
  rawset(self, '_disjoint_partitioning', partn)
  return partn
end
