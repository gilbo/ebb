-- file/module namespace table
local R = {}
package.loaded["compiler.relations"] = R

local use_legion = not not rawget(_G, '_legion_env')
local use_single = not use_legion

local L = require "compiler.lisztlib"
local T = require "compiler.types"
local C = require "compiler.c"
local DLD = require "compiler.dld"

local PN = require "lib.pathname"

local DynamicArray = use_single and
                     require('compiler.rawdata').DynamicArray
local LW = use_legion and require "compiler.legionwrap"

local valid_name_err_msg =
  "must be valid Lua Identifiers: a letter or underscore,"..
  " followed by zero or more underscores, letters, and/or numbers"
L.valid_name_err_msg = {
  relation = "Relation names " .. valid_name_err_msg,
  field    = "Field names " .. valid_name_err_msg,
  subset   = "Subset names " .. valid_name_err_msg
}
function L.is_valid_lua_identifier(name)
  if type(name) ~= 'string' then return false end

  -- regex for valid LUA identifiers
  if not name:match('^[_%a][_%w]*$') then return false end

  return true
end

-------------------------------------------------------------------------------
--[[ LRelation methods                                                     ]]--
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
--  [dim  = {45,90}, ]  -- IF mode == 'GRID'
-- }
function L.NewRelation(params)
  -- CHECK the parameters coming in
  if type(params) ~= 'table' then
    error("NewRelation() expects a table of named arguments", 2)
  elseif type(params.name) ~= 'string' then
    error("NewRelation() expects 'name' string argument", 2)
  end
  if not L.is_valid_lua_identifier(params.name) then
    error(L.valid_name_err_msg.relation, 2)
  end
  local mode = params.mode or 'PLAIN'
  if not params.mode and params.dim then mode = 'GRID' end
  if mode ~= 'PLAIN' and mode ~= 'GRID'  and mode ~= 'ELASTIC' then
    error("NewRelation(): Bad 'mode' argument.  Was expecting\n"..
          "  PLAIN, GRID, or ELASTIC", 2)
  end
  if mode == 'GRID' then
    if type(params.dim) ~= 'table' or
       (#params.dim ~= 2 and #params.dim ~= 3)
    then
      error("NewRelation(): Grids must specify 'dim' argument; "..
            "a table of 2 to 3 numbers specifying grid size", 2)
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

    _fields    = terralib.newlist(),
    _subsets   = terralib.newlist(),
    _macros    = terralib.newlist(),
    _functions = terralib.newlist(),

    _incoming_refs = {}, -- used for walking reference graph
  },
  L.LRelation)

  -- store mode dependent values
  local size = params.size
  if mode == 'GRID' then
    size = 1
    rawset(rel, '_dim', {})
    for i,n in ipairs(params.dim) do
      rel._dim[i] = n
      size = size * n
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
    error('TEMPORARY RELATIONS ARE BROKEN FOR LEGION')
    -- create a logical region.
    local dom_params =
    {
      relation = rel,
      n_rows   = size,
    }
    local logical_region_wrapper = LW.NewLogicalRegion(dom_params)
    rawset(rel, '_logical_region_wrapper', logical_region_wrapper)

    -- create a logical region.
    local dom_params =
    {
      relation = rel,
      dimensions = dim,
      bounds = bounds
    }
    local logical_region_wrapper = LW.NewGridLogicalRegion(dom_params)
    rawset(rel, '_logical_region_wrapper', logical_region_wrapper)
  end

  return rel
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
  for i,n in self._dims do dimret[i] = n end
  return dimret
end

--[[
function L.NewRelation(size, name)
    -- error check
  if not name or type(name) ~= "string" then
    error("NewRelation() expects a string as the 2nd argument", 2)
  end
  if not L.is_valid_lua_identifier(name) then
    error(L.valid_name_err_msg.relation, 2)
  end

  -- construct and return the relation
  local rel = setmetatable( {
    _concrete_size = size,
    _logical_size  = size,

    _fields    = terralib.newlist(),
    _subsets   = terralib.newlist(),
    _macros    = terralib.newlist(),
    _functions = terralib.newlist(),

    _incoming_refs = {}, -- used for walking reference graph
    _name      = name,
    _typestate = {
      --groupedby   = false, -- use _grouping entry's presence instead
      fragmented  = false,
      --has_subsets = false, -- use #_subsets
      grid        = false,
    },
  },
  L.LRelation)

  if use_single then
    -- create a mask to track which rows are live.
    rawset(rel, '_is_live_mask', L.LField.New(rel, '_is_live_mask', L.bool))
    rel._is_live_mask:Load(true)
  elseif use_legion then
    -- create a logical region.
    local dom_params =
    {
      relation = rel,
      n_rows   = size,
    }
    local logical_region_wrapper = LW.NewLogicalRegion(dom_params)
    rawset(rel, '_logical_region_wrapper', logical_region_wrapper)
  end

  return rel
end

-- Create a relation that has an underlying grid data layout
function L.NewGridRelation(name, params)
    -- error check
  if not name or type(name) ~= "string" then
    error("NewRelation() expects a string as the 2nd argument", 2)
  end
  if not L.is_valid_lua_identifier(name) then
    error(L.valid_name_err_msg.relation, 2)
  end

  local dim = #params.bounds
  local bounds = params.bounds
  if not (dim == 1 or dim == 2 or dim == 3) then
    error("Invalid dimension size " .. tostring(dim), 3)
  end
  local size = 1
  for d = 1, dim do
    size = size * bounds[d]
  end

  -- construct and return the relation
  local rel = setmetatable( {
    _concrete_size = size,
    _logical_size  = size,

    _fields    = terralib.newlist(),
    _subsets   = terralib.newlist(),
    _macros    = terralib.newlist(),
    _functions = terralib.newlist(),

    _incoming_refs = {}, -- used for walking reference graph
    _name      = name,
    _typestate = {
      --groupedby   = false, -- use _grouping entry's presence instead
      fragmented  = false,
      --has_subsets = false, -- use #_subsets
      grid        = true,
      dimensions  = dim,
    },
  },
  L.LRelation)

  if use_single then
    -- create a mask to track which rows are live.
    rawset(rel, '_is_live_mask', L.LField.New(rel, '_is_live_mask', L.bool))
    rel._is_live_mask:Load(true)
  elseif use_legion then
    -- create a logical region.
    local dom_params =
    {
      relation = rel,
      dimensions = dim,
      bounds = bounds
    }
    local logical_region_wrapper = LW.NewGridLogicalRegion(dom_params)
    rawset(rel, '_logical_region_wrapper', logical_region_wrapper)
  end
  return rel
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
]]

function L.LRelation:ResizeConcrete(new_size)
  if not self:isElastic() then
    error('Can only resize ELASTIC relations', 2)
  end
  if use_legion then error("Can't resize while using Legion", 2) end

  self._is_live_mask.array:resize(new_size)
  for _,field in ipairs(self._fields) do
    field.array:resize(new_size)
  end
  self._concrete_size = new_size
end

--function L.LRelation:isFragmented()
--  return self._typestate.fragmented
--end
--function L.LRelation:isCompact()
--  return not self._typestate.fragmented
--end
function L.LRelation:hasSubsets()
  return #self._subsets ~= 0
end
--function L.LRelation:isGrouped()
--  return self._grouping ~= nil
--end

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
  if not L.is_valid_lua_identifier(name) then
    error(L.valid_name_err_msg.field, 2)
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
  if not L.is_valid_lua_identifier(name) then
    error(L.valid_name_err_msg.field, 2)
  end
  if self[name] then
    error("Cannot create a new field-function with name '"..name.."'  "..
          "That name is already being used.", 2)
  end

  if not L.is_user_func(userfunc) then
    error("NewFieldFunction() expects a Liszt Function "..
          "as the 2nd argument", 2)
  end

  rawset(self, name, userfunc)
  self._functions:insert(userfunc)
  return userfunc
end

function L.LRelation:GroupBy(name)
  if not self:isPlain() then
    error("GroupBy(): Cannot group a relation "..
          "unless it's a PLAIN relation", 2)
  end

  local key_field = self[name]
  local live_mask = self._is_live_mask
  if self:isGrouped() then
    error("GroupBy(): Relation is already grouped", 2)
  elseif not L.is_field(key_field) then
    error("GroupBy(): Could not find a field named '"..name.."'", 2)
  elseif not key_field.type:isScalarKey() then
    error("GroupBy(): Grouping by non-scalar-key fields is "..
          "prohibited.", 2)
  elseif key_field.type.ndims > 1 then
    error("GroupBy(): Grouping by a grid relation "..
          "is currently unsupported", 2)
  end

  if use_legion then
    error('GroupBy(): Grouping unimplemented for Legion currently', 2)
  end

  -- WARNING: The sizing policy will break with dead rows
  --if self:isFragmented() then
  --  error("GroupBy(): Cannot group a fragmented relation", 2)
  --end
  --if key_field.type.relation:isFragmented() then
  --  error("GroupBy(): Cannot group by a fragmented relation", 2)
  --end

  local num_keys = key_field.type.relation:ConcreteSize() -- # possible keys
  local num_rows = key_field:ConcreteSize()
  rawset(self,'_grouping', {
    key_field = key_field,
    index = L.LIndex.New{
      owner=self,
      terra_type = L.addr_terra_types[key_field.type.ndims],
      processor = L.default_processor,
      name='groupby_'..key_field:Name(),
      size=num_keys+1
    },
  })

  self._grouping.index._array:write_ptr(function(indexdata)
    local prev,pos = 0,0
    key_field.array:read_ptr(function(keyptr)
      for i = 0, num_keys - 1 do
        indexdata[i].a[0] = pos
        while keyptr[pos].a[0] == i and pos < num_rows do
          if keyptr[pos].a[0] < prev then
            self._grouping.index:Release()
            self._grouping = nil
            error("GroupBy(): Key field '"..name.."' is not sorted.")
          end
          prev,pos = keyptr[pos].a[0], pos+1
        end
      end
    end) -- key_field read
    assert(pos == num_rows)
    indexdata[num_keys].a[0] = pos
  end) -- indexdata write

  -- record reference from this relation to the relation it's grouped by
  key_field.type.relation._incoming_refs[self] = 'group'
end

function L.LRelation:MoveTo( proc )
  if use_legion then error("MoveTo() unsupported using Legion", 2) end
  if proc ~= L.CPU and proc ~= L.GPU then
    error('must specify valid processor to move to', 2)
  end

  self._is_live_mask:MoveTo(proc)
  for _,f in ipairs(self._fields) do f:MoveTo(proc) end
  for _,s in ipairs(self._subsets) do s:MoveTo(proc) end
  if self._grouping then self._grouping.index:MoveTo(proc) end
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
--[[ Insert / Delete                                                       ]]--
-------------------------------------------------------------------------------
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
-------------------------------------------------------------------------------
--[[ Indices:                                                              ]]--
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
    index._array:write_ptr(function(ptr)
      for i=1,#params.data do
        ptr[i-1].a[0] = params.data[i]
      end
    end) -- write_ptr
  end

  return index
end

function L.LIndex:DataPtr()
  return self._array:ptr()
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
--[[ Subsets:                                                              ]]--
-------------------------------------------------------------------------------
-------------------------------------------------------------------------------

function L.LSubset:Relation()
  return self._owner
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
  if not L.is_valid_lua_identifier(name) then
    error(L.valid_name_err_msg.subset, 2)
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
  if use_legion then
    error("NewSubsetFromFunction(): subsets are currently unsupported "..
          "using legion", 2)
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
  boolmask:LoadFunction(function(i)
    local val = predicate(i)
    if val then
      table.insert(index_tbl, i)
      subset_size = subset_size + 1
    end
    return val
  end)

  if subset_size > SUBSET_CUTOFF then
  -- USE MASK
    subset._boolmask = boolmask
  else
  -- USE INDEX
    subset._index = L.LIndex.New{
      owner=self,
      terra_type = L.key(self):terraType(),
      name=name..'_subset_index',
      data=index_tbl
    }
    boolmask:ClearData() -- free memory
  end

  return subset
end


-------------------------------------------------------------------------------
-------------------------------------------------------------------------------
--[[ Fields:                                                               ]]--
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
    local logical_region_wrapper = rel._logical_region_wrapper
    field.fid = logical_region_wrapper:AllocateField(typ)
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
  return self.array:ptr()
end
function L.LField:Relation()
  return self.owner
end

function L.LRelation:NewField (name, typ)  
  if not name or type(name) ~= "string" then
    error("NewField() expects a string as the first argument", 2)
  end
  if not L.is_valid_lua_identifier(name) then
    error(L.valid_name_err_msg.field, 2)
  end
  if self[name] then
    error("Cannot create a new field with name '"..name.."'  "..
          "That name is already being used.", 2)
  end
  
  if L.is_relation(typ) then
    typ = L.key(typ)
  end
  if not T.isLisztType(typ) or not typ:isFieldType() then
    error("NewField() expects a Liszt type or "..
          "relation as the 2nd argument", 2)
  end

  -- prevent the creation of key fields pointing into elastic relations
  if typ:isKey() then
    local rel = typ:baseType().relation
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
  if typ:isKey() then
    typ:baseType().relation._incoming_refs[field] = 'key_field'
  end

  return field
end

-- TODO: Hide this function so it's not public
function L.LField:Allocate()
  if use_legion then error('No Allocate() using legion') end
  if not self.array then
    self.array = DynamicArray.New{
      size = self:ConcreteSize(),
      type = self:Type():terraType()
    }
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
  if self.owner._grouping and
     self.owner._grouping.key_field == self
  then
    self.owner._grouping.index:Release()
    self.owner._grouping = nil
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
  if use_legion then error('No Swap() using legion') end
  local f1 = self[f1_name]
  local f2 = self[f2_name]
  if not L.is_field(f1) then
    error('Could not find a field named "'..f1_name..'"', 2) end
  if not L.is_field(f2) then
    error('Could not find a field named "'..f2_name..'"', 2) end
  if f1.type ~= f2.type then
    error('Cannot Swap() fields of different type', 2)
  end

  local tmp = f1.array
  f1.array = f2.array
  f2.array = tmp
end

function L.LRelation:Copy( p )
  if use_legion then error('No Copy() using legion') end
  if type(p) ~= 'table' or not p.from or not p.to then
    error("relation:Copy() should be called using the form\n"..
          "  relation:Copy{from='f1',to='f2'}", 2)
  end
  local from = self[p.from]
  local to   = self[p.to]
  if not L.is_field(from) then
    error('Could not find a field named "'..p.from..'"', 2) end
  if not L.is_field(to) then
    error('Could not find a field named "'..p.to..'"', 2) end
  if from.type ~= to.type then
    error('Cannot Copy() fields of different type', 2)
  end

  if not from.array then
    error('Cannot Copy() from field with no data', 2) end
  if not to.array then
    to:Allocate()
  end

  to.array:copy(from.array)
end


--[[ convert lua tables or numbers to
-- Terra structs used to represent key values
local function convert_key(key_val, typ)
  if 1 == typ.ndims then
    return terralib.new(typ:terraType(), { { key_val } })
  elseif type(key_val) == 'table' and #key_val == typ.ndim then
    return terralib.new(typ:terraType(), {key_val})
  else
    error('Loaded Value was not recognizable as '..
          'Key of dimension '..tostring(typ.ndims), 3)
  end
end

-- convert lua tables or LVectors to
-- Terra structs used to represent vectors
local function convert_vec(vec_val, typ)
  if type(vec_val) == 'table' and #vec_val == typ.N then
    return terralib.new(typ:terraType(), {vec_val})
  else
    error('Loaded Value was not recognizable as compatible Vector', 3)
    --return nil
  end
end

-- convert lua tables to Terra structs used to represent matrices
local function convert_mat(mat_val, typ)
  if type(mat_val) == 'table' and #mat_val == typ.Nrow then
    local terraval = terralib.new(typ:terraType())
    for r=1,#mat_val do
      local row = mat_val[r]
      if type(row) ~= 'table' or #row ~= typ.Ncol then
        error('Loaded Value was not recognizable as compatible Matrix', 3)
      end
      for c=1,#row do terraval.d[r][c] = row[c] end
    end
    return terraval
  else
    error('Loaded Value was not recognizable as compatible Matrix', 3)
  end
end
]]

function L.LField:LoadFunction(lua_callback)
  if self.owner:isFragmented() then
    error('cannot load into fragmented relation', 2)
  end

  if use_legion then
    -- Ok, we need to map some stuff down here
    local scanner = LW.NewControlScanner {
      logical_region = self.owner._logical_region_wrapper.handle,
      n_rows         = self:Size(),
      privilege      = LW.WRITE_ONLY,
      fields         = {self.fid},
    }
    scanner:scan(function(i, dataptr)
      local lval = lua_callback(i)
      local tval = T.luaToLisztVal(lval)
      terralib.cast(&(self.type:terraType()), dataptr)[0] = tval
    end)
    scanner:close()
  elseif use_single then
    self:Allocate()

    self.array:write_ptr(function(dataptr)
      for i = 0, self:Size() - 1 do
        local val  = lua_callback(i)
        if not T.luaValConformsToType(val, self.type) then
          error("lua value does not conform to field type "..
                tostring(self.type), 3)
        end
        dataptr[i] = T.luaToLisztVal(val, self.type)
      end
      --if self.type:isSmallMatrix() then
      --  for i = 0, self:Size() - 1 do
      --    local val = lua_callback(i)
      --    dataptr[i] = convert_mat(val, self.type)
      --  end
      --elseif self.type:isVector() then
      --  for i = 0, self:Size() - 1 do
      --    local val = lua_callback(i)
      --    dataptr[i] = convert_vec(val, self.type)
      --  end
      --elseif self.type:isScalarKey() then
      --  for i = 0, self:Size() - 1 do
      --    local val = lua_callback(i)
      --    dataptr[i] = convert_key(val, self.type)
      --  end
      --else
      --  for i = 0, self:Size() - 1 do
      --    dataptr[i] = lua_callback(i)
      --  end
      --end
    end) -- write_ptr
  end
end

function L.LField:LoadList(tbl)
  if self.owner:isFragmented() then
    error('cannot load into fragmented relation', 2)
  end
  if type(tbl) ~= 'table' then
    error('bad type passed to LoadList().  Expecting a table', 2)
  end
  if #tbl ~= self:Size() then
    error('argument array has the wrong number of elements: '..
          tostring(#tbl)..
          ' (was expecting '..tostring(self:Size())..')', 2)
  end
  self:LoadFunction(function(i)
    return tbl[i+1]
  end)
end

-- TODO: DEPRECATED FORM.  (USE DLD?)
function L.LField:LoadFromMemory(mem)
  if self.owner:isFragmented() then
    error('cannot load into fragmented relation', 2)
  end
  if use_legion then
    error('Load from memory while using Legion is unimplemented', 2)
  end
  self:Allocate()

  -- avoid extra copies by wrapping and using the standard copy
  local wrapped = DynamicArray.Wrap{
    size = self:ConcreteSize(),
    type = self.type:terraType(),
    data = mem,
    processor = L.CPU,
  }
  self.array:copy(wrapped)
end

function L.LField:LoadConstant(constant)
  if self.owner:isFragmented() then
    error('cannot load into fragmented relation', 2)
  end

  self:LoadFunction(function()
    return constant
  end)
end

-- generic dispatch function for loads
function L.LField:Load(arg)
  if self.owner:isFragmented() then
    error('cannot load into fragmented relation', 2)
  end
  -- load from lua callback
  if      type(arg) == 'function' then
    return self:LoadFunction(arg)
  elseif  type(arg) == 'cdata' then
    if use_legion then
      error('Load from memory while using Legion is unimplemented', 2)
    end
    local typ = terralib.typeof(arg)
    if typ and typ:ispointer() then
      return self:LoadFromMemory(arg)
    end
  elseif  type(arg) == 'string' or PN.is_pathname(arg) then
    if use_legion then
      error('Load from file while using Legion is unimplemented', 2)
    end
    return self:LoadFromFile(arg)
  elseif  type(arg) == 'table' then
    if (self.type:isScalarKey() and #arg == self.type.ndims) or
       (self.type:isVector() and #arg == self.type.N) or
       (self.type:isSmallMatrix() and #arg == self.type.Nrow) 
    then
      return self:LoadConstant(arg)
    else
      return self:LoadList(arg)
    end
  end
  -- default to trying to load as a constant
  return self:LoadConstant(arg)
end



-- convert lua tables or LVectors to
-- Terra structs used to represent vectors
--local function terraval_to_lua(val, typ)
--  if typ:isSmallMatrix() then
--    local mat = {}
--    local btyp = typ:baseType()
--    for i=1,typ.Nrow do
--      mat[i] = {}
--      for j=1,typ.Ncol do
--        mat[i][j] = terraval_to_lua(val.d[i-1][j-1], btyp) end
--    end
--    return mat
--  elseif typ:isVector() then
--    local vec = {}
--    for i = 1, typ.N do
--      vec[i] = terraval_to_lua(val.d[i-1], typ:baseType())
--    end
--    return vec
--  elseif typ:isNumeric() then
--    return tonumber(val)
--  elseif typ:isLogical() then
--    if tonumber(val) == 0 then return false else return true end
--  elseif typ:isScalarKey() then
--    if typ.ndims == 1 then
--      return tonumber(val.a[0])
--    elseif typ.ndims == 2 then
--      return { tonumber(val.a[0]), tonumber(val.a[1]) }
--    elseif typ.ndims == 3 then
--      return { tonumber(val.a[0]), tonumber(val.a[1]), tonumber(val.a[2]) }
--    else
--      error('INTERNAL: Cannot have > 3 dimensional keys')
--    end
--  else
--    error('unhandled terra_to_lua conversion')
--  end
--end


-- helper to dump multiple fields jointly
function L.LRelation:JointDump(fields, lua_callback)
  if self:isFragmented() then
    error('cannot dump from fragmented relation', 2)
  end
  local size = self:ConcreteSize()
  local nfields = #fields
  local typs = {}
  local ptrs = {}

  -- main loop part
  local loop = function()
    for i=0,size-1 do
      local vals = {}
      for k=1,nfields do
        vals[k] = T.lisztToLuaVal(ptrs[k][i], typs[k])
      end
      lua_callback(i, unpack(vals))
    end
  end

  for k=1,nfields do
    local fname = fields[k]
    local f     = self[fname]
    typs[k]     = f.type
    local loopcapture = loop -- THIS IS NEEDED TO STOP INF. RECURSION
    local outerloop = function()
      f.array:read_ptr(function(dataptr)
        ptrs[k] = dataptr
        loopcapture()
      end)
    end
    loop = outerloop
  end

  loop()
end

-- callback(i, val)
--      i:      which row we're outputting (starting at 0)
--      val:    the value of this field for the ith row
function L.LField:DumpFunction(lua_callback)
  if self.owner:isFragmented() then
    error('cannot dump from fragmented relation', 2)
  end
  -- TODO: replace with the below call?
  self.array:read_ptr(function(dataptr)
    for i = 0, self:ConcreteSize()-1 do
      --local val = terraval_to_lua(dataptr[i], self.type)
      local val = T.lisztToLuaVal(dataptr[i], self.type)
      lua_callback(i, val)
    end
  end) -- read_ptr
  --self.owner:JointDump({self:Name()}, lua_callback)
end

function L.LField:DumpToList()
  if self.owner:isFragmented() then
    error('cannot dump from fragmented relation', 2)
  end
  local arr = {}
  self:DumpFunction(function(i,val)
    arr[i+1] = val
  end)
  return arr
end


--local function valtostring(val, typ)
--  if not typ:isScalarKey() then
--    local str = tostring(val):gsub('ULL','')
--    return str
--  else
--    local str = tostring(val.a[0]):gsub('ULL','')
--    if typ.ndims == 1 then
--      return str
--    else
--      local t2 = tostring(val.a[1]):gsub('ULL','')
--      str = '{ ' .. str .. ', ' .. t2
--      if typ.ndims == 2 then
--        return str .. ' }'
--      elseif typ.ndims == 3 then
--        local t3 = tostring(val.a[2]):gsub('ULL','')
--        return str .. ', ' .. t3 .. ' }'
--      else
--        error("INTERNAL: keys cannot be >3 dimensional")
--      end
--    end
--  end
--end

function L.LField:print()
  if use_legion then
    error('BROKEN; rewrite using single DUMP choke point')
  end
  print(self.name..": <" .. tostring(self.type:terraType()) .. '>')
  if not self.array then
    print("...not initialized")
    return
  else
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

  self.owner:JointDump(
  {'_is_live_mask', self:Name()},
  function (i, islive, datum)
    local alive = ' .'
    if not islive then alive = ' x' end

    if self.type:isSmallMatrix() then
      local s = ''
      for c=1,self.type.Ncol do s = s .. flattenkey(datum[1][c]) .. ' ' end
      print("", tostring(i) .. alive, s)

      for r=2,self.type.Nrow do
        local s = ''
        for c=1,self.type.Ncol do s = s .. flattenkey(datum[r][c]) .. ' ' end
        print("", "", s)
      end

    elseif self.type:isVector() then
      local s = ''
      for k=1,self.type.N do s = s .. flattenkey(datum[k]) .. ' ' end
      print("", tostring(i) .. alive, s)

    else
      print("", tostring(i) .. alive, flattenkey(datum))
    end
  end)

--  local N = self.owner:ConcreteSize()
--  local livemask = self.owner._is_live_mask
--
--  livemask.array:read_ptr(function(liveptr)
--  self.array:read_ptr(function(dataptr)
--    local alive
--    if self.type:isSmallMatrix() then
--      for i = 0, N-1 do
--        if liveptr[i] then alive = ' .'
--        else                alive = ' x' end
--        local s = ''
--        for c = 0, self.type.Ncol-1 do
--          s = s .. valtostring(dataptr[i].d[0][c], self.type) .. ' '
--        end
--        print("", tostring(i) .. alive, s)
--        for r = 1, self.type.Nrow-1 do
--          local s = ''
--          for c = 0, self.type.Ncol-1 do
--            s = s .. valtostring(dataptr[i].d[r][c], self.type) .. ' '
--          end
--          print("", "", s)
--        end
--      end
--    elseif self.type:isVector() then
--      for i = 0, N-1 do
--        if liveptr[i] then alive = ' .'
--        else                alive = ' x' end
--        local s = ''
--        for j = 0, self.type.N-1 do
--          s = s .. valtostring(dataptr[i].d[j], self.type) .. ' '
--        end
--        print("", tostring(i) .. alive, s)
--      end
--    else
--      for i = 0, N-1 do
--        if liveptr[i] then alive = ' .'
--        else                alive = ' x' end
--        local t = valtostring(dataptr[i], self.type)
--        print("", tostring(i) .. alive, t)
--      end
--    end
--  end) -- dataptr
--  end) -- liveptr
end


-------------------------------------------------------------------------------
-------------------------------------------------------------------------------
--[[ Data Sharing Hooks                                                    ]]--
-------------------------------------------------------------------------------
-------------------------------------------------------------------------------


function L.LField:getDLD()
  error("DLD NEEDS REVISION FOR MODES")
  if self.owner:isFragmented() then
    error('cannot get DLD from fragmented relation', 2)
  end
  if not self.type:baseType():isPrimitive() then
    error('Can only return DLDs for fields with primitive base type')
  end

  local terra_type = self.type:terraBaseType()
  local dims = {}
  if self.type:isVector() then
    dims = {self.type.N}
  elseif self.type:isSmallMatrix() then
    dims = {self.type.Nrow, self.type.Ncol}
  end

  local dld = DLD.new({
    location        = tostring(self.array:location()),
    type            = terra_type,
    type_dims       = dims,
    logical_size    = self.owner:ConcreteSize(),
    data            = self:DataPtr(),
    compact         = true,
  })
  return dld
end



-- Defrag

function L.LRelation:_INTERNAL_MarkFragmented()
  if not self:isElastic() then
    error("INTERNAL: Cannot Fragment a non-elastic relation")
  end
  rawset(self, '_is_fragmented', true)
end

function L.LRelation:Defrag()
  error("BROKEN IN SUBTLE WAYS")
  if not self:isElastic() then
    error("Defrag(): Cannot Defrag a non-elastic relation")
  end
  -- TODO: MAKE IDEMPOTENT FOR EFFICIENCY

  -- Check that all fields are on CPU:
  for _,field in ipairs(self._fields) do
    if field.array:location() ~= L.CPU then
      error('Defrag on GPU unimplemented')
    end
  end
  if self._is_live_mask.array:location() ~= L.CPU then
    error('Defrag on GPU unimplemented')
  end

  -- ok, build a terra function that we can execute to compact
  -- we can cache it!
  local defrag_func = self._cpu_defrag_func
  if not defrag_func then
    -- read and write heads for copy
    local dst = symbol(L.addr:terraType(), 'dst')
    local src = symbol(L.addr:terraType(), 'src')

    -- create data copying chunk for fields
    local do_copy = quote end
    for _,field in ipairs(self._fields) do
      local ptr = field:DataPtr()
      do_copy = quote
        do_copy
        ptr[dst] = ptr[src]
      end
    end

    local liveptr = self._is_live_mask:DataPtr()
    local addrtype = L.addr:terraType()
    defrag_func = terra ( concrete_size : addrtype )
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
  end

  -- run the defrag func
  defrag_func(self:ConcreteSize())

  -- now cleanup by resizing the relation
  local logical_size = self:Size()
  -- since the data is now compact, we can shrink down the size
  self:ResizeConcrete(logical_size)

  -- mark as compact
  rawset(self, '_is_fragmented', false)
end
