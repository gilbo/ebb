
-- file/module namespace table
local R = {}
package.loaded["compiler.relations"] = R

local L = terralib.require "compiler.lisztlib"
local T = terralib.require "compiler.types"
local C = terralib.require "compiler.c"
local DLD = terralib.require "compiler.dld"

local PN = terralib.require "lib.pathname"

local JSON = require('compiler.JSON')


terra allocateAligned(alignment : uint64, size : uint64)
    var r : &opaque
    C.posix_memalign(&r,alignment,size)
    return r
end
-- vector(double,4) requires 32-byte alignment
-- note: it _is safe_ to free memory allocated this way with C.free
local function MallocArray(T,N)
    return terralib.cast(&T,allocateAligned(32,N * terralib.sizeof(T)))
end


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
        _size      = size,
        _fields    = terralib.newlist(),
        _subsets   = terralib.newlist(),
        _macros    = terralib.newlist(),
        _functions = terralib.newlist(),
        _name      = name,
    },
    L.LRelation)
    return rel
end

function L.LRelation:Size()
    return self._size
end
function L.LRelation:Name()
    return self._name
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
          "(did you mean to call self:New...?)", 2)
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
    local key_field = self[name]
    if self._grouping then
        error("GroupBy(): Relation is already grouped", 2)
    elseif not L.is_field(key_field) then
        error("GroupBy(): Could not find a field named '"..name.."'", 2)
    elseif not key_field.type:isRow() then
        error("GroupBy(): Grouping by non-row-type fields is "..
              "currently prohibited.", 2)
    end

    -- WARNING: The sizing policy will break once we start supporting dead rows
    local num_keys = key_field.type.relation:Size()
    local num_rows = key_field:Size()
    rawset(self,'_grouping', {
        key_field = key_field,
        index = L.LIndex.New(self, 'groupby_'..key_field:Name(), num_keys+1)
    })
    local indexdata = self._grouping.index._data

    local prev,pos = 0,0
    for i = 0, num_keys - 1 do
        indexdata[i] = pos
        while key_field.data[pos] == i and pos < num_rows do
            if key_field.data[pos] < prev then
                self._grouping.index:Release()
                self._grouping = nil
                error("GroupBy(): Key field '"..name.."' is not sorted.")
            end
            prev,pos = key_field.data[pos], pos+1
        end
    end
    assert(pos == num_rows)
    indexdata[num_keys] = pos
end



function L.LRelation:print()
    print(self._name, "size: ".. tostring(self._size))
    for i,f in ipairs(self._fields) do
        f:print()
    end
end


-------------------------------------------------------------------------------
-------------------------------------------------------------------------------
--[[ Indices:                                                              ]]--
-------------------------------------------------------------------------------
-------------------------------------------------------------------------------

function L.LIndex.New(rel, name, size_or_table)
    local index = setmetatable({
        _owner = rel,
        _name  = name,
    }, L.LIndex)
    if type(size_or_table) == 'number' then
        index:ReAllocate(size_or_table)
    else
        index:ReAllocate(#size_or_table)
        for i=1,#size_or_table do
            index._data[i-1] = size_or_table[i]
        end
    end

    return index
end

function L.LIndex:Relation()
    return self._owner
end

function L.LIndex:ReAllocate(size)
    self:Release()

    local taddr = L.addr:terraType()
    self._data = MallocArray(taddr, size)
    self._size = size
end

function L.LIndex:Release()
    if self._data then
        C.free(self._data)
        self._data = nil
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
        subset._index = L.LIndex.New(self, name..'_subset_index', index_tbl)
        boolmask:ClearData() -- free memory
    end

    return subset
end

--function L.LRelation:NewSubsetFromField (name, field_name)
--    if not name or type(name) ~= "string" then
--        error("NewSubsetFromField() "..
--              "expects a string as the first argument", 2)
--    end
--    if not L.is_valid_lua_identifier(name) then
--        error(L.valid_name_err_msg.subset, 2)
--    end
--    if self[name] then
--        error("Cannot create a new subset with name '"..name.."'  "..
--              "That name is already being used.", 2)
--    end
--
--    local field = self[field_name]
--    if not L.is_field(field) or not field.type == L.bool then
--        error("NewSubsetFromField(): second argument "..
--              "'"..tostring(field_name).."' is not a bool-type field name", 2)
--    end
--
--    self:NewSubsetFromFunction(function(i)
--        return field.data[i]
--    end)
--end

-------------------------------------------------------------------------------
-------------------------------------------------------------------------------
--[[ Fields:                                                               ]]--
-------------------------------------------------------------------------------
-------------------------------------------------------------------------------

-- Client code should never call this constructor
-- For internal use only.  Does not install on relation...
function L.LField.New(rel, name, typ)
    local field   = setmetatable({}, L.LField)
    field.data    = nil
    field.type    = typ
    field.name    = name
    field.owner   = rel

    return field
end

function L.LField:Name()
    return self.name
end
function L.LField:FullName()
    return self.owner._name .. '.' .. self.name
end
function L.LField:Size()
    return self.owner._size
end
function L.LField:Type()
    return self.type
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
        typ = L.row(typ)
    end
    if not T.isLisztType(typ) or not typ:isFieldType() then
        error("NewField() expects a Liszt type or "..
              "relation as the 2nd argument", 2)
    end

    local field = L.LField.New(self, name, typ)
    rawset(self, name, field)
    self._fields:insert(field)
    return field
end

-- TODO: Hide this function so it's not public
function L.LField:Allocate()
    if self.data then self:ClearData() end
    self.data = MallocArray(self.type:terraType(),self:Size())
end

-- TODO: Hide this function so it's not public
-- remove allocated data and clear any depedent data, such as indices
function L.LField:ClearData ()
    if self.data then
        C.free(self.data)
        self.data = nil
    end
    -- clear grouping data if set on this field
    if self.owner._grouping and
       self.owner._grouping.key_field == self
    then
        self.owner._grouping.index:Release()
        self.owner._grouping = nil
    end
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

    local tmp = f1.data
    f1.data = f2.data
    f2.data = tmp
end

function L.LRelation:Copy( p )
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

    if not from.data then
        error('Cannot Copy() from field with no data', 2) end
    if not to.data then
        to:Allocate()
    end

    local copy_size = self:Size() * terralib.sizeof(from.type:terraType())
    C.memcpy(to.data, from.data, copy_size)
end


-- convert lua tables or LVectors to
-- Terra structs used to represent vectors
local function convert_vec(vec_val, typ)
    if L.is_vector(vec_val) then
        -- re-route to the common handler for Lua tables...
        return convert_vec(vec_val.data, typ)
    elseif type(vec_val) == 'table' and #vec_val == typ.N then
        return terralib.new(typ:terraType(), {vec_val})
    else
        return nil
    end
end

function L.LField:LoadFunction(lua_callback)
    self:Allocate()
    if self.type:isVector() then
        for i = 0, self:Size() - 1 do
            local val = lua_callback(i)
            self.data[i] = convert_vec(val, self.type)
        end
    else
        for i = 0, self:Size() - 1 do
            self.data[i] = lua_callback(i)
        end
    end
end

function L.LField:LoadList(tbl)
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

-- TODO: Hide this function so it's not public  (maybe not?)
function L.LField:LoadFromMemory(mem)
    self:Allocate()
    local copy_size = self:Size() * terralib.sizeof(self.type:terraType())
    C.memcpy(self.data, mem, copy_size)
end

function L.LField:LoadConstant(constant)
    self:Allocate()
    if self.type:isVector() then
        constant = convert_vec(constant, self.type)
    end

    for i = 0, self:Size() - 1 do
        self.data[i] = constant
    end
end

-- generic dispatch function for loads
function L.LField:Load(arg)
    -- load from lua callback
    if      type(arg) == 'function' then
        return self:LoadFunction(arg)
    elseif  type(arg) == 'cdata' then
        local typ = terralib.typeof(arg)
        if typ and typ:ispointer() then
            return self:LoadFromMemory(arg)
        end
    elseif  type(arg) == 'string' or PN.is_pathname(arg) then
        return self:LoadFromFile(arg)
    elseif  type(arg) == 'table' and not L.is_vector(arg) then
        if self.type:isVector() and #arg == self.type.N then
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
local function terraval_to_lua(val, typ)
    if typ:isVector() then
        local vec = {}
        for i = 1, typ.N do
            vec[i] = terraval_to_lua(val.d[i-1], typ:baseType())
        end
        return vec
    elseif typ:isNumeric() then
        return tonumber(val)
    elseif typ:isLogical() then
        if tonumber(val) == 0 then return false else return true end
    else
        error('unhandled terra_to_lua conversion')
    end
end

function L.LField:DumpToList()
    local arr = {}
    for i = 0, self:Size()-1 do
        arr[i+1] = terraval_to_lua(self.data[i], self.type)
    end
    return arr
end

-- callback(i, val)
--      i:      which row we're outputting (starting at 0)
--      val:    the value of this field for the ith row
function L.LField:DumpFunction(lua_callback)
    for i = 0, self:Size()-1 do
        local val = terraval_to_lua(self.data[i], self.type)
        lua_callback(i, val)
    end
end

function L.LField:print()
    print(self.name..": <" .. tostring(self.type:terraType()) .. '>')
    if not self.data then
        print("...not initialized")
        return
    end

    local N = self.owner._size
    if (self.type:isVector()) then
        for i = 0, N-1 do
            local s = ''
            for j = 0, self.type.N-1 do
                local t = tostring(self.data[i].d[j]):gsub('ULL','')
                s = s .. t .. ' '
            end
            print("", i, s)
        end
    else
        for i = 0, N-1 do
            local t = tostring(self.data[i]):gsub('ULL', '')
            print("", i, t)
        end
    end
end




-------------------------------------------------------------------------------
-------------------------------------------------------------------------------
--[[ Data Sharing Hooks                                                    ]]--
-------------------------------------------------------------------------------
-------------------------------------------------------------------------------


function L.LField:getDLD()
    if not self.type:isPrimitive() and not self.type:isVector() then
        error('Can only return DLDs for primitives and vectors, '..
              'not Row types or other types given to fields')
    end

    local terra_type = self.type:terraType()
    local n = nil
    if self.type:isVector() then
        terra_type = self.type:terraBaseType()
        n = self.type.N
    end
    local dld = DLD.new({
        type            = terra_type,
        type_n          = n,
        logical_size    = self.owner:Size(),
        data            = self.data,
        compact         = true,
    })

    return dld
end










