
-- LDB = Liszt DataBase

-- The "database" is responsible for tracking all relations
-- currently present in the runtime.  This helps us view
-- relations from a closed rather than open-world perspective,
-- so it's easy to ask "Does a relation named XXX exist?" or
-- Are all row types currently in the relation tables valid?

-- If we eventually implement some form of garbage collector or
-- other auto memory manager, this database will give the needed
-- total view of memory for those subsystems.

-- file/module namespace table
local LDB = {}

local T = terralib.require('compiler/types')

local DECL = terralib.require('include/decl')
local LRelation = DECL.LRelation
local LField    = DECL.LField
--local LScalar   = DECL.LScalar
--local 
local LMacro    = DECL.LMacro

local C = terralib.includecstring [[
    #include <stdlib.h>
    #include <string.h>
    #include <stdio.h>
    #include <math.h>
]]


-- terra type of a field that represents a row in another relation
local ROW_TERRA_TYPE       = T.t.uint:terraType()
-- terra type of an orientation field
local ORIENT_TYPE = T.t.uint8



local relation_database = {}

-------------------------------------------------------------------------------
--[[ LRelation methods                                                     ]]--
-------------------------------------------------------------------------------

local valid_relation_name_err_msg =
    "Relation names must be valid Lua Identifiers: a letter or underscore,"..
    " followed by zero or more underscores, letters, and/or numbers"
local valid_field_name_err_msg =
    "Field names must be valid Lua Identifiers: a letter or underscore,"..
    " followed by zero or more underscores, letters, and/or numbers"
local function is_valid_lua_identifier(name)
    if type(name) ~= 'string' then return false end

    -- regex for valid LUA identifiers
    if not name:match('^[_%a][_%w]*$') then return false end

    return true
end

function LDB.NewRelation(size, name)
    -- error check
    if not name or type(name) ~= "string" then
        error("NewRelation() expects a string as the 2nd argument", 2)
    end
    if not is_valid_lua_identifier(name) then
        error(valid_relation_name_err_msg, 2)
    end
    if relation_database[name] then
        error("Cannot create duplicate relation with name "..name, 2)
    end

    -- construct and return the relation
    local rel = setmetatable( {
        _size      = size,
        _fields    = terralib.newlist(),
        _macros    = terralib.newlist(),
        _name      = name,
    },
    LRelation)
    relation_database[name] = rel
    return rel
end

function LRelation:size()
    return self._size
end
function LRelation:name()
    return self._name
end

-- prevent user from modifying the lua table
function LRelation:__newindex(fieldname,value)
    error("Cannot assign members to LRelation object "..
          "(did you mean to call self:NewField?)", 2)
end

function LRelation:NewField (name, typ)  
    if not name or type(name) ~= "string" then
        error("NewField() expects a string as the first argument", 2)
    end
    if not is_valid_lua_identifier(name) then
        error(valid_fieldName_err_msg, 2)
    end
    if self[name] then
        error("Cannot create a new field with name '"..name.."'  "..
              "That name is already being used.", 2)
    end

    if not T.Type.isLisztType(typ) or
       not (typ:isExpressionType() or typ:isRow())
    then
        error("NewField() expects a Liszt type as the 2nd argument", 2)
    end

    local field = setmetatable({}, LField)
    field.type  = typ
    field.name  = name
    field.owner = self
    rawset(self, name, field)
    self._fields:insert(field)
    return field
end

function LRelation:NewFieldMacro (name, macro)  
    if not name or type(name) ~= "string" then
        error("NewFieldMacro() expects a string as the first argument", 2)
    end
    if not is_valid_lua_identifier(name) then
        error(valid_fieldName_err_msg, 2)
    end
    if self[name] then
        error("Cannot create a new field-macro with name '"..name.."'  "..
              "That name is already being used.", 2)
    end

    if not DECL.is_macro(macro) then
        error("NewFieldMacro() expects a Macro as the 2nd argument", 2)
    end

    rawset(self, name, macro)
    self._macros:insert(macro)
    return macro
end

function LRelation:LoadIndexFromMemory(name, row_idx)
    assert(self._index == nil)
    local f = self[name]
    assert(f ~= nil)
    assert(DECL.is_field(f))
    assert(f.data == nil)
    assert(f.type)
    assert(f.type:isRow())
    --assert(f.relation ~= nil) -- index field must be a relational type

    local target_relation = f.type.relation
    local ttype  = ROW_TERRA_TYPE
    local tsize  = terralib.sizeof(ttype)
    -- Why is this +1?
    local nbytes = (target_relation._size + 1) * tsize

    local mem = terralib.cast(&ttype,C.malloc(nbytes))
    rawset(self, "_index", mem)
    local memT = terralib.typeof(row_idx)
    assert(memT == &ttype)

    C.memcpy(self._index,row_idx,nbytes)
    f.data = terralib.cast(&ttype,C.malloc(self._size*tsize))
    for i = 0, f.type.relation._size - 1 do
        local b = self._index[i]
        local e = self._index[i+1]
        for j = b, e - 1 do
            f.data[j] = i
        end
    end
end


function LRelation:json_serialize()
    local json = {
        name      = self._name,
        size      = self._size,
        fields    = {},
    }
    -- serialize fields
    for i,f in ipairs(self._fields) do
        if DECL.is_field(f) then
            json.fields[i] = f:json_serialize()
        end
    end
    return json
end

-- we split de-serialization into two phases so that all of the
-- Relations can be reconstructed before any of the Fields.
-- This ensures that Row Fields will safely match some existing
-- Relation when deserialized.
function LRelation.json_deserialize_rel(json_tbl)
    local relation = LDB.NewRelation(json_tbl.size, json_tbl.name)
    return relation
end
-- the second call should supply the actual relation object...
function LRelation:json_deserialize_fields(json_tbl)
    for i,json_field in ipairs(json_tbl.fields) do
        local f = LField.json_deserialize(json_field)
        f.owner = self
        rawset(self, f.name, f)
        self._fields:insert(f)
    end
end


--function LRelation:dump()
--    print(self._name, "size: ".. tostring(self._size))
--    for i,f in ipairs(self._fields) do
--        f:dump()
--    end
--end

-------------------------------------------------------------------------------
--[[ LField methods:                                                       ]]--
-------------------------------------------------------------------------------

local terra copy_bytes (dest : &uint8, src : &uint8, length : uint, size : uint, stride : uint, offset : uint)
    src = src + offset

    -- dont potentially copy past the length of the source array:
    var copy_len : int
    if stride < size then copy_len = stride else copy_len = size end

    for i = 0, length do
        C.memcpy(dest,src,copy_len)
        src  = src  + stride
        dest = dest + size
    end
end

-- specify stride, offset in bytes, default stride reads memory contiguously and default offset reads from mem ptr directly
function LField:LoadFromMemory (mem, stride, offset)
    local ttype = self.type:terraType()
    local tsize = terralib.sizeof(ttype)

    -- Terra vectors are sized at a power of two bytes, whereas arrays are just N * sizeof(basetype)
    -- so, if we are storing arrays as terra vectors, tsize is >= the size of the data for one field
    -- entry in a contiguous array.
    local dsize = ttype:isvector() and terralib.sizeof(ttype.type[ttype.N]) or tsize

    if not stride then stride = dsize end
    if not offset then offset = 0     end

    assert(stride >= dsize)
    assert(self.data == nil)

    local nbytes = self.owner._size * tsize
    local bytes  = C.malloc(nbytes)
    self.data    = terralib.cast(&ttype,bytes)

    -- cast mem to void* to avoid automatic conversion issues for pointers to non-primitive types
    mem = terralib.cast(&opaque, mem)

    -- If the array is laid out contiguously in memory, just do a memcpy
    -- otherwise, read with a stride
    if (stride == tsize) then
        C.memcpy(self.data,mem,nbytes)
    else
        copy_bytes(bytes,mem,self.owner._size, tsize, stride, offset)
    end
end

function LField:LoadFromCallback (callback)
    local bytes  = C.malloc(self.owner._size * terralib.sizeof(self.type:terraType()))
    self.data    = terralib.cast(&self.type:terraType(),bytes)

    for i = 0, self.owner._size-1 do
        callback(self.data + i, i)
    end
end    

function LField:print()
    print(self.name..":")
    if not self.data then
        print("...not initialized")
        return
    end

    local N = self.owner._size
    if (self.type:isVector()) then
        for i = 0, N-1 do
            local s = ''
            for j = 0, self.type.N-1 do
                s = s .. tostring(self.data[i][j]) .. ' '
            end
            print("", i, s)
        end
    else
        for i = 0, N-1 do
            print("",i, self.data[i])
        end
    end
end















return LDB
