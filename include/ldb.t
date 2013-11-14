
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
local LMacro    = DECL.LMacro

local C = DECL.C


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

    if not (T.Type.isLisztType(typ) and typ:isValueType()) and
       not DECL.is_relation(typ)
    then
        error("NewField() expects a Liszt type as the 2nd argument", 2)
    end

    if DECL.is_relation(typ) then
        typ = T.t.row(typ)
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

function LRelation:print()
    print(self._name, "size: ".. tostring(self._size))
    for i,f in ipairs(self._fields) do
        f:print()
    end
end

-------------------------------------------------------------------------------
--[[ LField methods:                                                       ]]--
-------------------------------------------------------------------------------

function LField:LoadFromCallback (callback)
    -- TODO: It would be nice to typecheck the callback's type signature...
    local bytes  =
        C.malloc(self.owner._size * terralib.sizeof(self.type:terraType()))
    self.data    = terralib.cast(&self.type:terraType(),bytes)

    for i = 0, self.owner._size-1 do
        callback(self.data + i, i)
    end
end    

function LField:print()
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
                local t = tostring(self.data[i][j]):gsub('ULL','')
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















return LDB
