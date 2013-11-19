
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
package.loaded["compiler.ldb"] = LDB

local L = terralib.require "compiler.lisztlib"
local T = terralib.require "compiler.types"
local C = terralib.require "compiler.c"


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

    -- construct and return the relation
    local rel = setmetatable( {
        _size      = size,
        _fields    = terralib.newlist(),
        _macros    = terralib.newlist(),
        _name      = name,
    },
    L.LRelation)
    return rel
end

function L.LRelation:size()
    return self._size
end
function L.LRelation:name()
    return self._name
end

-- prevent user from modifying the lua table
function L.LRelation:__newindex(fieldname,value)
    error("Cannot assign members to LRelation object "..
          "(did you mean to call self:NewField?)", 2)
end

function L.LRelation:NewField (name, typ)  
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

    if not (T.isLisztType(typ) and typ:isValueType()) and
       not L.is_relation(typ)
    then
        error("NewField() expects a Liszt type as the 2nd argument", 2)
    end

    if L.is_relation(typ) then
        typ = L.row(typ)
    end

    local field = setmetatable({}, L.LField)
    field.type  = typ
    field.name  = name
    field.owner = self
    rawset(self, name, field)
    self._fields:insert(field)
    return field
end

function L.LRelation:NewFieldMacro (name, macro)  
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

    if not L.is_macro(macro) then
        error("NewFieldMacro() expects a Macro as the 2nd argument", 2)
    end

    rawset(self, name, macro)
    self._macros:insert(macro)
    return macro
end

function L.LRelation:json_serialize(rel_to_name)
    local json = {
        size      = self._size,
        fields    = {},
    }
    -- serialize fields
    for i,f in ipairs(self._fields) do
        local name = f.name
        json.fields[name] = f:json_serialize(rel_to_name)
    end
    return json
end

function L.LField:json_serialize(rel_to_name)
    local json = {
        type = self.type:json_serialize(rel_to_name)
    }
    return json
end

-- we split de-serialization into two phases so that all of the
-- Relations can be reconstructed before any of the Fields.
-- This ensures that Row Fields will safely match some existing
-- Relation when deserialized.
function L.LRelation.json_deserialize_stub(json_tbl, relation_name)
    if not type(json_tbl.size) == 'number' then
        error('tried to deserialize relation missing size', 2)
    end
    relation_name = relation_name or nil

    local relation = LDB.NewRelation(json_tbl.size, relation_name)
    return relation
end
-- the second call should supply the actual relation object...
function L.LRelation:json_deserialize_fields(json_tbl, name_to_rel)
    for field_name, field_json in pairs(json_tbl.fields) do

        local field = L.LField.json_deserialize(field_json, name_to_rel)

        field.owner = self -- i.e. the relation
        field.name  = field_name
        rawset(self, field.name, field)
        self._fields:insert(field)
    end
end

function L.LField.json_deserialize(json_tbl, name_to_rel)
    if not json_tbl.type then
        error('could not find field type', 2)
    end

    local typ = T.Type.json_deserialize(json_tbl.type, name_to_rel)
    local field = setmetatable({
        type = typ
    }, L.LField)
    return field
end



function L.LRelation:print()
    print(self._name, "size: ".. tostring(self._size))
    for i,f in ipairs(self._fields) do
        f:print()
    end
end

-------------------------------------------------------------------------------
--[[ LField methods:                                                       ]]--
-------------------------------------------------------------------------------


function L.LField:Size()
    return self.owner._size
end
local bit = require "bit"


--vector(double,4) requires 32-byte alignment
--WARNING: this will need more bookkeeping since you cannot call
-- free on the returned pointer
local terra allocateAligned32(size : uint64)
    var r = [uint64](C.malloc(size + 32))
    r = (r + 31) and not 31
    return [&opaque](r)
end
function L.LField:Allocate()
    local bytes  =
        allocateAligned32(self:Size() * terralib.sizeof(self.type:terraType()))
    self.data    = terralib.cast(&self.type:terraType(),bytes)
end

function L.LField:LoadFromCallback (callback)
    -- TODO: It would be nice to typecheck the callback's type signature...
    self:Allocate()
    for i = 0, self:Size() - 1 do
        callback(self.data + i, i)
    end
end

function L.LField:LoadFromMemory(mem)
    self:Allocate()
    C.memcpy(self.data,mem, self:Size() * terralib.sizeof(self.type:terraType()))
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


function LDB.SaveRelationIndex(params)
local interface_description =
[[
    SaveRelationIndex assumes that it will be passed named arguments
    Arguments are as follows:

    relations    = {...}    -- a table with { name = relation } pairs
    filename     = "..."    -- file to save to on disk
                            -- if string does not end in .json,
                            -- then .json will be appended
    notes        = "..."    -- optionally a string with some notes can
                            -- be saved out so the index's purpose is less
                            -- inscrutable later.
    compressed   = bool     -- if 'compressed' is present and set to true,
                            -- then the JSON will not be pretty-printed.
                            -- this will save on filesize slightly
                            -- at the expense of being less human-readable
]]
    if not (params.relations and type(params.relations) == 'table') or
       not (params.filename  and type(params.filename)  == 'string')
    then
        error(interface_description, 2)
    end
    local relations   = params.relations
    local filename    = params.filename

    local rel_to_name = {}
    for k,v in pairs(relations) do rel_to_name[v] = k end

    local notes = params.notes
    if notes and type(notes) ~= 'string' then notes = nil end

    -- construct JSON object to encode
    local json_obj = {}
    if notes then json_obj.notes = notes end
    json_obj.relations = {}
    for name, rel in pairs(relations) do
        json_obj.relations[name] = rel:json_serialize(rel_to_name)
    end
    -- version numbers here are for the file format
    -- so that we can detect old data moving forward
    json_obj.major_version   = 0
    json_obj.minor_version   = 0

    -- perform JSON encoding
    local json_str, err_msg
    if params.compressed then
        json_str, err_msg = JSON.stringify(json_obj)
    else
        json_str, err_msg = JSON.stringify(json_obj, '  ')
    end
    if err_msg then
        io.stderr:write(err_msg..'\n')
        return err_msg
    end

    -- dump text to the appropriate file.  For now, dump it to console
    local f, err_msg, err_num = io.open(filename, 'w')
    if not f then
        io.stderr:write(err_msg..'\n')
        return err_msg, err_num
    else
        f:write(json_str)
        f:close()
    end
end

function LDB.LoadRelationIndex(params)
local interface_description = [[
    LoadRelationIndex assumes that it will be passed named arguments
    Arguments are as follows:

    filename    = "..."     -- where to look for the index.json file on disk
]]
    if not (params.filename and type(params.filename)  == 'string')
    then
        error(interface_description, 2)
    end
    local filename    = params.filename

    -- get the json string sucked in from the disk on file
    local f, err_msg, err_num = io.open(filename, 'r')
    if not f then
        io.stderr:write(err_msg..'\n')
        return {}, err_msg, err_num
    end
    -- otherwise, let's grab this string
    local json_str = f:read('*all')
    f:close()

    -- parse the JSON string into an object
    local json_obj, err_msg = JSON.parse(json_str)
    if err_msg then
        io.stderr:write(err_msg..'\n')
        return {}, err_msg
    end

    -- check version #
    if json_obj.major_version ~= 0 or json_obj.minor_version ~= 0 then
        error('This Liszt relation loader only supports index.json files '..
              'of version 0.0 (development);    '..
              'The file that was just attempted to load is version '..
              json_obj.major_version..'.'..json_obj.minor_version..
              ' ')
    end

    local relations = {}

    -- unpack the JSON object into a reconstructed set of relations
    if not json_obj.relations or not type(json_obj.relations) == 'table' then
        return {}, 'JSON object does not have a relations table'
    end
    -- first, fill out stub relations
    for id, json_val in pairs(json_obj.relations) do
        if not is_valid_lua_identifier(id) then
            return {}, valid_relation_name_err_msg
        end

        relations[id] = L.LRelation.json_deserialize_stub(json_val, id)
    end
    -- then we make a second pass to actually recover the fields
    for id, relation in pairs(relations) do
        local json_val = json_obj.relations[id]

        relation:json_deserialize_fields(json_val, relations)
    end

    for k,v in pairs(relations) do
        print(k)
        relations[k]:print()
    end


    --print(json_str)
end












