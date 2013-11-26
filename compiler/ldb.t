
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

local PN = terralib.require "compiler.pathname"

local JSON = require('compiler.JSON')

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


-- construct a Lua String from a Terra &int8 given the known length
local function tstr_to_lstr(tstr, len)
    local lstr = ''
    for i=1,len do
        lstr = lstr..string.char(tstr[i-1])
    end
    return lstr
end

--vector(double,4) requires 32-byte alignment
--WARNING: this will need more bookkeeping since you cannot call
-- free on the returned pointer
local terra allocateAligned32(size : uint64)
    var raw                 = [uint64](C.malloc(size + 32))
    -- round r up to the next multiple of 32,
    -- making sure there is at least one hidden preceding byte
    var aligned = (raw + 32) and not 31
    var diff : uint8        = aligned - raw
    -- hide the offset before the array
    @([&uint8](aligned-1))  = diff
    -- return the aligned pointer
    return [&opaque](aligned)
end
-- WARNING: THIS IS DEFINITELY UNTESTED
local terra freeAligned32(ptr : &opaque)
    -- extract the hidden offset
    var addr         = [uint64](ptr)
    var diff : uint8 = @([&uint8](addr-1))
    -- and reconstruct the allocated pointer to free
    var raw_ptr      = [&opaque](addr - diff)
    C.free(raw_ptr)
end
local function MallocArray(T,N)
    return terralib.cast(&T,allocateAligned32( N * terralib.sizeof(T) ))
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

function L.LRelation:Size()
    return self._size
end
function L.LRelation:Name()
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

function L.LRelation:CreateIndex(name)
    local f = self[name]
    if self._index then
        error("CreateIndex(): Relation already has an index")
    end
    if not L.is_field(f) then
        error("CreateIndex(): No field "..name)
    end
    if not f.type:isRow() then
        error("CreateIndex(): index field must refer to a relation")
    end
    local rel = f.type.relation
    rawset(self,"_index",f)
    local numindices = rel:Size()
    local numvalues = f:Size()
    --rawset(self,"_indexdata",MallocArray(uint64,numindices+1))
    rawset(self,"_indexdata",
           terralib.cast(&uint64,
                C.malloc((numindices+1) * terralib.sizeof(uint64))))
    local prev,pos = 0,0
    for i = 0, numindices - 1 do
        self._indexdata[i] = pos
        while f.data[pos] == i and pos < numvalues do
            if f.data[pos] < prev then
                -- TODO: NEED TO FREE ALLOCATION SAFELY IN THIS CASE
                error("CreateIndex(): Index field is not sorted")
            end
            prev,pos = f.data[pos],pos + 1
        end
    end
    assert(pos == numvalues)
    self._indexdata[numindices] = pos
end

function L.LRelation:json_serialize(rel_to_name, filedir)
    if filedir then
        os.execute("mkdir " .. filedir)
    end
    local json = {
        size      = self._size,
        fields    = {},
    }
    -- serialize fields
    for i,f in ipairs(self._fields) do
        local name = f.name
        local filename
        if filedir then
            filename = filedir..'/'..name..'.field'
        end
        json.fields[name] = f:json_serialize(rel_to_name, filename)
    end
    return json
end

function L.LField:json_serialize(rel_to_name, filename)
    if filename then
        self:SaveToFile(filename)
    end
    local json = {
        type = self.type:json_serialize(rel_to_name),
        path = filename,
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
-- the next set of calls are made on the actual relation object
function L.LRelation:json_deserialize_field(fname, json_tbl, name_to_rel)
    if not json_tbl.type then
        error('could not find field type', 2)
    end

    local typ = T.Type.json_deserialize(json_tbl.type, name_to_rel)
    local field = setmetatable({
        type  = typ,
        owner = self, -- the relation
        name  = fname,
    }, L.LField)

    -- install the field
    rawset(self, field.name, field)
    self._fields:insert(field)

    if json_tbl.path then
        field:LoadFromFile(json_tbl.path)
    else
        -- TODO: throw warning about uninitialized data here?
    end
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

function L.LField:Allocate()
    --self.data = MallocArray(self.type:terraType(),self:Size())
    local ttype = self.type:terraType()
    local typesize = terralib.sizeof(ttype)
    self.data = terralib.cast(&ttype, C.malloc(self:Size() * typesize))
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
    local copy_size = self:Size() * terralib.sizeof(self.type:terraType())
    C.memcpy(self.data, mem, copy_size)
end

-- remove allocated data and clear any depedent data, such as indices
function L.LField:ClearData ()
    if self.data then
        C.free(self.data)
        self.data = nil
    end
    -- clear index if installed on this field
    if self.owner._index == self then
        C.free(self.owner._indexdata)
        self.owner._indexdata = nil
        self.owner._index = nil
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


-- A Field file is laid out as follows
--------------------------------------------------------------------
--                          -- HEADER --                          --
-- MAGIC SIGNATURE (8 bytes; uint64) (LITFIELD) (endianness check)
-- VERSION NUMBERS (8 bytes; uint64)
-- FILE SIZE in BYTES (8 bytes; uint64)
-- HEADER SIZE in BYTES (8 bytes; uint64)
-- ARRAY SIZE in #entries (8 bytes; uint64)
-- TYPE_STR LENGTH in BYTES (8 bytes; uint64)
-- HINT_STR LENGTH in BYTES (8 bytes; uint64)
--------------------------------------------------------------------
--                         -- TYPE_STR --   (still header)        --
--------------------------------------------------------------------
--                         -- HINT_STR --   (still header)        --
--------------------------------------------------------------------
--  optional dead space for data alignment  (still header)        --
--================================================================--
--                        -- DATA BLOCK --   (not header)         --
--                              ...                               --
--------------------------------------------------------------------
local struct FieldHeader {
    version      : uint64;
    file_size    : uint64;
    header_size  : uint64;
    array_size   : uint64;
    type_str_len : uint64;
    hint_str_len : uint64;
    type_str     : &int8;
    hint_str     : &int8;
}
local function new_field_header()
    local header =
        terralib.cast(&FieldHeader, C.malloc(terralib.sizeof(FieldHeader)))
    header.type_str = nil
    header.hint_str = nil
    return header
end
local terra write_field_header(
    file       : &C.FILE,
    header     : &FieldHeader
)
    var LITFIELD    : uint64 = 0x4c49544649454c44ULL
    var data_size   : uint64 = header.file_size - header.header_size
    var dead_space  : uint64 = header.header_size - 7*8 -
                               header.type_str_len - header.hint_str_len

    -- header metadata
    C.fwrite( &LITFIELD,                8, 1, file )
    C.fwrite( &header.version,          8, 1, file )
    C.fwrite( &header.file_size,        8, 1, file )
    C.fwrite( &header.header_size,      8, 1, file )
    C.fwrite( &header.array_size,       8, 1, file )
    C.fwrite( &header.type_str_len,     8, 1, file )
    C.fwrite( &header.hint_str_len,     8, 1, file )
    -- header strings
    C.fwrite( header.type_str,          1, header.type_str_len, file )
    C.fwrite( header.hint_str,          1, header.hint_str_len, file )
    -- jump to end of header dead space
    C.fseek ( file, header.header_size, C.SEEK_SET_value() )
end

-- allocates arrays for string
local terra read_field_header(
    file       : &C.FILE,
    header     : &FieldHeader
)
    -- Check the Magic Number
    var LITFIELD        : uint64 = 0x4c49544649454c44ULL
    var magic_number    : uint64
    C.fread( &magic_number,             8, 1, file )
    if magic_number ~= LITFIELD then
        return 1 -- TODO: better error message somehow?
    end

    -- Check the version
    C.fread( &header.version,           8, 1, file )
    if header.version ~= 0x00 then
        return 1 -- TODO: better error
    end

    -- read in the rest of the metadata
    C.fread( &header.file_size,         8, 1, file )
    C.fread( &header.header_size,       8, 1, file )
    C.fread( &header.array_size,        8, 1, file )
    C.fread( &header.type_str_len,      8, 1, file )
    C.fread( &header.hint_str_len,      8, 1, file )

    -- allocate space for strings
    header.type_str = [&int8](C.malloc( header.type_str_len ))
    header.hint_str = [&int8](C.malloc( header.hint_str_len ))

    -- read in strings
    C.fread( header.type_str,           1, header.type_str_len, file )
    C.fread( header.hint_str,           1, header.hint_str_len, file )

    -- jump to end of header dead space
    C.fseek( file, header.header_size,  C.SEEK_SET_value() )
    
    return 0
end

function L.LField:SaveToFile(filename)
    assert(self.data)

    -- open the file for writing
    local file = C.fopen(filename, 'wb')
    if not file then
        error("failed to open file "..filename.." to write to.", 2)
    end

    -- version is currently 0.0 (dev)
    local major_version = 0x00
    local minor_version = 0x00
    local version =
        bit.lshift(bit.bor(bit.lshift(major_version, 8), minor_version), 16)

    -- compute some useful values...
    local n_rows    = self.owner._size
    local tsize     = terralib.sizeof(self.type:terraType())
    local type_str  = self.type:toString()
    local hint_str  = "" -- for future use?

    -- structs / arrays to pack data for Terra functions
    local header = new_field_header()
    -- pack the header metadata
    header.version          = version
    header.array_size       = n_rows
    header.type_str_len     = #type_str + 1
    header.hint_str_len     = #hint_str + 1
    header.header_size      = 7 * 8 +
                              header.type_str_len +
                              header.hint_str_len
    header.file_size        = header.header_size +
                              header.array_size * tsize
    -- pack the header strings
    -- TODO: Is this leaking memory?
    --       Does the Terra strings' backing memory need to be freed?
    (terra()
        header.type_str = type_str
        header.hint_str = hint_str
    end)()

    -- Error escape which will make sure to close the file on the way out
    local function err(msg)
        C.free(header)
        C.fclose(file)
        error(msg, 3)
    end

    write_field_header( file, header )
    if C.ferror(file) ~= 0 then
        C.perror('field file header write error: ')
        err("error writing field file header to "..filename)
    end

    -- write data block out
    C.fwrite( self.data, tsize, n_rows, file )
    if C.ferror(file) ~= 0 then
        C.perror('field file data write error: ')
        err("error writing field file data to "..filename)
    end

    C.free(header)
    C.fclose(file)
end


local function check_field_type(field, type_str)
    -- right now, we let any row match any other row.
    -- TODO: add a warning message system to alert the user
    --       to potential errors when loading a row field
    --       with inconsistent types.
    -- Also note that this is subject to re-design
    if field.type:isRow() then
        return type_str:sub(1,3) == 'Row'
    else
        return type_str == field.type:toString()
    end
end
function L.LField:LoadFromFile(filename)
    self:ClearData()

    -- open the file for reading
    local file = C.fopen(filename, 'rb')
    if not file then
        error("failed to open file "..filename.." to read from.", 2)
    end
    -- allocate space for header data
    local header = new_field_header()

    -- early exit error helper
    local function err(msg)
        C.free(header.type_str)
        C.free(header.hint_str)
        C.free(header)
        C.fclose(file)
        error('field file '..filename..' load error: '..msg, 3)
    end

    -- extract the file header
    read_field_header( file, header )
    if C.ferror(file) ~= 0 then
        C.perror('field file header read error: ')
        err('error reading header')
    end

    -- version check
    if header.version ~= 0 then err('version must be 0.0') end
    -- metadata extraction
    local type_str =
        tstr_to_lstr(header.type_str, tonumber(header.type_str_len) - 1)
    local hint_str =
        tstr_to_lstr(header.hint_str, tonumber(header.hint_str_len) - 1)
    local array_size = tonumber(header.array_size)
    -- metadata consistency check
    if not check_field_type(self, type_str) then
        err('type mismatch, expected '..self.type:toString()..
            ' but got '..type_str)
    end
    if self:Size() ~= array_size then
        err('size mismatch, expected '..self:Size()..' elements, but '..
            'the file contained '..array_size..' elements')
    end

    -- read data block out
    self:Allocate()
    local type_size = terralib.sizeof(self.type:terraType())
    C.fread( self.data, type_size, array_size, file )
    if C.ferror(file) ~= 0 then
        C.perror('field file data read error: ')
        C.free(self.data)
        err('error reading data')
    end

    -- cleanup intermediary allocations and close the file
    C.free(header.type_str)
    C.free(header.hint_str)
    C.free(header)
    C.fclose(file)
end


function L.SaveRelationIndex(params)
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
        local filedir = './blah/'..name
        json_obj.relations[name] =
            rel:json_serialize(rel_to_name, filedir)
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

    -- dump text to the appropriate file.
    local f, err_msg, err_num = io.open(filename, 'w')
    if not f then
        io.stderr:write(err_msg..'\n')
        return err_msg, err_num
    else
        f:write(json_str)
        f:close()
    end
end

function L.LoadRelationIndex(params)
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
              'The file '..filename..' has version '..
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
        local json_relation = json_obj.relations[id]

        -- deserialize each field
        for f_name, f_json in pairs(json_relation.fields) do
            relation:json_deserialize_field(f_name, f_json, relations)
        end
    end

    --for k,v in pairs(relations) do
    --    print(k)
    --    relations[k]:print()
    --end

    return relations, nil

    --print(json_str)
end












