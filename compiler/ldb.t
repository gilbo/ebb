
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
local Pathname = PN.Pathname
local ffi = require('ffi')

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


-------------------------------------------------------------------------------
--[[ LRelation methods                                                     ]]--
-------------------------------------------------------------------------------

function L.NewRelation(size, name)
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
    
    local function is_value_or_row_type()
        return T.isLisztType(typ) and (typ:isValueType() or typ:isRow())
    end
    if not L.is_relation(typ) and not is_value_or_row_type() then
        error("NewField() expects a Liszt type or "..
              "relation as the 2nd argument", 2)
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
    rawset(self,"_indexdata",MallocArray(uint64,numindices+1))
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

-- TODO: Hide this function so it's not public
function L.LField:Allocate()
    self.data = MallocArray(self.type:terraType(),self:Size())
end

-- TODO: Hide this function so it's not public
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

function L.LField:LoadFromCallback (callback)
    -- TODO: It would be nice to typecheck the callback's type signature...
    self:Allocate()
    for i = 0, self:Size() - 1 do
        callback(terralib.cast(&self.type:terraBaseType(), self.data + i), i)
    end
end

-- TODO: Hide this function so it's not public
function L.LField:LoadFromMemory(mem)
    self:Allocate()
    local copy_size = self:Size() * terralib.sizeof(self.type:terraType())
    C.memcpy(self.data, mem, copy_size)
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
                local t = tostring(self.data[i]._0[j]):gsub('ULL','')
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
--[[ Serialization / Deserialization                                       ]]--
-------------------------------------------------------------------------------
-------------------------------------------------------------------------------

local function is_path_or_str(obj)
    return type(obj) == 'string' or PN.is_pathname(obj)
end



-------------------------------------------------------------------------------
--[[ Field saving / loading                                                ]]--
-------------------------------------------------------------------------------


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
    if file == nil then -- b/c of cdata, must check explicitly
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
    if not is_path_or_str(filename) then
        error('LoadFromFile() expects a string or pathname as argument', 2)
    end
    local path = Pathname.new(filename)

    self:ClearData()

    -- open the file for reading
    local file = C.fopen(tostring(path), 'rb')
    if file == nil then -- b/c of cdata, must check explicitly
        error("failed to open file "..tostring(path).." to read from.", 2)
    end
    -- allocate space for header data
    local header = new_field_header()

    -- early exit error helper
    local function err(msg)
        C.free(header.type_str)
        C.free(header.hint_str)
        C.free(header)
        C.fclose(file)
        error('field file '..tostring(path)..' load error: '..msg, 3)
    end

    -- extract the file header
    read_field_header( file, header )
    if C.ferror(file) ~= 0 then
        C.perror('field file header read error: ')
        err('error reading header')
    end

    -- version check
    if header.version ~= 0x0 then err('version must be 0.0') end
    -- metadata extraction
    local type_str      = ffi.string(header.type_str, header.type_str_len - 1)
    local hint_str      = ffi.string(header.hint_str, header.hint_str_len - 1)
    local array_size    = tonumber(header.array_size)
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





-------------------------------------------------------------------------------
--[[ Relation Schema Saving                                                ]]--
-------------------------------------------------------------------------------



local function save_opt_hint_str(opt_str)
    return 'Pass argument '..opt_str..' to SaveRelationSchema()\n'..
           '  to suppress this error.'
end

local function check_save_relations(relations, params)
    local rel_to_name       = params.rel_to_name
    local no_file_data      = params.no_file_data

    for rname, rel in pairs(relations) do
        if type(rname) ~= 'string' then
            error('SaveRelationSchema() Error: Found a non-string '..
                  'key in the relations table', 3)
        end
        local rstr = 'Relation "'..rname..'"'

        -- The relations must actually be relations
        if not L.is_relation(rel) then
            error('SaveRelationSchema() Error: '..rstr..
                  ' is not a relation', 3)
        end

        -- The relation names must be valid
        if not is_valid_lua_identifier(rname) then
            error('SaveRelationSchema() Error: '..rstr..
                  ' has an invalid name\n'..
                  valid_relation_name_err_msg, 3)
        end

        -- The relation names must be writable to disk
        -- (redundant; present in case of name policy changes)
        if not no_file_data then
            local status, err_msg = pcall(function() Pathname.new(rname) end)
            if not status then
                error('SaveRelationSchema() Error: '..rstr..
                      ' cannot be used as a directory name\n'..err_msg, 3)
            end
        end

        for _,f in ipairs(rel._fields) do
            local fname = f.name
            local fstr  = 'Field "'..fname..'" on '..rstr

            -- The field names must be valid
            if not is_valid_lua_identifier(fname)
            then
                error('SaveRelationSchema() Error: '..fstr..
                      ' has an invalid name\n'..
                      valid_field_name_err_msg, 3)
            end

            -- The field names must be writable to disk
            -- (redundant; present in case of name policy changes)
            if not no_file_data then
                local status, err_msg =
                    pcall(function() Pathname.new(fname) end)
                if not status then
                    error('SaveRelationSchema() Error: '..fstr..
                          ' cannot be used as a file name\n'..err_msg, 3)
                end
            end

            -- Row fields must reference Relations that are being saved
            if f.type:isRow() then
                local lookup = rel_to_name[f.type.relation]
                if not lookup then
                    error('SaveRelationSchema() Error: '..fstr..
                          ' has type '..f.type:toString()..
                          ' which references a Relation not being saved.', 3)
                end
            end

            -- Ensure that data is present
            if not no_file_data then
                if f.data == nil then -- in case of cdata pointer
                    error('SaveRelationSchema() Error: '..fstr..' is '..
                          'uninitialized; it has no data', 3)
                end
            end
        end
    end
end

local function json_serialize_relation(relation, params)
    local rel_to_name   = params.rel_to_name
    local basedir       = params.basedir
    local rdir          = params.rdir
    local no_file_data  = params.no_file_data

    local json = {
        size    = relation._size,
        fields  = {},
    }
    -- serialize fields
    for _, field in ipairs(relation._fields) do
        local fname = field.name
        local fstr  = 'Field "'..fname..'" on '..
                      'Relation "'..rel_to_name[relation]..'"'

        -- get the field type
        local typ
        local status, err_msg = pcall(function()
            typ = field.type:json_serialize(rel_to_name)
        end)
        if not status then
            error('SaveRelationSchema() Error: Bad Type!\n'..err_msg, 3)
        end

        -- build the field json
        json.fields[fname] = { type = typ }

        -- Save field data to disk and add pathname if appropriate
        if not no_file_data then
            local file  = rdir..Pathname.new(fname..'.field')

            local status, err_msg = pcall(function()
                field:SaveToFile(tostring(basedir..file))
            end)
            if not status then
                error('SaveRelationSchema() Error: Error while saving '..
                      'field data of '..fstr..'\n'..err_msg, 3)
            end

            json.fields[fname].path = file:tostring()
        end
    end

    -- attach an index flag if present
    if relation._index then
        json.index = relation._index.name
    end

    return json
end

function L.SaveRelationSchema(params)
local interface_description =
[[

    SaveRelationSchema assumes that it will be passed named arguments
    Arguments are as follows:

    relations       = {...} -- a table with { name = relation } pairs
    file            = "..." -- file or directory to save to on disk
                            -- if the string ends in .json, then
                            --   file is interpreted as the schema file
                            --   to write.
                            -- otherwise,
                            --   file is interpreted as a directory
                            --   in which to write a schema.json file.
                            --   The directory will be created if it
                            --   doesn't exist.
    notes           = "..." -- optionally a string with some notes can
                            -- be saved out so the schema's purpose is less
                            -- inscrutable later.
    compressed      = bool  -- if 'compressed' is present and set to true,
                            -- then the JSON will not be pretty-printed.
                            -- This will save on filesize slightly
                            -- at the expense of being less human-readable.
    no_file_data    = bool  -- If set to true, this option will save the
                            --  schema file without paths to default
                            --  field data, and will not try to save out
                            --  the default field data.
                            --  RECOMMENDED SETTING: false
                            --  This option is present to support building
                            --  more elaborate storage systems using
                            --  the basic one.
]]
    if type(params) ~= 'table' or
       type(params.relations) ~= 'table' or
       not is_path_or_str(params.file)
    then
        error(interface_description, 2)
    end
    local relations = params.relations
    local file      = Pathname.new(params.file)
    local filedir   = file:dirpath()
    local notes     = params.notes or ''
    local no_file_data    = params.no_file_data

    -- build inverse mapping relation -> relation_name
    local rel_to_name = {}
    for rname, rel in pairs(relations) do rel_to_name[rel] = rname end

    -- Check whether or not names are valid...
    check_save_relations(relations, {
        rel_to_name     = rel_to_name,
        no_file_data    = no_file_data,
    })


    -- Handle all filesystem / directory mangling up front
    local function check_filedir() -- utility function
        if not filedir:exists() then
            error('SaveRelationSchema() Error: Cannot save '..
                  '"'..tostring(file)..'" because directory '..
                  '"'..tostring(filedir)..'" does not exist.', 3)
        end
    end
    if no_file_data then
        if file:extname():lower() ~= 'json' then
            error('SaveRelationSchema() Error: If no_file_data is set, '..
                  'then the provided filename must name a .json file\n', 2)
        end
        check_filedir()
    else
        if file:extname():lower() ~= 'json' then
            filedir     = file
            file        = file .. 'schema.json'

            -- create the file directory if necessary
            if not filedir:exists() then
                if not filedir:mkdir() then
                    error('SaveRelationSchema() Error: Could not create '..
                          'file directory "'..tostring(filedir)..'"', 2)
                end
            end
        else
            check_filedir()
        end

        -- create the required sub-directory structure
        for rname, rel in pairs(relations) do
            local rdir = filedir..rname
            if not rdir:exists() and not rdir:mkdir() then
                error('SaveRelationSchema() Error: Could not create '..
                      'sub-directory "'..tostring(rdir)..'"', 2)
            end
        end
    end


    -- construct JSON object to encode
    local json = {
        major_version = 0,
        minor_version = 0,
        notes = notes,
        relations = {},
    }
    -- serialize the relations
    for rname, rel in pairs(relations) do
        -- want path relative to schema.json file
        local rdir = Pathname.new(rname)
        json.relations[rname] = json_serialize_relation(rel, {
            rel_to_name     = rel_to_name,
            basedir         = filedir,
            rdir            = rdir,
            no_file_data    = no_file_data,
        })
    end


    -- perform JSON encoding
    local space
    if not params.compressed then space = '  ' end
    local json_str, err_msg = JSON.stringify(json, space)
    if err_msg then
        error('SaveRelationSchema() Error: JSON encode failed\n'..
              err_msg, 2)
    end

    -- dump text to the appropriate JSON file.
    local f, err_msg = io.open(tostring(file), 'w')
    if not f then
        error('SaveRelationSchema() Error: failed to open '..
              '"'..tostring(file)..'" for writing\n'..
              err_msg, 2)
    end
    local _, err_msg = f:write(json_str)
    f:close()
    if err_msg then
        error('SaveRelationSchema() Error: failed to write to '..
              '"'..tostring(file)..'"\n'..
              err_msg, 2)
    end
end



-------------------------------------------------------------------------------
--[[ Relation Schema Loading                                               ]]--
-------------------------------------------------------------------------------


-- helper for the following load functions
local function load_schema_json(filepath)
    -- check that the file is in fact a file
    if not filepath:is_file() then
        error('Could not find schema file "'..
              filepath:tostring()..'" on disk', 3)
    end

    -- now we can try to read
    local f, err_msg = io.open(filepath:tostring(), 'r')
    if not f then
        error('Error opening schema file "'..filepath:tostring()..'"\n'..
              err_msg, 3)
    end

    -- given a successful read, let's suck down the contents
    local json_str, err_msg = f:read('*all')
    f:close()
    if not json_str then
        error('Error reading schema file "'..filepath:tostring()..'"\n'..
              err_msg, 3)
    end

    -- parse the JSON string into an object
    local json_obj, err_msg = JSON.parse(json_str)
    if err_msg then
        error('Error parsing schema file "'..filepath:tostring()..'"\n'..
              err_msg, 3)
    end

    if type(json_obj) ~= 'table' then
        error('Schema JSON file "'..filepath:tostring()..'" '..
              'did not parse to an object', 3)
    end

    return json_obj
end

function L.LoadRelationSchemaNotes(params)
local interface_description = [[

    LoadRelationSchemaNotes assumes that it will be passed named arguments
    Arguments are as follows:

    file        = "..."     -- where to look for the schema.json file on disk
                            -- If the string ends in .json, then
                            --      try to load this exact file
                            -- Otherwise,
                            --      interpret as directory and look for
                            --      a schema.json file in that directory
]]
    if type(params) ~= 'table' or
       not is_path_or_str(params.file)
    then
        error(interface_description, 2)
    end

    -- ensure we have a pathname and allow for directory name convention
    local file = Pathname.new(params.file)
    if file:extname() ~= 'json' then
        file = file .. 'schema.json'
    end

    local json_obj = load_schema_json(file)

    -- extract the notes
    local notes = json_obj.notes
    if notes then
        if type(notes) ~= 'string' then
            error('Bad Schema JSON File "'..tostring(file)..'":\n'..
                  'Expected \'notes\' to be a string', 2)
        end
    end

    return notes
end



local function load_opt_hint_str(opt_str)
    return 'Pass argument '..opt_str..' to LoadRelationSchema()\n'..
           '  to suppress this error'
end

local function check_schema_json(json, opts)
    local err                   = opts.err or ''
    local allow_null_paths      = opts.allow_null_paths
    local allow_abs_paths       = opts.allow_abs_paths

    -- check version #
    if json.major_version ~= 0 or json.minor_version ~= 0 then
        error(err..
              'This Liszt relation loader only supports schema.json files '..
              'of version 0.0 (development);   given file has version '..
              json.major_version..'.'..json.minor_version, 3)
    end

    -- certify that the JSON object has relations
    if type(json.relations) ~= 'table' then
        error(err..'Could not find \'relations\' object', 3)
    end

    -- certify each relation
    for rname, rjson in pairs(json.relations) do
        -- check for valid name
        if not is_valid_lua_identifier(rname) then
            error(err..
                  'Invalid Relation name "'..rname..'"\n'..
                  valid_relation_name_err_msg, 3)
        end

        local relstr = 'Relation "'..rname..'"'

        -- check that the relation object is present,
        -- that it has a size,
        -- and that it has fields
        if type(rjson) ~= 'table' then
            error(err..relstr..' was not a JSON object', 3)
        elseif type(rjson.size) ~= 'number' then
            error(err..relstr..' was missing a "size" count', 3)
        elseif type(rjson.fields) ~= 'table' then
            error(err..relstr..' does not have a \'fields\' object', 3)
        end

        -- if the relation has an index, make sure it's a string and
        -- that it's a field-name present in the fields object
        if rjson.index then
            if type(rjson.index) ~= 'string' then
                error(err..relstr..' has an index of non-string type', 3)
            end

            local lookup = rjson.fields[rjson.index]
            if not lookup then
                error(err..relstr..' has index field "'..rjson.index..'" '..
                      'but an entry in the "fields" object '..
                      'could not be found.', 3)
            end
        end

        -- certify each field
        for fname, fjson in pairs(rjson.fields) do
            -- check for valid name
            if not is_valid_lua_identifier(fname) then
                error(err..'Invalid Field name "'..fname..'" on '..
                      relstr..'\n'..
                      valid_field_name_err_msg, 3)
            end

            local fstr = 'Field "'..fname..'" on '..relstr

            -- check that the field object is present and that it has a type
            if type(fjson) ~= 'table' then
                error(err..fstr..' was not a JSON object', 3)
            elseif type(fjson.type) ~= 'table' then
                error(err..fstr..' was missing a type object', 3)
            end
            
            -- check that the field object has a path (if req.),
            -- that the path is a valid pathname,
            -- and that the path is relative (if req.)
            local null_path_err -- hold any errors encountered
            if type(fjson.path) ~= 'string' then
                null_path_err = err..fstr..' was missing a path object'
            else
                local fpath
                local status, err_msg = pcall(function()
                    fpath = Pathname.new(fjson.path)
                end)

                if not status then
                    null_path_err = err..'Invalid path for '..fstr..'\n'..
                                    err_msg
                elseif fpath:is_absolute() and not allow_abs_paths then
                    null_path_err = err..'Absolute path for '..fstr..'\n'..
                                    'Absolute paths in schema files are '..
                                    'prohibited\n'..
                                    load_opt_hint_str('allow_abs_paths')
                end
            end
            if null_path_err and not allow_null_paths then
                error(null_path_err..'\n'..
                      load_opt_hint_str('allow_null_paths'), 3)
            end
        end
    end
end


-- the next set of calls are made on the actual relation object
local function json_deserialize_field(fname, fjson, params)
    local owner         = params.owner
    local schema_path   = params.schema_path
    local name_to_rel   = params.name_to_rel
    local err           = params.err
    local fstr          = 'Field "'..fname..'" '..
                          'on Relation "'..params.owner._name..'"'

    -- get the type
    local typ
    local status, err_msg = pcall(function()
        typ = T.Type.json_deserialize(fjson.type, params.name_to_rel)
    end)
    if not status then
        error(err..'Error while deserializing type of '..fstr..'\n'..
              err_msg, 3)
    end

    -- create the field in question
    local field = params.owner:NewField(fname, typ)

    -- load field data if we can get a reasonable path...
    local fpath
    if type(fjson.path) == 'string' and
       pcall(function() fpath = Pathname.new(fjson.path) end)
    then
        if fpath:is_relative() then
            fpath = params.schema_path:dirpath() .. fpath
        end

        local status, err_msg = pcall(function()
            field:LoadFromFile(tostring(fpath))
        end)
        if not status then
            error(err..'Error while loading field data of '..fstr..'\n'..
                  err_msg, 3)
        end
    end
end

function L.LoadRelationSchema(params)
local interface_description = [[

    LoadRelationSchema assumes that it will be passed named arguments
    Arguments are as follows:

    file            = "..." -- where to look for the schema.json file on disk
                            -- If the string ends in .json, then
                            --      try to load this exact file
                            -- Otherwise,
                            --      interpret as directory and look for
                            --      a schema.json file in that directory
                            --  (e.g. relations['id w/ spaces'] )
    allow_null_paths = bool -- If set to true, this option allows
                            --  the schema to load without successfully
                            --  loading default field data for all fields
    allow_abs_paths = bool  -- If set to true, this option allows
                            --  the schema to load field data from
                            --  absolute file paths.  By default,
                            --  absolute file paths will be considered
                            --  unreliable.
]]
    if type(params) ~= 'table' or
       not is_path_or_str(params.file)
    then
        error(interface_description, 2)
    end

    -- ensure we have a pathname and allow for directory name convention
    local file = Pathname.new(params.file)
    if file:extname():lower() ~= 'json' then
        file = file .. 'schema.json'
    end

    local json = load_schema_json(file)

    -- check for JSON formatting errors
    -- do this first to simplify the actual data extraction below
    local err_prefix = 'Bad Schema JSON File "'..tostring(file)..'":\n'
    check_schema_json(json, {
        err                 = err_prefix,
        allow_null_paths    = params.allow_null_paths,
        allow_abs_paths     = params.allow_abs_paths,
    })


    local relations = {}

    -- First, fill out all stub relations
    -- (do this first to allow Row types to match something)
    for rname, rjson in pairs(json.relations) do
        relations[rname] = L.NewRelation(rjson.size, rname)
    end
    -- Second, add fields to the stub relations
    for rname, rjson in pairs(json.relations) do
        local rel = relations[rname]

        -- deserialize each field
        -- TODO: there should probably be some kind of cleanup
        --      that kicks in to clean up memory if the load
        --      fails halfway through reading in all the fields.
        --    Punting under the assumption that the user is going
        --      to usually just restart the process in this case.
        for fname, fjson in pairs(rjson.fields) do
            json_deserialize_field(fname, fjson, {
                owner = rel,
                name_to_rel = relations,
                schema_path = file,
                err = err_prefix,
            })
        end

        -- reconstitute the index if one is present
        if rjson.index then
            rel:CreateIndex(rjson.index)
        end
    end

    return relations
end












