
-- file/module namespace table
local Serial = {}
package.loaded["compiler.serialization"] = Serial

terralib.require('compiler.relations')

local L = terralib.require "compiler.lisztlib"
local T = terralib.require "compiler.types"
local C = terralib.require "compiler.c"

local PN = terralib.require "lib.pathname"
local Pathname = PN.Pathname
local ffi = require('ffi')

local JSON = require('compiler.JSON')


-------------------------------------------------------------------------------
-------------------------------------------------------------------------------
--[[ Serialization / Deserialization                                       ]]--
-------------------------------------------------------------------------------
-------------------------------------------------------------------------------

local function is_path_or_str(obj)
    return type(obj) == 'string' or PN.is_pathname(obj)
end

local bit = require "bit"



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
    C.fseek ( file, header.header_size, C.SEEK_SET )
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
    C.fseek( file, header.header_size,  C.SEEK_SET )
    
    return 0
end

function L.LField:SaveToFile(filename)
    assert(self.array)

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
    local n_rows    = self:Size()
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
    C.fwrite( self.array:ptr(), tsize, n_rows, file )
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
    C.fread( self.array:ptr(), type_size, array_size, file )
    if C.ferror(file) ~= 0 then
        C.perror('field file data read error: ')
        self:ClearData()
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

local JSONSchema = terralib.require 'compiler.JSONSchema'

local primitive_type_schema = {
  basic_kind  = 'primitive',
  primitive   = JSONSchema.String
}
local type_schema = JSONSchema.OR {
  primitive_type_schema,
  {
    basic_kind  = 'vector',
    base        = primitive_type_schema,
    n           = JSONSchema.Num
  },
  {
    basic_kind  = 'row',
    relation    = JSONSchema.String
  }
}
local schema_file_format = JSONSchema.New {
  ['major_version'] = JSONSchema.Num,
  ['minor_version'] = JSONSchema.Num,
  ['?'] = {
    ['notes'] = JSONSchema.String,
  },
  ['relations'] = {
    ['*'] = {
      ['fields'] = {
        ['*'] = {
          ['?'] = {
            ['path'] = JSONSchema.String
          },
          ['type'] = type_schema
        }
      },
      ['size'] = JSONSchema.Num,
      ['?'] = { -- optional fields
          ['grouped_by'] = JSONSchema.String
      }
    }
  }
}




--local function save_opt_hint_str(opt_str)
--    return 'Pass argument '..opt_str..' to SaveRelationSchema()\n'..
--           '  to suppress this error.'
--end

local function check_save_relations(relations, params)
    local rel_to_name       = params.rel_to_name
    local no_file_data      = params.no_file_data

    for rname, rel in pairs(relations) do
        local rstr = 'Relation "'..rname..'"'

        -- The relations must actually be relations
        if not L.is_relation(rel) then
            error('SaveRelationSchema() Error: '..rstr..
                  ' is not a relation', 3)
        end

        -- The relation names must be valid
        if not L.is_valid_lua_identifier(rname) then
            error('SaveRelationSchema() Error: '..rstr..
                  ' has an invalid name\n'..
                  L.valid_name_err_msg.relation, 3)
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
            if not L.is_valid_lua_identifier(fname)
            then
                error('SaveRelationSchema() Error: '..fstr..
                      ' has an invalid name\n'..
                      L.valid_name_err_msg.field, 3)
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
                if not f.array then -- in case of cdata pointer
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
        size    = relation:Size(),
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

    -- attach a grouping flag if the relation is grouped
    if relation._grouping then
        json.grouped_by = relation._grouping.key_field:Name()
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

    local schema_errors = {}
    if not JSONSchema.match(schema_file_format, json_obj, schema_errors) then
        local match_err = 'Schema JSON file "'..filepath:tostring()..'" '..
                          ' had formatting errors:'
        for _,e in ipairs(schema_errors) do
            match_err = match_err .. '\n' .. e
        end
        error(match_err, 3)
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

    local  json_obj = load_schema_json(file)
    return json_obj.notes
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

    -- certify each relation
    for rname, rjson in pairs(json.relations) do
        -- check for valid name
        if not L.is_valid_lua_identifier(rname) then
            error(err..
                  'Invalid Relation name "'..rname..'"\n'..
                  L.valid_name_err_msg.relation, 3)
        end

        local relstr = 'Relation "'..rname..'"'

        -- if the relation is grouped, then make sure it's a valid grouping
        if rjson.grouped_by then
            local lookup = rjson.fields[rjson.grouped_by]
            if not lookup then
                error(err..relstr..' is grouped by field '..
                      '"'..rjson.grouped_by..'" but that field '..
                      'couldn\'t be found.', 3)
            end
        end

        -- certify each field
        for fname, fjson in pairs(rjson.fields) do
            -- check for valid name
            if not L.is_valid_lua_identifier(fname) then
                error(err..'Invalid Field name "'..fname..'" on '..
                      relstr..'\n'..
                      L.valid_name_err_msg.field, 3)
            end

            local fstr = 'Field "'..fname..'" on '..relstr
            
            -- check that the field object has a path (if req.),
            -- that the path is a valid pathname,
            -- and that the path is relative (if req.)
            if not allow_null_paths then
                local null_path_err
                if not fjson.path then
                    error(err..fstr..' was missing a path object\n'..
                          load_opt_hint_str('allow_null_paths'), 3)
                end

                local fpath
                local status, err_msg = pcall(function()
                    fpath = Pathname.new(fjson.path)
                end)

                if not status then
                    error(err..'Invalid path for '..fstr..'\n'..err_msg, 3)
                end
                if not allow_abs_paths and fpath:is_absolute() then
                    error(err..'Absolute path for '..fstr..'\n'..
                               'Absolute paths are prohibited\n'..
                               load_opt_hint_str('allow_abs_paths'), 3)
                end
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
    local field = owner:NewField(fname, typ)

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

        -- install a grouping if one was declared
        if rjson.grouped_by then
            rel:GroupBy(rjson.grouped_by)
        end
    end

    return relations
end



