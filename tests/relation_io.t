import "compiler.liszt"

local test  = require "tests/test"

local LMesh = terralib.require 'compiler.lmesh'
local PN    = terralib.require 'compiler.pathname'
local C     = terralib.require 'compiler.c'

local datadir = PN.scriptdir() .. 'relation_io_data'

local compare_field_cache = {}
local function compare_field_data(fa, fb)
  local ttype = fa.type:terraType()
  local typeN = 0
  if ttype:isvector() then typeN = ttype.N end

  local comparefn = compare_field_cache[ttype]
  if not comparefn then
    local testeq
    if ttype:isvector() then
      testeq = terra(a : &ttype, b : &ttype, i : int)
        var boolvec = a[i] ~= b[i]
        var test = false
        for k = 0, ttype.N do
          test = test or boolvec[k]
        end
        return test
      end
    else
      testeq = terra(a : &ttype, b : &ttype, i : int)
        return a[i] ~= b[i]
      end
    end

    comparefn = terra(a : &ttype, b : &ttype, n : int) : int
      for i = 0, n do
        if testeq(a,b,i) then return i end
      end
      return n
    end

    compare_field_cache[ttype] = comparefn
  end

  local i = comparefn(fa.data, fb.data, fa:Size())
  if i ~= fa:Size() then
    error('data inconsistency at position #'..tostring(i)..
          '  '..tostring(fa.data[i])..'  vs.  '..tostring(fb.data[i]), 2)
  end
end

local function test_db_eq(dba, dbb)
  -- helper
  local function fnameset(rel)
    local fs = {}
    for _, f in ipairs(rel._fields) do fs[f.name] = true end
    return fs
  end

  local dba_name = {}
  for k,v in pairs(dba) do dba_name[v] = k end
  local dbb_name = {}
  for k,v in pairs(dbb) do dbb_name[v] = k end

  -- wrap work for error reporting
  local function subcall()
    test.seteq(dba, dbb)
    for rname, rela in pairs(dba) do
      local relb = dbb[rname]

      -- check index equality
      if rela._index then
        test.eq(rela._index.name, relb._index.name)
      else
        test.eq(rela._index, relb._index)
      end

      -- check size equality
      test.eq(rela:Size(), relb:Size())

      -- check field equality...
      test.seteq(fnameset(rela), fnameset(relb))
      for _, fielda in ipairs(rela._fields) do
        local fname = fielda.name
        local fieldb = relb[fname]

        -- check type equality
        if fielda.type:isRow() then
          test.eq(fieldb.type:isRow(), true)
          test.eq(dba_name[fielda.type.relation],
                  dbb_name[fieldb.type.relation])
        else
          test.eq(fielda.type, fieldb.type)
        end

        -- check data equality
        compare_field_data(fielda, fieldb)
      end
    end
  end

  local status, err_msg = pcall(subcall)
  if not status then
    error('error while comparing relational dbs\n'..err_msg, 2)
  end
end


-- ensure that the LMesh loading matches the
-- equivalent relational mesh loading
local compat_lmesh = LMesh.Load(datadir:concat('compat.lmesh'):tostring())
local compat_relmesh = L.LoadRelationSchema {
  file = datadir:concat('compat_relmesh')
}
test_db_eq(compat_lmesh, compat_relmesh)

-- calling improperly will produce calling instructions
test.fail_function(L.SaveRelationSchema, 'Arguments are as follows:')
test.fail_function(L.LoadRelationSchema, 'Arguments are as follows:')
test.fail_function(L.LoadRelationSchemaNotes, 'Arguments are as follows:')
test.fail_function(function()
  L.LoadRelationSchemaNotes { file = true } -- bad type
end, 'Arguments are as follows:')
test.fail_function(function()
  L.LoadRelationSchema { file = true }
end, 'Arguments are as follows:')
test.fail_function(function()
  L.SaveRelationSchema { relations = {}, file = true }
end, 'Arguments are as follows:')
test.fail_function(function()
  L.SaveRelationSchema { relations = 23, file = 'abc123' }
end, 'Arguments are as follows:')

-- Using invalid pathnames will produce errors
test.fail_function(function() L.SaveRelationSchema{
  relations = {},
  file = '#@#9ijadfpoin',
} end, 'Bad Pathname')

-- broken basic schema file loading should produce the following errors
test.fail_function(function() L.LoadRelationSchema{
  file = datadir .. 'does_not_exist.json',
} end, 'Could not find schema file')
test.fail_function(function() L.LoadRelationSchema{
  file = datadir .. 'invalid_json.json',
} end, 'Error parsing schema file')
test.fail_function(function() L.LoadRelationSchema{
  file = datadir .. 'not_json_obj.json',
} end, 'Schema JSON file .* did not parse to an object')

-- should be able to read in minimum but useless schema without error
test.seteq(L.LoadRelationSchema{ file = datadir .. 'min_schema.json' }, {})

-- test note reading
test.eq(L.LoadRelationSchemaNotes{ file = datadir .. 'min_schema.json' }, nil)
test.fail_function(function() L.LoadRelationSchemaNotes{
  file = datadir .. 'min_schema_w_bad_notes.json'
} end, "Expected 'notes' to be a string")
test.eq(L.LoadRelationSchemaNotes{
  file = datadir .. 'min_schema_w_notes.json'
}, 'sample note')

-- loading broken schema file formats should produce the following errors
test.fail_function(function() L.LoadRelationSchema{
  file = datadir .. 'unsupported_version.json',
} end, 'Liszt relation loader only supports schema.json files of version 0.0')
test.fail_function(function() L.LoadRelationSchema{
  file = datadir .. 'missing_relations.json',
} end, 'Could not find \'relations\' object')
test.fail_function(function() L.LoadRelationSchema{
  file = datadir .. 'broken_simp/invalid_rel_name.json',
} end, 'Invalid Relation name')
test.fail_function(function() L.LoadRelationSchema{
  file = datadir .. 'broken_simp/no_rel_obj.json',
} end, 'Relation .* was not a JSON object')
test.fail_function(function() L.LoadRelationSchema{
  file = datadir .. 'broken_simp/no_rel_size.json',
} end, 'Relation .* was missing a "size" count')
test.fail_function(function() L.LoadRelationSchema{
  file = datadir .. 'broken_simp/no_rel_fields.json',
} end, 'does not have a \'fields\' object')
test.fail_function(function() L.LoadRelationSchema{
  file = datadir .. 'broken_simp/non_str_index.json',
} end, 'has an index of non%-string type') -- % is Lua pattern matching esc.
test.fail_function(function() L.LoadRelationSchema{
  file = datadir .. 'broken_simp/dangling_index.json',
} end, 'an entry in the "fields" object could not be found.')
test.fail_function(function() L.LoadRelationSchema{
  file = datadir .. 'broken_simp/invalid_field_name.json',
} end, 'Invalid Field name')
test.fail_function(function() L.LoadRelationSchema{
  file = datadir .. 'broken_simp/no_field_obj.json',
} end, 'Field .* was not a JSON object')
test.fail_function(function() L.LoadRelationSchema{
  file = datadir .. 'broken_simp/no_field_type.json',
} end, 'Field .* was missing a type object')
test.fail_function(function() L.LoadRelationSchema{
  file = datadir .. 'broken_simp/no_field_path.json',
} end, 'Field .* was missing a path object')
test.fail_function(function() L.LoadRelationSchema{
  file = datadir .. 'broken_simp/invalid_field_path.json',
} end, 'Invalid path for Field')
test.fail_function(function() L.LoadRelationSchema{
  file = datadir .. 'broken_simp/absolute_field_path.json',
} end, 'Absolute paths in schema files are prohibited')


-- broken type json formatting should produce the following errors
-- TODO




-- build relation schema
local simp_rels = {}
simp_rels.cells         = L.NewRelation(10, 'cells')
  simp_rels.cells:NewField('temperature', L.double)
simp_rels.particles     = L.NewRelation(30, 'particles')
  simp_rels.particles:NewField('temperature', L.double)
  simp_rels.particles:NewField('position', L.vector(L.double, 3))
simp_rels.particle_cell = L.NewRelation(30, 'particle_cell')
  simp_rels.particle_cell:NewField('p', simp_rels.particles)
  simp_rels.particle_cell:NewField('c', simp_rels.cells)

-- intialize relation
simp_rels.cells.temperature:LoadFunction(function(i)
  return 25.0 - (4.5-i)*(4.5-i)
end)
simp_rels.particles.temperature:LoadConstant(0.0)
simp_rels.particles.position:LoadFunction(function(i)
  return vector(float, i % 10 + 0.5, 0.0, 0.0)
end)
simp_rels.particle_cell.p:LoadFunction(function(i) return i end)
simp_rels.particle_cell.c:LoadFunction(function(i) return i % 10 end)


-- load a disk copy of the relation above
local simp_rels_load = L.LoadRelationSchema{ file = datadir .. 'simp' }
test_db_eq(simp_rels, simp_rels_load)




-- basic schema file loading should work regardless of whether we
-- use a directory path or the schema file itself.
-- TODO




print('done')












