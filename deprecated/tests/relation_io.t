import "compiler.liszt"

local test  = require "tests/test"

local LMesh = L.require "domains.lmesh"
local PN    = require 'lib.pathname'
local C     = require 'compiler.c'

local datadir = PN.scriptdir() .. 'relation_io_data'

local compare_field_cache = {}
local function compare_field_data(fa, fb, fname)
  local ttype = fa.type:terraType()
  local N = (fa.type:isVector() and fa.type.N) or 0

  local comparefn = compare_field_cache[fa.type]
  if not comparefn then
    local test_not_eq
    if fa.type:isVector() then
      test_not_eq = terra(a : &ttype, b : &ttype, i : int)
        for j = 0, N do
          if a[i].d[j] ~= b[i].d[j] then return true end
        end
        return false
      end
    else
      test_not_eq = terra(a : &ttype, b : &ttype, i : int)
        return a[i] ~= b[i]
      end
    end

    comparefn = terra(a : &ttype, b : &ttype, n : int) : int
      for i = 0, n do
        if test_not_eq(a,b,i) then return i end
      end
      return n
    end

    compare_field_cache[fa.type] = comparefn
  end

  local function vecprint(cdata)
    local res = '{'
    for i=0,N-1 do res = res .. tostring(cdata.d[i]) .. ',' end
    return res .. '}'
  end
  local i = comparefn(fa:DataPtr(), fb:DataPtr(), fa:Size())
  if i ~= fa:Size() then
    local da = tostring(fa:DataPtr()[i])
    local db = tostring(fb:DataPtr()[i])
    if fa.type:isVector() then
      da = vecprint(fa:DataPtr()[i])
      db = vecprint(fb:DataPtr()[i])
    end
    error('data inconsistency in field "'..fname..
          '" at position #'..tostring(i)..
          '  '..da..'  vs.  '..db, 2)
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

      -- check grouping equality
      if rela._grouped and relb._grouped then
        test.eq(rela._grouped.key_field:Name(),
                relb._grouped.key_field:Name())
      else
        test.eq(rela._grouped, relb._grouped)
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
        compare_field_data(fielda, fieldb, fname)
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
} end, '<root>: expected object')

-- should be able to read in minimum but useless schema without error
test.seteq(L.LoadRelationSchema{ file = datadir .. 'min_schema.json' }, {})

-- test note reading
test.eq(L.LoadRelationSchemaNotes{ file = datadir .. 'min_schema.json' }, nil)
test.fail_function(function() L.LoadRelationSchemaNotes{
  file = datadir .. 'min_schema_w_bad_notes.json'
} end, "<root>.notes: expected string")
test.eq(L.LoadRelationSchemaNotes{
  file = datadir .. 'min_schema_w_notes.json'
}, 'sample note')

-- loading broken schema file formats should produce the following errors
test.fail_function(function() L.LoadRelationSchema{
  file = datadir .. 'unsupported_version.json',
} end, 'Liszt relation loader only supports schema.json files of version 0.0')
test.fail_function(function() L.LoadRelationSchema{
  file = datadir .. 'missing_relations.json',
} end, 'expected to find key \'relations\'')
test.fail_function(function() L.LoadRelationSchema{
  file = datadir .. 'broken_simp/invalid_rel_name.json',
} end, 'Invalid Relation name')
test.fail_function(function() L.LoadRelationSchema{
  file = datadir .. 'broken_simp/no_rel_obj.json',
} end, '<root>.relations.cells: expected object')
test.fail_function(function() L.LoadRelationSchema{
  file = datadir .. 'broken_simp/no_rel_size.json',
} end, 'expected to find key \'size\'')
test.fail_function(function() L.LoadRelationSchema{
  file = datadir .. 'broken_simp/no_rel_fields.json',
} end, 'expected to find key \'fields\'')
test.fail_function(function() L.LoadRelationSchema{
  file = datadir .. 'broken_simp/non_str_index.json',
} end, '<root>.relations.cells.grouped_by: expected string')
test.fail_function(function() L.LoadRelationSchema{
  file = datadir .. 'broken_simp/dangling_index.json',
} end, 'is grouped by field "abc" but that field couldn\'t be found.')
test.fail_function(function() L.LoadRelationSchema{
  file = datadir .. 'broken_simp/invalid_field_name.json',
} end, 'Invalid Field name')
test.fail_function(function() L.LoadRelationSchema{
  file = datadir .. 'broken_simp/no_field_obj.json',
} end, '<root>.relations.cells.fields.temperature: expected object')
test.fail_function(function() L.LoadRelationSchema{
  file = datadir .. 'broken_simp/no_field_type.json',
} end, '<root>.relations.cells.fields.temperature: '..
       'expected to find key \'type\'')
test.fail_function(function() L.LoadRelationSchema{
  file = datadir .. 'broken_simp/no_field_path.json',
} end, 'Field "temperature" on Relation "cells" was missing a path object')
test.fail_function(function() L.LoadRelationSchema{
  file = datadir .. 'broken_simp/invalid_field_path.json',
} end, 'Invalid path for Field')
test.fail_function(function() L.LoadRelationSchema{
  file = datadir .. 'broken_simp/absolute_field_path.json',
} end, 'Absolute paths are prohibited')


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
  return L.NewVector(L.float, {i % 10 + 0.5, 0.0, 0.0})
end)
simp_rels.particle_cell.p:LoadFunction(function(i) return i end)
simp_rels.particle_cell.c:LoadFunction(function(i) return i % 10 end)

-- load a disk copy of the relation above
local simp_rels_load = L.LoadRelationSchema{ file = datadir .. 'simp' }
test_db_eq(simp_rels, simp_rels_load)




-- basic schema file loading should work regardless of whether we
-- use a directory path or the schema file itself.
-- TODO
















