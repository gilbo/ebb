import 'compiler/liszt'
local length, lprint = L.length, L.print

-- Test code
local LMesh = terralib.require "compiler.lmesh"
local PN = terralib.require 'compiler.pathname'
local M = LMesh.Load(PN.scriptdir():concat("rmesh.lmesh"):tostring())


local relation_list = {}
for k,v in pairs(M) do
  if not k:match('^__') then
    relation_list[k] = v
  end
end

L.SaveRelationSchema {
  relations = relation_list,
  file      = "./relation_io_test_relation",
  notes     = "these are some notes",
}

local relations, err_msg = L.LoadRelationSchema {
  file = "./relation_io_test_relation",
}


for k,v in pairs(relations) do
    relations[k]:print()
end


print('BEFORE')
relations.vertices.position:print()

relations.vertices.position:LoadFromFile(
  './relation_io_test_relation/vertices/position.field'
)

print('AFTER')
relations.vertices.position:print()


