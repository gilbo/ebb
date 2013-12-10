import 'compiler.liszt'

local LMesh = terralib.require "compiler.lmesh"
local mesh = LMesh.LoadUniformGrid(0, {1, 1, 1})
--local mesh = LMesh.Load("examples/mesh.lmesh")

local mesh_rels = {
    'verticesofcell',
    'edgesofcell',
}

for i,rel in ipairs(mesh_rels) do
    mesh[rel]:print()
end
