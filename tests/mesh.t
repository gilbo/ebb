import "compiler/liszt"

local M = L.initMeshRelationsFromFile('examples/mesh.lmesh')

M.vertices:print()
M.faces:print()
M.edges:print()
M.cells:print()
M.edgesofface:print()
M.edgesofcell:print()
M.verticesofface:print()