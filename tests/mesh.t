--DISABLE-ON-LEGION
import "compiler.liszt"

local LMesh = L.require "domains.lmesh"
local M = LMesh.Load("examples/mesh.lmesh")

M.vertices:print()
M.faces:print()
M.edges:print()
M.cells:print()
M.edgesofface:print()
M.edgesofcell:print()
M.verticesofface:print()