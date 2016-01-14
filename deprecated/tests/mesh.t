--DISABLE-TEST
--DISABLE-ON-LEGION
import "ebb"

local LMesh = require "ebb.domains.lmesh"
local M = LMesh.Load("examples/mesh.lmesh")

M.vertices:Print()
M.faces:Print()
M.edges:Print()
M.cells:Print()
M.edgesofface:Print()
M.edgesofcell:Print()
M.verticesofface:Print()