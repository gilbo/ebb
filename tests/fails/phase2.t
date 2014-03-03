import "compiler.liszt"
local LMesh = terralib.require "compiler.lmesh"
local M = LMesh.Load("examples/mesh.lmesh")

M.vertices:NewField('field1', L.float):LoadConstant(0)
M.vertices:NewField('field2', L.float):LoadConstant(0)

local kernel = liszt_kernel (v : M.vertices)
	v.field1 = 1.3
	v.field1 += 1
end

kernel(M.vertices)