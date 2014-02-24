import "compiler.liszt"
local LMesh = terralib.require "compiler.lmesh"
local M = LMesh.Load("examples/mesh.lmesh")

M.vertices:NewField('field1', L.float):LoadConstant(0)
M.vertices:NewField('field2', L.float):LoadConstant(0)

local kernel = liszt_kernel (v : M.vertices)
	var x = v.field1
	v.field1 += 3
end

kernel(M.vertices)