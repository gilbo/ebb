import "compiler.liszt"
local LMesh = terralib.require "compiler.lmesh"
local M = LMesh.Load("examples/mesh.lmesh")

local init_to_zero = terra (mem : &float, i : int) mem[0] = 0 end
M.vertices:NewField('field1', L.float):LoadFromCallback(init_to_zero)
M.vertices:NewField('field2', L.float):LoadFromCallback(init_to_zero)

local kernel = liszt_kernel (v in M.vertices)
	var x = v.field1
	v.field1 += 3
end

kernel(M.vertices)