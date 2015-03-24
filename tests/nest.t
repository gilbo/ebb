import "compiler.liszt"

local LMesh = L.require "domains.lmesh"
local mesh = LMesh.Load("examples/mesh.lmesh")

mesh.faces:NewField('field', L.float)
mesh.faces.field:LoadConstant(1)
local count = L.Global(L.float, 0)

local test_for = liszt (f : mesh.faces)
	for v in f.vertices do
	  count += 1
	end
end

mesh.faces:map(test_for)

assert(count:get() == 24)
