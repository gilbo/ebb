import "compiler.liszt"

local LMesh = L.require "domains.lmesh"
local mesh = LMesh.Load("examples/mesh.lmesh")

mesh.faces:NewField('field', L.float)
mesh.faces.field:LoadConstant(0)

local test_for = liszt (f : mesh.faces)
	for v in f.vertices do
	    L.print(f,v)
	end
end

mesh.faces:map(test_for)

