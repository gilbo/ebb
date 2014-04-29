import "compiler.liszt"

local LMesh = L.require "domains.lmesh"
local mesh = LMesh.Load("examples/mesh.lmesh")

mesh.faces:NewField('field', L.float)
mesh.faces.field:LoadConstant(0)

local lassert, lprint, length = L.assert, L.print, L.length

local a     = 43
local com   = L.NewGlobal(L.vector(L.float, 3), {0, 0, 0})--Vector.new(float, {0.0, 0.0, 0.0})
local upval = 5
local vv    = L.NewVector(L.float, {1,2,3})

local test_for = liszt kernel (f : mesh.faces)
	for v in f.vertices do
	    lprint(f,v)
	end
end
test_for(mesh.faces)

