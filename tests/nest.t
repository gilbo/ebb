import "compiler.liszt"

local LMesh = terralib.require "compiler.lmesh"
local mesh = LMesh.Load("examples/mesh.lmesh")

mesh.faces:NewField('field', L.float)
mesh.faces.field:LoadFromCallback(terra (mem: &float, i : uint) mem[0] = 0 end)

local lassert, lprint, length = L.assert, L.print, L.length

local a     = 43
local com   = L.NewScalar(L.vector(L.float, 3), {0, 0, 0})--Vector.new(float, {0.0, 0.0, 0.0})
local upval = 5
local vv    = L.NewVector(L.float, {1,2,3})

local test_for = liszt_kernel (f in mesh.faces)
	for v in f.vertices do
	    lprint(f,v)
	end
end
test_for(mesh.faces)

