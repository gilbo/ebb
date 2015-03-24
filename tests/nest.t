import "compiler.liszt"

local ioOff = L.require 'domains.ioOff'
local mesh  = ioOff.LoadTrimesh('tests/octa.off')

mesh.vertices:NewField('field', L.float)
mesh.vertices.field:Load(1)
local count = L.Global(L.float, 0)

local test_for = liszt (v : mesh.vertices)
	for e in v.edges do
	  count += 1
	end
end

mesh.vertices:map(test_for)

assert(count:get() == 24)
