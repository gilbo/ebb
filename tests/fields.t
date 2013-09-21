--[[ Note: this test file is not at all comprehensive for making sure that field reads/writes
	 translate correctly to terra code.  Right now, all it does it make sure that the codegen
	 produces something that can compile.
]]

import "compiler/liszt"

mesh  = LoadMesh("examples/mesh.lmesh")
pos   = mesh:fieldWithLabel(Vertex, Vector.type(float, 3), "position")
field = mesh:field(Face, float, 0.0)
field2 = mesh:field(Face, float, 0.0)
field3 = mesh:field(Face, float, 0.0)
field4 = mesh:field(Cell, bool, 0.0)
field5 = mesh:field(Vertex, Vector.type(float, 4), {0.0, 0.0, 0.0, 0.0})

local a = global(int, 6)
local b = Vector.new(float, {1, 3, 4, 5})

local reduce1 = liszt_kernel (f)
	field(f) = field(f) - 3 + 243.3 * a
end

local reduce2 = liszt_kernel (f)
	field2(f) = field2(f) * 3 * 7 / 3
end

local reduce3 = liszt_kernel (f)
	field3(f) = field3(f) 
end

local read1 = liszt_kernel (f)
	var tmp = field3(f) + 5
end

local write1 = liszt_kernel(f)
	field3(f) = 0.0
end

local write2 = liszt_kernel (f)
	field5(f) = b
end

mesh.faces:map(reduce1)
mesh.faces:map(reduce2)
mesh.faces:map(reduce3)
mesh.faces:map(read1)
mesh.faces:map(write1)
mesh.faces:map(write2)
