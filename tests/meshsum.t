import "compiler.liszt"
local LMesh = terralib.require "compiler.lmesh"

local mesh = LMesh.Load("examples/mesh.lmesh")
mesh.vertices:NewField('count', L.float)
mesh.vertices.count:LoadFromCallback(terra (mem: &float, i : uint) mem[0] = 0 end)


local sum_count = liszt_kernel (e in mesh.edges)
	e.head.count += 1
	e.tail.count += 1
end

local check_count = liszt_kernel(v in mesh.vertices)
    var c = 0
    for e in v.edges do
        c += 1
    end
    L.assert(c == v.count)
end

sum_count()
check_count()
