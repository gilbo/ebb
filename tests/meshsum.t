import "compiler.liszt"
local LMesh = L.require "domains.lmesh"

local mesh = LMesh.Load("examples/mesh.lmesh")
mesh.vertices:NewField('count', L.float)
mesh.vertices.count:LoadConstant(0)


local sum_count = liszt (e : mesh.edges)
	e.head.count += 1
	e.tail.count += 1
end

local check_count = liszt(v : mesh.vertices)
    var c = 0
    for e in v.edges do
        c += 1
    end
    L.assert(c == v.count)
end

mesh.edges:map(sum_count)
mesh.vertices:map(check_count)
