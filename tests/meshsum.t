import "compiler.liszt"

local ioOff = L.require 'domains.ioOff'
local mesh  = ioOff.LoadTrimesh('tests/octa.off')

mesh.vertices:NewField('count', L.float)
mesh.vertices.count:LoadConstant(0)


local sum_count = liszt (e : mesh.edges)
	e.tail.count += 1
end

local check_count = liszt(v : mesh.vertices)
    var c = 0
    for e in v.edges do
        c += 1
    end
    L.assert(c == v.count)
end

mesh.edges:foreach(sum_count)
mesh.vertices:foreach(check_count)
