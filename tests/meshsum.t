import 'ebb'
local L = require 'ebblib'

local ioOff = require 'ebb.domains.ioOff'
local mesh  = ioOff.LoadTrimesh('tests/octa.off')

mesh.vertices:NewField('count', L.float)
mesh.vertices.count:Load(0)


local sum_count = ebb (e : mesh.edges)
	e.tail.count += 1
end

local check_count = ebb(v : mesh.vertices)
    var c = 0
    for e in v.edges do
        c += 1
    end
    L.assert(c == v.count)
end

mesh.edges:foreach(sum_count)
mesh.vertices:foreach(check_count)
