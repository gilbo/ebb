import "ebb"
require "tests/test"


local ioOff = require 'ebb.domains.ioOff'
local mesh  = ioOff.LoadTrimesh('tests/octa.off')

test.fail_function(function()
  local ebb bad_id (e : mesh.edges)
    var v = L.id({ e.head, e.tail })
    L.print(v)
  end
  mesh.edges:foreach(bad_id)
end, "expected a relational key")


test.fail_function(function()
  local ebb bad_length (e : mesh.edges)
    var l = L.length({e.head, e.tail})
    L.print(l)
  end
  mesh.edges:foreach(bad_length)
end, "length expects vectors of numeric type")
