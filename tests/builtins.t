import "compiler.liszt"
require "tests/test"


local ioOff = L.require 'domains.ioOff'
local mesh  = ioOff.LoadTrimesh('tests/octa.off')

test.fail_function(function()
  local liszt bad_id (e : mesh.edges)
    var v = L.id({ e.head, e.tail })
    L.print(v)
  end
  mesh.edges:map(bad_id)
end, "expected a relational key")


test.fail_function(function()
  local liszt bad_length (e : mesh.edges)
    var l = L.length({e.head, e.tail})
    L.print(l)
  end
  mesh.edges:map(bad_length)
end, "length expects vectors of numeric type")
