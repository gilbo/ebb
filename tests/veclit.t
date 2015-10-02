import "ebb"
require "tests/test"

local ioOff = require 'ebb.domains.ioOff'
local mesh  = ioOff.LoadTrimesh('tests/octa.off')

test.fail_function(function()
  local ebb t(v : mesh.vertices)
  	var x = {1, 2, true}
  end
  mesh.vertices:foreach(t)
end, "must be of the same type")

test.fail_function(function()
  local ebb t(v : mesh.vertices)
  	var x = {1, 2, {2, 3}}
  end
  mesh.vertices:foreach(t)
end, "can only contain scalar values")
