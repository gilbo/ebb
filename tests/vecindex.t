import "compiler.liszt"
local test = require "tests/test"


local ioOff = L.require 'domains.ioOff'
local mesh  = ioOff.LoadTrimesh('tests/octa.off')

------------------
-- Should pass: --
------------------
local vk = liszt (v : mesh.vertices)
    var x = {5, 4, 3}
    v.pos += x
end
mesh.vertices:foreach(vk)

local x_out = L.Global(L.float, 0.0)
local y_out = L.Global(L.float, 0.0)
local y_idx = L.Global(L.int, 1)
local read_out_const = liszt(v : mesh.vertices)
    x_out += L.float(v.pos[0])
end
local read_out_var = liszt(v : mesh.vertices)
    y_out += L.float(v.pos[y_idx])
end
mesh.vertices:foreach(read_out_const)
mesh.vertices:foreach(read_out_var)

local avgx = x_out:get() / mesh.vertices:Size()
local avgy = y_out:get() / mesh.vertices:Size()
test.fuzzy_eq(avgx, 5)
test.fuzzy_eq(avgy, 4)

------------------
-- Should fail: --
------------------
idx = 3.5
test.fail_function(function()
  local liszt t(v : mesh.vertices)
      v.pos[idx] = 5
  end
  mesh.vertices:foreach(t)
end, "expected an integer")

