import "compiler.liszt"
local test = require "tests/test"


--------------------------
-- Kernel vector tests: --
--------------------------
local LMesh = L.require "domains.lmesh"
local mesh = LMesh.Load("examples/mesh.lmesh")

------------------
-- Should pass: --
------------------
local vk = liszt kernel (v : mesh.vertices)
    var x = {5, 4, 3}
    v.position += x
end
vk(mesh.vertices)

local x_out = L.NewGlobal(L.float, 0.0)
local y_out = L.NewGlobal(L.float, 0.0)
local y_idx = L.NewGlobal(L.int, 1)
local read_out = liszt kernel(v : mesh.vertices)
    x_out += L.float(v.position[0])
    y_out += L.float(v.position[y_idx])
end
read_out(mesh.vertices)

local avgx = x_out:get() / mesh.vertices:Size()
local avgy = y_out:get() / mesh.vertices:Size()
test.fuzzy_eq(avgx, 5)
test.fuzzy_eq(avgy, 4)

------------------
-- Should fail: --
------------------
idx = 3.5
test.fail_function(function()
  liszt kernel(v : mesh.vertices)
      v.position[idx] = 5
  end
end, "expected an integer")

