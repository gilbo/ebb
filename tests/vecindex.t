import "compiler.liszt"
local test = require "tests/test"


--------------------------
-- Kernel vector tests: --
--------------------------
local LMesh = terralib.require "compiler.lmesh"
local mesh = LMesh.Load("examples/mesh.lmesh")

------------------
-- Should pass: --
------------------
local vk = liszt_kernel (v : mesh.vertices)
    var x = {5, 4, 3}
    v.position += x
end
vk(mesh.vertices)

local x_out = L.NewScalar(L.float, 0.0)
local y_out = L.NewScalar(L.float, 0.0)
local y_idx = L.NewScalar(L.int, 1)
local read_out = liszt_kernel(v : mesh.vertices)
    x_out += v.position[0]
    y_out += v.position[y_idx]
end
read_out(mesh.vertices)

local avgx = x_out:value() / mesh.vertices._size
local avgy = y_out:value() / mesh.vertices._size
test.fuzzy_eq(avgx, 5)
test.fuzzy_eq(avgy, 4)

------------------
-- Should fail: --
------------------
idx = 3.5
local vk2 = liszt_kernel(v : mesh.vertices)
    v.position[idx] = 5
end
test.fail_kernel(vk2, mesh.vertices, "expected an integer")

