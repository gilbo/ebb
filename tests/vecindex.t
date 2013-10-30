package.path = package.path .. ";./tests/?.lua;?.lua"
local test = require "test"
import "compiler/liszt"


--------------------------
-- Kernel vector tests: --
--------------------------
mesh   = LoadMesh("examples/mesh.lmesh")
pos    = mesh:fieldWithLabel(Vertex, Vector(float, 3), "position")

------------------
-- Should pass: --
------------------
function test_vector_indexing ()
  local vk = liszt_kernel (v)
    var x = {5, 4, 3}
    pos(v) += x
  end
  mesh.vertices:map(vk)

  local x_out = mesh:scalar(float, 0.0)
  local y_out = mesh:scalar(float, 0.0)
  local y_idx = mesh:scalar(int, 1)
  local read_out = liszt_kernel(v)
    x_out += pos(v)[0]
    y_out += pos(v)[y_idx]
  end
  mesh.vertices:map(read_out)

  local avgx = x_out:value() / mesh.vertices:size()
  local avgy = y_out:value() / mesh.vertices:size()
  test.fuzzy_eq(avgx, 5)
  test.fuzzy_eq(avgy, 4)
end
test_vector_indexing()

