package.path = package.path .. ";./tests/?.lua;?.lua"
local test = require "test"
import "compiler/liszt"


--------------------------
-- Kernel vector tests: --
--------------------------
mesh   = LoadMesh("examples/mesh.lmesh")
pos    = mesh:fieldWithLabel(Vertex, Vector(float, 3), "position")

------------------
-- Should fail: --
------------------
function test_types ()
  local idx = mesh:scalar(float, 0.0) -- cannot index with float
  local vk = liszt_kernel(v)
    pos(v)[idx] = 5
  end
  mesh.vertices:map(vk)
end
test_types()
