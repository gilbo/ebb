import "compiler/liszt"

mesh = LoadMesh("examples/mesh.lmesh")
pos  = mesh:fieldWithLabel(L.vertex, L.vector(L.float, 3), "position")
idx  = mesh:scalar(L.float, 0.0) -- cannot index with float

local vk = liszt_kernel(v)
    pos(v)[idx] = 5
end
mesh.vertices:map(vk)
