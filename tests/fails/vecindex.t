import "compiler/liszt"
mesh = L.initMeshRelationsFromFile("examples/mesh.lmesh")

idx = 3.5
local vk = liszt_kernel(v in mesh.vertices)
    v.position[idx] = 5
end

vk()
