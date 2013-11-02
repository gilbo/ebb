import "compiler/liszt"
mesh = L.initMeshRelationsFromFile("examples/mesh.lmesh")

local vk = liszt_kernel(v in mesh.vertices)
	var x = {1, 2, {2, 3}}
end
vk()
