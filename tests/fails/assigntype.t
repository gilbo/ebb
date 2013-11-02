import "compiler/liszt"

mesh = L.initMeshRelationsFromFile("examples/mesh.lmesh")
sf   = L.NewScalar(L.float, 0.0)

local k = liszt_kernel (c in mesh.cells)
	sf.a = 1
end

k()
