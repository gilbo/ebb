import "compiler/liszt"

mesh = LoadMesh("examples/mesh.lmesh")
sf = mesh:scalar(L.float, 0.0)

local k = liszt_kernel (c)
	sf.a = 1
end
mesh.cells:map(k)
