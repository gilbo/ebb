import "compiler/liszt"
local mesh = L.initMeshRelationsFromFile("examples/mesh.lmesh")
local faces = mesh.faces

-- The identity kernel:
local pass_kernel = liszt_kernel(f in faces) end
pass_kernel()
