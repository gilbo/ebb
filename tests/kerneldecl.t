import "compiler.liszt"

local LMesh = L.require "domains.lmesh"
local mesh = LMesh.Load("examples/mesh.lmesh")

local faces = mesh.faces

-- The identity kernel:
local pass_kernel = liszt kernel(f : faces) end
pass_kernel(faces)
