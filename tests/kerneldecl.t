import "compiler.liszt"

local LMesh = terralib.require "compiler.lmesh"
local mesh = LMesh.Load("examples/mesh.lmesh")

local faces = mesh.faces

-- The identity kernel:
local pass_kernel = liszt_kernel(f in faces) end
pass_kernel()
