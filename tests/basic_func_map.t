import "compiler.liszt"

local LMesh = L.require "domains.lmesh"
local mesh = LMesh.Load("examples/mesh.lmesh")

local faces = mesh.faces

-- The identity function:
local pass_func = liszt(f : faces) end
faces:map(pass_func)
