import "compiler.liszt"
local LMesh = L.require "domains.lmesh"
local mesh = LMesh.Load("examples/mesh.lmesh")

local liszt kernel vk (v : mesh.vertices)
    var v = { }
end
vk(mesh.vertices)
