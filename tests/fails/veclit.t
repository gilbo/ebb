import "compiler.liszt"
local LMesh = L.require "domains.lmesh"
local mesh = LMesh.Load("examples/mesh.lmesh")

local liszt vk (v : mesh.vertices)
    var v = { }
end
mesh.vertices:map(vk)
