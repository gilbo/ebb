import "compiler.liszt"
local LMesh = terralib.require "compiler.lmesh"
local mesh = LMesh.Load("examples/mesh.lmesh")

local vk = liszt_kernel(v in mesh.vertices)
    var v = { }
end
vk(mesh.vertices)
