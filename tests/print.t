import "compiler.liszt"
local LMesh = terralib.require "compiler.lmesh"
local mesh = LMesh.Load("examples/mesh.lmesh")

local v = L.NewVector(L.float, {1, 2, 3}) 

local print_stuff = liszt_kernel(f : mesh.faces)
    L.print(true)
    L.print(4)
    L.print(2.2)
    var x = 2 + 3
    L.print(x)
    L.print(v)
    L.print(L.id(f))
end

print_stuff(mesh.faces)
