import "compiler.liszt"
local LMesh = L.require "domains.lmesh"

local mesh = LMesh.Load("examples/mesh.lmesh")

local v = L.Constant(L.vec3f, {1, 2, 3}) 

local print_stuff = liszt(f : mesh.faces)
    var m = { { 1.2, 0 }, { 0.4, 1 } }
    L.print(true)
    L.print(m)
    L.print(4)
    L.print(2.2)
    L.print()
    L.print(1,2,3,4,5,false,{3.3,3.3})
    var x = 2 + 3
    L.print(x)
    L.print(v)
    L.print(L.id(f))
end

mesh.faces:map(print_stuff)
