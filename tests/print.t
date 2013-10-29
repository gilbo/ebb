import "compiler/liszt"

mesh  = LoadMesh("examples/mesh.lmesh")

local v = Vector.new(L.float, {1, 2, 3}) 
local print_stuff = liszt_kernel(f)
    print(true)
    print(4)
    print(2.2)
    var x = 2 + 3
    print(x)
    print(v)
end

mesh.faces:map(print_stuff)
