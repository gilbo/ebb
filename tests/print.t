import "compiler.liszt"
local LMesh = L.require "domains.lmesh"

local mesh = LMesh.Load("examples/mesh.lmesh")

local v = L.Constant(L.vec3f, {1, 2, 3}) 

-- We break each print statement into its own liszt function,
-- since otherwise the order of the print statements is
-- technically non-deterministic under Liszt's execution semantics

local liszt print_1 ( f : mesh.faces )
    L.print(true)
end
local liszt print_2 ( f : mesh.faces )
    var m = { { 1.2, 0 }, { 0.4, 1 } }
    L.print(m)
end
local liszt print_3 ( f : mesh.faces )
    L.print(4)
end
local liszt print_4 ( f : mesh.faces )
    L.print(2.2)
end
local liszt print_5 ( f : mesh.faces )
    L.print()
end
local liszt print_6 ( f : mesh.faces )
    L.print(1,2,3,4,5,false,{3.3,3.3})
end
local liszt print_7 ( f : mesh.faces )
    var x = 2 + 3
    L.print(x)
end
local liszt print_8 ( f : mesh.faces )
    L.print(v)
end
-- cannot rely on order of execution
--local print_stuff = liszt(f : mesh.faces)
--    L.print(L.id(f))
--end

mesh.faces:map(print_1)
mesh.faces:map(print_2)
mesh.faces:map(print_3)
mesh.faces:map(print_4)
mesh.faces:map(print_5)
mesh.faces:map(print_6)
mesh.faces:map(print_7)
mesh.faces:map(print_8)
