require "include/liszt"
mesh = LoadMesh("examples/mesh.lmesh")

f1   = mesh:field(Cell,   int, 0)

print("Defining checkthis1")
checkthis1 = 1

function main ()
    -- init statements, binary expressions, 
    -- field lookups (lvalues), function calls (multiple arguments)

    print("Beginning to execute main")

    print("Defining checkthis2")
    local checkthis2 = 2

	print("f1 is a field of", f1.data_type, "over", f1.topo_type)

    local k = liszt_kernel (cell)
        checkthis1 = 0.5
        checkthis2 = 4
        position = 2
        var a = Vector.new(Int, 1, 2)
    end

print("Completed main")

end

main()
