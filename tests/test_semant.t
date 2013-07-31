import "compiler/liszt"
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
        checkthis1 = 5
        checkthis2 = 1
		var local1 = 9.0
		var local2 = 2.0
		var local3 = local1 + local2
--		var local4 = checkthis1 + checkthis2
		global1 = 2.0
		var local5 = true
		var local6 = local5 and false
		var local7 = 8 <= 9

		while true
		do
			var local8 = 2
		end

		while local7 == local7
		do
			local8 = 2.0
			local7 = 2.0
			local7 = false
		end

		if 4 < 2 then
			var local8 = true
		elseif local8 then
			var local9 = true
		elseif 4 < 3 then
			var local9 = 2
		else
			var local10 = local7
		end

		do
			var local1 = true
			local1 = 2.0
		end

		local1

    end

print("Completed main")

end

main()
