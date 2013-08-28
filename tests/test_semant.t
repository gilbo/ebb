import "compiler/liszt"
mesh = LoadMesh("examples/mesh.lmesh")
f1   = mesh:field(Cell,   int, 0)
f2   = mesh:field(Vertex, Vector.type(float, 3), 0)
checkthis1 = 1

gb = global(int, 3)

function main ()
    -- init statements, binary expressions, 
    -- field lookups (lvalues), function calls (multiple arguments)

    local checkthis2 = 2

    local k = liszt_kernel (cell)
        checkthis1 = f1(cell)
        cell -- cell's type should be inferred as cell now
        gb -- should be able to understand global terra variables (a shortcut to allow users to specify types for numbers used in liszt kernels)
        f2
        checkthis2 = 1
		var local1 = 9.0
		var local2 = 2.0
		var local3 = local1 + local2 -- ints should upcast to floats
		var local5 = 2 + 3.3
--		var local4 = checkthis1 + checkthis2
		global1    = 2.0
		var local5 = true
		var local6 = local5 and false
		var local7 = 8 <= 9

		3 + 4

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

		var local9 = 0
		for i = 1, 4, 1 do
			local9 = local9 + i * i
		end

		local1

    end
	print("Completed main")
end

main()
