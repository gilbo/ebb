require "include/liszt"
import "compiler/liszt"
mesh = LoadMesh("examples/mesh.lmesh")

f1   = mesh:field(Cell, int, 0)
f2 = mesh:fieldWithLabel(Vertex, Vector.type(float,3), "position")

v1 = Vector.new(int, 1, 2, 3) 
v2 = Vector.new(int, 2, 4, 6)
v3 = Vector.new(float, 1.0, 2.0, 3.0)
v4 = Vector.new(int, 1.0, 1.0)

t1 = {}

print("Defining checkthis1")
checkthis1 = 1

function main ()
    -- init statements, binary expressions, 
    -- field lookups (lvalues), function calls (multiple arguments)

    local checkthis2 = 2

    local k = liszt_kernel (cell)
		v1 = v1
		v1 = v1 + v2
		v1 = v2 + v3
		v1 = v1 + v4
		v1 = v1 + v2
		t1 = t1
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

        v1

        for cell in mesh.cells do
            f1(cell)
			f2(cell)
        end

        var local11 = cell
        var local12 = local11
        var local13 = mesh.cells + mesh.vertices

        var local14 = f1
        var local15 = f2

    end

end

main()
