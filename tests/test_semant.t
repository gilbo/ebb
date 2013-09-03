import "compiler/liszt"
mesh = LoadMesh("examples/mesh.lmesh")
f1   = mesh:field(Cell, float, 0)
f2   = mesh:field(Vertex, Vector.type(float, 3), 0)
s1   = mesh:scalar(int)
checkthis1 = 1

gb = global(float, 3)
gb1 = global(int, 4)
function main ()
    -- init statements, binary expressions, 
    -- field lookups (lvalues), function calls (multiple arguments)

    local checkthis2 = 2

    local k = liszt_kernel (cell)
    	-- this should fail - checkthis1 is not a scalar
        checkthis1 = f1(cell)

        var x = cell -- this should pass
        x = cell     -- ...but not this, since we don't allow re-assignments to variables references topo types

        --s1 = gb1

        -- this should fail, since we're trying to assign to a field directly (and since the types don't match)
        f1 = 5

        -- this should fail, because we don't allow local temporaries to refer to fields, scalars, or other stencil-complicating entities
        var z = f1

        -- this should pass
        f1(cell) = gb

        -- this should pass
        var lc = gb
        lc = gb + 1

        -- this should fail, because checkthis2 is not declared in the global scope.
        checkthis2 = 1

		var local1 = 9.0
		var local2 = 2.0
		var local3 = local1 + local2 -- ints should upcast to floats
		var local5 = 2 + 3.3
		var local4 = checkthis1 + checkthis2
		global1    = 2.0 -- fail, global1 not defined
		var local5 = true
		var local6 = local5 and false
		var local7 = 8 <= 9

		3 + 4

		while true
		do
			var local8 = 2
		end

		while local7 ~= local7
		do
			local8 = 2.0 -- fail, local8 not defined in this scope
			local7 = 2.0
			local7 = false
		end

		if 4 < 2 then
			var local8 = true
		elseif local8 then -- should fail - local8 not defined in this scope
			var local9 = true
		elseif 4 < 3 then
			var local9 = 2
		else
			var local10 = local7
		end

		do
			var local1 = true
			local1 = 2.0 -- should fail, local1 is of type bool
		end

		var local9 = 0
		for i = 1, 4, 1 do
			local9 = local9 + i * i
		end

		local1

    end
end

main()
