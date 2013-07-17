-- require "include/liszt"

function main ()

	print("In main")
	-- init statements, binary expressions, 
	-- field lookups (lvalues), function calls (multiple arguments)
	local b = liszt_kernel (cell) 
		var a = Vector.new(Int, 1, 2)
		var b = 3 + 4 * 5
		var c = 6 / 7 + -8 ^ 9
		field(cell) = b * a
	end

	local c = liszt_kernel (cell)
		var x = a.b + c(d)
	end

	-- if statement, field index, init statements
	local d = liszt_kernel (cell)
		if true then
			var a = field(cell)
		else
			var a = field2(cell)
		end
	end

	-- table lookup and field index:
	local e = liszt_kernel (cell)
		table.field(cell) = x*(5-3)
	end

	-- break statement
	local f = liszt_kernel (cell)
		table.field(cell) = x*(5-3)
		var x = y
		break
	end

end

main()
