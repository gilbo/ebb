import "compiler.liszt"

local R = L.NewRelation { name="R", size=6 }

local v = L.Constant(L.vec3f, {1, 2, 3}) 

-- We break each print statement into its own liszt function,
-- since otherwise the order of the print statements is
-- technically non-deterministic under Liszt's execution semantics

local liszt print_1 ( r : R )
    L.print(true)
end
local liszt print_2 ( r : R )
    var m = { { 1.2, 0 }, { 0.4, 1 } }
    L.print(m)
end
local liszt print_3 ( r : R )
    L.print(4)
end
local liszt print_4 ( r : R )
    L.print(2.2)
end
local liszt print_5 ( r : R )
    L.print()
end
local liszt print_6 ( r : R )
    L.print(1,2,3,4,5,false,{3.3,3.3})
end
local liszt print_7 ( r : R )
    var x = 2 + 3
    L.print(x)
end
local liszt print_8 ( r : R )
    L.print(v)
end
-- cannot rely on order of execution
--local print_stuff = liszt(r : R)
--    L.print(L.id(f))
--end

R:map(print_1)
R:map(print_2)
R:map(print_3)
R:map(print_4)
R:map(print_5)
R:map(print_6)
R:map(print_7)
R:map(print_8)
