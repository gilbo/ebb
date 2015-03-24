import "compiler.liszt"

local R = L.NewRelation { name="R", size=5 }

local liszt fail_assert (r : R)
    var x = 5
    L.assert(x == 4)
end

R:map(fail_assert)
