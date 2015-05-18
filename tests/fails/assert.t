import "compiler.liszt"

local R = L.NewRelation { name="R", size=5 }

local liszt fail_assert (r : R)
    L.assert(false)
end
R:foreach(fail_assert)
