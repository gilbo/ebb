import "compiler.liszt"
local R = L.NewRelation { name="R", size=5 }

local liszt fail_assert (r : R)
    L.assert(2 + 2 == 5)
end
R:foreach(fail_assert)

