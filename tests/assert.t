import "ebb.liszt"

local R = L.NewRelation { name="R", size=5 }

local pass_assert = liszt(r : R)
    L.assert(true)
end
R:foreach(pass_assert)
