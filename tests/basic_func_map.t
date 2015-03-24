import "compiler.liszt"


local R = L.NewRelation { name="R", size=5 }

-- The identity function:
local pass_func = liszt(r : R) end
R:map(pass_func)
