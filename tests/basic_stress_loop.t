import "compiler.liszt"


local R = L.NewRelation { name="R", size=5 }

-- The identity function:
local liszt pass_func (r : R) end

for i=1,50 do
  R:foreach(pass_func)
  for j=1,100000 do end
end