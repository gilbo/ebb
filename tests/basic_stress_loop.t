import "ebb"


local R = L.NewRelation { name="R", size=5 }

-- The identity function:
local ebb pass_func (r : R) end

for i=1,1000 do
  R:foreach(pass_func)
  --for j=1,10 do end
end

--pass_func:getCompileTime():print()
--pass_func:getExecutionTime():print()