import "ebb"

local R = L.NewRelation { size = 4, name = 'relation' }
R:NewField("result", L. uint64)

-- This macro emits a side effect every time it is evaluated
local side_effect = L.NewMacro(function(x)
	return ebb quote
		L.print(L.id(x))
	in
		x
	end
end)

local test_macro = L.NewMacro(function (y)
	return ebb `L.id(y)+ L.id(y)
end)

local ebb test (r : R)
	--side effect should only be evaluated once!
	r.result = test_macro(side_effect(r))

end

R:foreach(test)