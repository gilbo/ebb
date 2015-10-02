import "ebb"
require "tests/test"

local R = L.NewRelation { name="R", size=6 }

R:NewField('val', L.float)
R.val:LoadConstant(1)
red = L.Global(L.float, 0.0)

-- checking decl statement, if statement, proper scoping
local l = ebb (v : R)
  var y : L.float
  if v.val == 1.0f then
    y = 1.0f
  else
    y = 0.0f
  end

  red += y
end
R:foreach(l)

test.eq(red:get(), R:Size())
