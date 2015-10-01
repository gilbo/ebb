import "ebb.liszt"
require "tests/test"

local R = L.NewRelation { name="R", size=5 }

sf   = L.Global(L.float, 0.0)

test.fail_function(function()
  local liszt test (r : R)
    sf.a = 1
  end
  R:foreach(test)
end, "select operator not supported")
