import "ebb.liszt"
require "tests.test"


local R = L.NewRelation { name="R", size=5 }

test.fail_function(function()
  local liszt t(r : R)
    var x = {1, 2, 3} + 4
  end
  R:foreach(t)
end, "incompatible types")

test.fail_function(function()
  local liszt t(r : R)
    var x = {1, 2, 3} / {4, 5, 6}
  end
  R:foreach(t)
end, "invalid types")

test.fail_function(function()
  local liszt t(r : R)
    var x = 5 < true
  end
  R:foreach(t)
end, "invalid types")

local t = {}
local s = {}
test.fail_function(function()
  local liszt t(r : R)
    var x = s < t
  end
  R:foreach(t)
end, "invalid types")
