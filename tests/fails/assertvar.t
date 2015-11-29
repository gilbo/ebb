import 'ebb'
local L = require 'ebblib'

local R = L.NewRelation { name="R", size=5 }

local ebb fail_assert (r : R)
    var x = 5
    L.assert(x == 4)
end

R:foreach(fail_assert)
