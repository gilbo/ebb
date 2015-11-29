import 'ebb'
local L = require 'ebblib'
local R = L.NewRelation { name="R", size=5 }

local ebb fail_assert (r : R)
    L.assert(2 + 2 == 5)
end
R:foreach(fail_assert)

