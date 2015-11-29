import 'ebb'
local L = require 'ebblib'

local R = L.NewRelation { name="R", size=5 }

local pass_assert = ebb(r : R)
    L.assert(true)
end
R:foreach(pass_assert)
