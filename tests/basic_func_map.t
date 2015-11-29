import 'ebb'
local L = require 'ebblib'


local R = L.NewRelation { name="R", size=5 }

-- The identity function:
local pass_func = ebb(r : R) end
R:foreach(pass_func)
