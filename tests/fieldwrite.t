import 'ebb'
local L = require 'ebblib'

local R = L.NewRelation { name="R", size=5 }
R:NewField('position', L.vec3d):Load({0,0,0})

function shift(x,y,z)
  local ebb pass_func (r : R)
      r.position = {x,y,z}
  end
  R:foreach(pass_func)
end

function check(x,y,z)
  R.position:Dump(function(pos, i)
    assert(pos[1] == x)
    assert(pos[2] == y)
    assert(pos[3] == z)
  end)
end

shift(0,0,0)
check(0,0,0)

shift(5,5,5)
check(5,5,5)

shift(-1,6,3)
check(-1,6,3)
