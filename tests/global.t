import "compiler.liszt"
require "tests/test"

local R = L.NewRelation { name="R", size=5 }

nr = R:Size()
sf = L.Global(L.float, 0.0)
si = L.Global(L.int,     0)
sb = L.Global(L.bool, true)
sd = L.Global(L.double, 1.0)

sf3 = L.Global(L.vector(L.float, 3), {0, 0, 0})
si4 = L.Global(L.vector(L.int,   4), {1, 2, 3, 4})
sb5 = L.Global(L.vector(L.bool,  5), {true, true, true, true, true})

vf  = L.Constant(L.vec3f, {1, 2, 3})
vi  = L.Constant(L.vec4i,   {2, 2, 2, 2})
vb  = L.Constant(L.vector(L.bool, 5),  {true, false, true, false, true})

local two = 2

-- test vector codegen:
local f1 = liszt (r : R) sf3 +=   vf    end
local f2 = liszt (r : R) si4 +=   -vi   end
local f3 = liszt (r : R) sb5 and= vb    end
local f4 = liszt (r : R) sf  +=   1     end
local f5 = liszt (r : R) si  +=   -two  end
local f6 = liszt (r : R) sb  and= false end
--local f7 = liszt (r : R) sd  /=   2.0   end

R:map(f1)
test.fuzzy_aeq(sf3:get(), {nr, 2*nr, 3*nr})

R:map(f2)
test.fuzzy_aeq(si4:get(), {1-2*nr,2-2*nr,3-2*nr,4-2*nr})

R:map(f3)
test.aeq(sb5:get(), {true, false, true, false, true})

R:map(f4)
test.eq(sf:get(), nr)

R:map(f5)
test.eq(si:get(), -2*nr)

R:map(f6)
test.eq(sb:get(), false)

--R:map(f7)
--test.eq(sd:get(), math.pow(2,-nr))
