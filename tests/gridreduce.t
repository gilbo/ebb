import "ebb"
require "tests/test"


local rel1 = L.NewRelation { name="rel1", size = 5 }
local rel2 = L.NewRelation { name="rel2", dims = {2,3} }
local rel3 = L.NewRelation { name="rel3", dims = {3,2,2} }

rel1:NewField('ones', L.int):Load(1)
rel2:NewField('ones', L.int):Load(1)
rel3:NewField('ones', L.int):Load(1)

local glob_one_count_1 = L.Global(L.int, 0)
local glob_one_count_2 = L.Global(L.int, 0)
local glob_one_count_3 = L.Global(L.int, 0)

local glob_raw_count_1 = L.Global(L.int, 0)
local glob_raw_count_2 = L.Global(L.int, 0)
local glob_raw_count_3 = L.Global(L.int, 0)

local ebb count_one1 ( c : rel1 )
  L.assert(c.ones == 1)
  glob_one_count_1 += c.ones
end
local ebb count_one2 ( c : rel2 )
  L.assert(c.ones == 1)
  glob_one_count_2 += c.ones
end
local ebb count_one3 ( c : rel3 )
  L.assert(c.ones == 1)
  glob_one_count_3 += c.ones
end

local ebb count_raw1 ( c : rel1 )
  glob_raw_count_1 += 1
end
local ebb count_raw2 ( c : rel2 )
  glob_raw_count_2 += 1
end
local ebb count_raw3 ( c : rel3 )
  glob_raw_count_3 += 1
end

rel1:foreach(count_one1)
rel2:foreach(count_one2)
rel3:foreach(count_one3)

rel1:foreach(count_raw1)
rel2:foreach(count_raw2)
rel3:foreach(count_raw3)

test.eq(rel1:Size(), glob_one_count_1:get())
test.eq(rel1:Size(), glob_raw_count_1:get())
test.eq(rel2:Size(), glob_one_count_2:get())
test.eq(rel2:Size(), glob_raw_count_2:get())
test.eq(rel3:Size(), glob_one_count_3:get())
test.eq(rel3:Size(), glob_raw_count_3:get())

