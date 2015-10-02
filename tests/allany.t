import "ebb"
require "tests/test"

local R = L.NewRelation { name="R", size=5 }

local ebb pass(r : R)
    L.assert(L.any({true}))
    L.assert(not L.any({false}))
    L.assert(L.any({true, true, true}))
    L.assert(L.any({false, true, false, false}))
    L.assert(not L.any({false, false, false, false}))
    L.assert(L.all({true}))
    L.assert(L.all({true, true, true}))
    L.assert(not L.all({true, false}))
end
R:foreach(pass)
