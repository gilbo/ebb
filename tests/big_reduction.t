
import 'ebb.liszt'
require 'tests.test'

local N = 1000000

local vertices = L.NewRelation { size = N, name = 'vertices' }
local gerr = L.Global(L.int, 0)

local liszt RunRed(v : vertices)
  gerr += 1
end

function run_test()
  gerr:set(0)
  vertices:foreach(RunRed)
  test.eq(N, gerr:get())
end

run_test()