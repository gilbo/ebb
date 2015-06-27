
import 'compiler.liszt'
require 'tests.test'

local N = 10000
local T = 5000

local vertices = L.NewRelation { size = N, name = 'vertices' }
local gerr = L.Global(L.double, 0)

local liszt RunRed(v : vertices)
  gerr += 1
end

function run_test()
  gerr:set(0)
  for k=1,T do
    vertices:foreach(RunRed)
  end
  test.eq(N*T, gerr:get())
end

run_test()