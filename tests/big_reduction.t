
import 'compiler.liszt'

local N = 1000000

local vertices = L.NewRelation { size = N, name = 'vertices' }
local gerr = L.Global(L.int, 0)

local liszt RunRed(v : vertices)
  gerr += 1
end

function run_test()
	gerr:set(0)
	vertices:foreach(RunRed)
	L.assert(N == gerr:get())
end

run_test()