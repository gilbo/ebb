--GPU-TEST
if not terralib.cudacompile then return end

import 'compiler.liszt'
L.default_processor = L.GPU

local N = 1000000

local vertices = L.NewRelation(N, 'vertices')
local gerr = L.NewGlobal(L.int, 0)

local liszt kernel RunRed(v : vertices)
  gerr += 1
end

function run_test()
	gerr:set(0)
	RunRed(vertices)
	L.assert(N == gerr:get())
end

run_test()