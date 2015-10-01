import "ebb.liszt"
require 'tests.test'

local ioOff = require 'ebb.domains.ioOff'
local M     = ioOff.LoadTrimesh('tests/octa.off')

local V = M.vertices
local P = V.pos

local max_pos = L.Global(L.vec3d, {-10, -10, -10})
local min_pos = L.Global(L.vec3d, { 10,  10,  10})

-- Test max reduction operator
local max_func = liszt (v : M.vertices)
	max_pos max= v.pos
end
M.vertices:foreach(max_func)
test.aeq(max_pos:get(), {0.5,0.5,0.5})

-- Test min reduction operator
local min_func = liszt (v : M.vertices)
	min_pos min= v.pos
end
M.vertices:foreach(min_func)
test.aeq(min_pos:get(), {-0.5, -0.5, -0.5})
