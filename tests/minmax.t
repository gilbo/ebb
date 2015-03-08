import "compiler.liszt"
require 'tests.test'

local LMesh = L.require "domains.lmesh"
local M = LMesh.Load("examples/mesh.lmesh")

local V = M.vertices
local P = V.position

local max_pos = L.Global(L.vec3i, {-10, -10, -10})
local min_pos = L.Global(L.vec3i, { 10,  10,  10})

-- Test max reduction operator
local max_func = liszt (v : M.vertices)
	max_pos max= L.vec3i(v.position)
end
M.vertices:map(max_func)
test.aeq(max_pos:get(), {1,1,1})

-- Test min reduction operator
local min_func = liszt (v : M.vertices)
	min_pos min= L.vec3i(v.position)
end
M.vertices:map(min_func)
test.aeq(min_pos:get(), {-1, -1, -1})
