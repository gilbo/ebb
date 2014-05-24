import "compiler.liszt"
require 'tests.test'

local LMesh = L.require "domains.lmesh"
local M = LMesh.Load("examples/mesh.lmesh")

local V = M.vertices
local P = V.position

local max_pos = L.NewGlobal(L.vector(L.double, 3), {-10, -10, -10})
local min_pos = L.NewGlobal(L.vector(L.double, 3), { 10,  10,  10})

-- Test max reduction operator
local max_kernel = liszt kernel (v : M.vertices)
	max_pos max= v.position
end
max_kernel(M.vertices)
test.aeq(max_pos:get().data, {1,1,1})

-- Test min reduction operator
local min_kernel = liszt kernel (v : M.vertices)
	min_pos min= v.position
end
min_kernel(M.vertices)
test.aeq(min_pos:get().data, {-1, -1, -1})
