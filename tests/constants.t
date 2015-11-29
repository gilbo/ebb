--DISABLE-TEST
import 'ebb'
local L = require 'ebblib'
require "tests/test"

local R = L.NewRelation { name="R", size=5 }

--[[
nf = mesh.faces:Size()
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

]]--