import "compiler.liszt"
require "tests/test"

local LMesh = L.require "domains.lmesh"
local mesh = LMesh.Load("examples/mesh.lmesh")

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

-- test vector codegen:
local f1 = liszt (f : mesh.faces) sf3 +=   vf    end
local f2 = liszt (f : mesh.faces) si4 -=   vi    end
local f3 = liszt (f : mesh.faces) sb5 and= vb    end
local f4 = liszt (f : mesh.faces) sf  +=   1     end
local f5 = liszt (f : mesh.faces) si  -=   two   end
local f6 = liszt (f : mesh.faces) sb  and= false end
--local f7 = liszt (f : mesh.faces) sd  /=   2.0   end

mesh.faces:map(f1)
test.fuzzy_aeq(sf3:get(), {nf, 2*nf, 3*nf})

mesh.faces:map(f2)
test.fuzzy_aeq(si4:get(), {1-2*nf,2-2*nf,3-2*nf,4-2*nf})

mesh.faces:map(f3)
test.aeq(sb5:get(), {true, false, true, false, true})

mesh.faces:map(f4)
test.eq(sf:get(), mesh.faces:Size())

mesh.faces:map(f5)
test.eq(si:get(), -2*nf)

mesh.faces:map(f6)
test.eq(sb:get(), false)

--mesh.faces:map(f7)
--test.eq(sd:get(), math.pow(2,-nf))
