import "compiler/liszt"
require "tests/test"
local LMesh = terralib.require("compiler/liblmesh")
local mesh = LMesh.Load("examples/mesh.lmesh")

nf = mesh.faces._size
sf = L.NewScalar(L.float, 0.0)
si = L.NewScalar(L.int,     0)
sb = L.NewScalar(L.bool, true)

sf3 = L.NewScalar(L.vector(L.float, 3), {0, 0, 0})
si4 = L.NewScalar(L.vector(L.int,   4), {1, 2, 3, 4})
--sb5 = L.NewScalar(L.vector(L.bool,  5), {true, true, true, true, true})

vf  = L.NewVector(L.float, {1, 2, 3})
vi  = L.NewVector(L.int,   {2, 2, 2, 2})
--vb  = L.NewVector(L.bool,  {true, false, true, false, true})

local two = 2

-- test vector codegen:
local f1 = liszt_kernel (f in mesh.faces) sf3 +=   vf    end
local f2 = liszt_kernel (f in mesh.faces) si4 -=   vi    end
--local f3 = liszt_kernel (f in mesh.faces) sb5 and= vb    end
local f4 = liszt_kernel (f in mesh.faces) sf  +=   1     end
local f5 = liszt_kernel (f in mesh.faces) si  -=   two   end
local f6 = liszt_kernel (f in mesh.faces) sb  and= false end

f1()
test.fuzzy_aeq(sf3:value().data, {nf, 2*nf, 3*nf})

f2()
test.fuzzy_aeq(si4:value().data, {1-2*nf,2-2*nf,3-2*nf,4-2*nf})

--f3()
--test.aeq(sb5:value().data, {true, false, true, false, true})

f4()
test.eq(sf:value(), mesh.faces._size)

f5()
test.eq(si:value(), -2*nf)

f6()
test.eq(sb:value(), false)
