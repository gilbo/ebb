import "compiler.liszt"
require "tests/test"
local LMesh = L.require "domains.lmesh"
local mesh = LMesh.Load("examples/mesh.lmesh")

nf = mesh.faces._size
sf = L.NewGlobal(L.float, 0.0)
si = L.NewGlobal(L.int,     0)
sb = L.NewGlobal(L.bool, true)

sf3 = L.NewGlobal(L.vector(L.float, 3), {0, 0, 0})
si4 = L.NewGlobal(L.vector(L.int,   4), {1, 2, 3, 4})
--sb5 = L.NewGlobal(L.vector(L.bool,  5), {true, true, true, true, true})

vf  = L.NewVector(L.float, {1, 2, 3})
vi  = L.NewVector(L.int,   {2, 2, 2, 2})
--vb  = L.NewVector(L.bool,  {true, false, true, false, true})

local two = 2

-- test vector codegen:
local f1 = liszt kernel (f : mesh.faces) sf3 +=   vf    end
local f2 = liszt kernel (f : mesh.faces) si4 -=   vi    end
--local f3 = liszt kernel (f : mesh.faces) sb5 and= vb    end
local f4 = liszt kernel (f : mesh.faces) sf  +=   1     end
local f5 = liszt kernel (f : mesh.faces) si  -=   two   end
local f6 = liszt kernel (f : mesh.faces) sb  and= false end

f1(mesh.faces)
test.fuzzy_aeq(sf3:value().data, {nf, 2*nf, 3*nf})

f2(mesh.faces)
test.fuzzy_aeq(si4:value().data, {1-2*nf,2-2*nf,3-2*nf,4-2*nf})

--f3(mesh.faces)
--test.aeq(sb5:value().data, {true, false, true, false, true})

f4(mesh.faces)
test.eq(sf:value(), mesh.faces._size)

f5(mesh.faces)
test.eq(si:value(), -2*nf)

f6(mesh.faces)
test.eq(sb:value(), false)
