import "compiler/liszt"
require "tests/test"

mesh  = LoadMesh("examples/mesh.lmesh")

nf = mesh.faces:size()
sf = mesh:scalar(L.float, 0.0)
si = mesh:scalar(L.int,     0)
sb = mesh:scalar(L.bool, true)

sf3 = mesh:scalar(L.vector(L.float, 3), {0, 0, 0})
si4 = mesh:scalar(L.vector(L.int,   4), {1, 2, 3, 4})
sb5 = mesh:scalar(L.vector(L.bool,  5), {true, true, true, true, true})

vf  = Vector.new(L.float, {1, 2, 3})
vi  = Vector.new(L.int,   {2, 2, 2, 2})
vb  = Vector.new(L.bool,  {true, true, true, true, true})

two = global(int)
terra set_two()
	two = 2
end
set_two()

-- test vector codegen:
mesh.faces:map(liszt_kernel (f) sf3 = sf3 + vf end)
test.fuzzy_aeq(sf3:value().data, {nf, 2*nf, 3*nf})

mesh.faces:map(liszt_kernel (f) si4 = si4 - vi end)
test.fuzzy_aeq(si4:value().data, {1-2*nf,2-2*nf,3-2*nf,4-2*nf})

--mesh.faces:map(liszt_kernel(f) sb5 = sb5 and vb end)
--test.aeq(sb5:value().data, {true, false, true, false, true})

-- test simple type codegen:
mesh.faces:map(liszt_kernel (f) sf = sf + 1 end)
test.eq(sf:value(), mesh.faces:size())

mesh.faces:map(liszt_kernel (f) si = si - two end)
test.eq(si:value(), -2*nf)

mesh.faces:map(liszt_kernel(f) sb = sb and false end)
test.eq(sb:value(), false)


-- todo: test scalar reads