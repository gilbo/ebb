import "compiler/liszt"

mesh = L.initMeshRelationsFromFile("examples/mesh.lmesh")

local assert = L.assert

square = L.NewMacro(function(x)
    return liszt `x*x
end)

local test_macro = liszt kernel(v in mesh.vertices)
	assert(square(7) == 49)
end
test_macro()

