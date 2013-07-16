require "include/liszt"

mesh     = LoadMesh("examples/mesh.lmesh")
position = mesh:fieldWithLabel(Vertex, Vector.type(float,3), "position")
field    = mesh:field(Face, float, 0.0)


local terra main_unnested ( ) : { }
	var com : vector(float,3) = { 0.0, 0.0, 0.0 };
	return
end