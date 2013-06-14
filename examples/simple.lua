require "include/liszt"

mesh = LoadMesh("simple.obj")
pos  = mesh.fieldWithLabel(Vertex, Vector.type(3, Float), "position")