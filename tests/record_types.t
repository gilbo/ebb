import "compiler.liszt"
require "tests/test"
--local LMesh = terralib.require "compiler.lmesh"
--local mesh = LMesh.Load("examples/mesh.lmesh")




local triangles = L.NewRelation(1, 'triangles')
local vertices  = L.NewRelation(3, 'vertices')

triangles:NewField('v0', vertices)
triangles:NewField('v1', vertices)
triangles:NewField('v2', vertices)

vertices:NewField('pos', L.vector(L.float, 4))
vertices:NewField('color', L.vector(L.float, 3))

test.eq(triangles:StructuralType():toString(),
  'Record({ v0=Row(vertices), v1=Row(vertices), v2=Row(vertices) })')
test.eq(vertices:StructuralType():toString(),
  'Record({ color=Vector(float,3), pos=Vector(float,4) })')

