import "compiler.liszt"
require "tests/test"



local triangles = L.NewRelation { size = 1, name = 'triangles' }
local vertices  = L.NewRelation { size = 3, name = 'vertices' }

triangles:NewField('v0', vertices)
triangles:NewField('v1', vertices)
triangles:NewField('v2', vertices)

vertices:NewField('pos', L.vector(L.float, 4))
vertices:NewField('color', L.vector(L.float, 3))

test.eq(triangles:StructuralType():toString(),
  'Record({ v0=Row(vertices), v1=Row(vertices), v2=Row(vertices) })')
test.eq(vertices:StructuralType():toString(),
  'Record({ color=Vector(float,3), pos=Vector(float,4) })')

