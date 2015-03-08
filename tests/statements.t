import "compiler.liszt"
require "tests/test"

local LMesh = L.require "domains.lmesh"
local mesh = LMesh.Load("examples/mesh.lmesh")
mesh.vertices:NewField('val', L.float)
mesh.vertices.val:LoadConstant(1)
red = L.Global(L.float, 0.0)

-- checking decl statement, if statement, proper scoping
local l = liszt (v : mesh.vertices)
  var y : L.float
  if v.val == 1.0 then
    y = 1.0
  else
    y = 0.0
  end

  red += y
end
mesh.vertices:map(l)

test.eq(red:get(), mesh.vertices:Size())
