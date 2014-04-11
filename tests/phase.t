import "compiler.liszt"
local LMesh = L.require "domains.lmesh"
local M     = LMesh.Load("examples/mesh.lmesh")
local test  = require "tests/test"

M.vertices:NewField('field1', L.float):LoadConstant(0)
M.vertices:NewField('field2', L.float):LoadConstant(0)

test.fail_function(function()
  local kernel = liszt_kernel (v : M.vertices)
    v.field1 = 1.3
    v.field2 = v.field1
  end
end, '<Read> phase conflicts with earlier access in <Write> phase')


test.fail_function(function()
  local kernel = liszt_kernel (v : M.vertices)
  v.field1 = 1.3
  v.field1 += 1
  end
end, 'field in <Additive Reduction> phase conflicts with earlier access in <Write> phase')


test.fail_function(function()
  local kernel = liszt_kernel (v : M.vertices)
  var x = v.field1
  v.field1 += 3
  end
end, 'field in <Additive Reduction> phase conflicts with earlier access in <Read> phase')


test.fail_function(function()
  local kernel = liszt_kernel (v : M.vertices)
  v.field1 += 3
  v.field1 *= 2
  end
end, 'field in <Multiplicative Reduction> phase conflicts with earlier access in <Additive Reduction> phase')