import "compiler.liszt"
require 'tests/test'
--local types = require "compiler.types"


local LMesh = L.require "domains.lmesh"
local mesh = LMesh.Load("examples/mesh.lmesh")

mesh.vertices:NewField('tensor_pos', L.mat3d)
mesh.vertices:NewField('posmag', L.double)

------------------
-- Should pass: --
------------------
local populate_tensor = liszt (v : mesh.vertices)
  var p = v.position
  var tensor = {
    { p[0]*p[0], p[0]*p[1], p[0]*p[2] },
    { p[1]*p[0], p[1]*p[1], p[1]*p[2] },
    { p[2]*p[0], p[2]*p[1], p[2]*p[2] }
  }
  v.tensor_pos = tensor
end
mesh.vertices:map(populate_tensor)

local trace = liszt (v : mesh.vertices)
  var tp = v.tensor_pos
  v.posmag = tp[0,0]*tp[0,0] + tp[1,1]*tp[1,1] + tp[2,2]*tp[2,2]
end
mesh.vertices:map(trace)

local k = liszt (v : mesh.vertices)
  var x       = { { 5, 5, 5 }, { 4, 4, 4 }, { 5, 5, 5 } }
  v.tensor_pos += x + { { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 } }
end
mesh.vertices:map(k)

local zero_corner = liszt(v : mesh.vertices)
  v.tensor_pos[0,2] = 0
end
mesh.vertices:map(zero_corner)

local arith_test = liszt(v : mesh.vertices)
  var id : L.mat3d = {{1,0,0},{0,1,0},{0,0,1}}
  var A  : L.mat3d = {{2,2,2},{3,3,3},{4,4,4}}
  var B  : L.mat3d = {{7,2,2},{3,8,3},{4,4,9}}
  L.assert((A + 5*id) == B)
end
mesh.vertices:map(arith_test)

local unsym_mat = liszt(v : mesh.vertices)
  var A : L.mat3x4i = { { 1, 2, 3, 4 }, { 5, 6, 7, 8 }, { 9, 10, 11, 12 } }
  A
end
mesh.vertices:map(unsym_mat)

------------------
-- Should fail: --
------------------

test.fail_function(function()
  local liszt t(v : mesh.vertices)
    var m34 = L.mat3x4f({{1.1,0,2.3},{0.1,0,0},{0,5.2,0}})
    var m33 = L.mat3f(m34)
  end
  mesh.vertices:map(t)
end,
'Cannot cast between primitives, vectors, matrices of different dimensions')

test.fail_function(function()
  local liszt t(v : mesh.vertices)
    var x = {{1, 2, 3},{true,false,true},{1,2,5}}
  end
  mesh.vertices:map(t)
end, "must be of the same type")

test.fail_function(function()
  local liszt t(v : mesh.vertices)
    var x = {{1,2}, {2,3}, {2,3,4}}
  end
  mesh.vertices:map(t)
end, "matrix literals must contain vectors of the same size")

idx = 3.5
test.fail_function(function()
  local liszt t(v : mesh.vertices)
      v.tensor_pos[idx,2] = 5
  end
  mesh.vertices:map(t)
end, "expected an integer")

test.fail_function(function()
  local liszt t(v : mesh.vertices)
      v.tensor_pos[0] = 5
  end
  mesh.vertices:map(t)
end, "expected vector to index into, not Matrix")

-- Parse error, so not safe to test this way
--test.fail_function(function()
--  local liszt t(v : mesh.vertices)
--      v.tensor_pos[0,0,1] = 5
--  end
--  mesh.vertices:map(t)
--end, "expected 2 indices")




