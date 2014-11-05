import "compiler.liszt"
require 'tests/test'
--local types = terralib.require "compiler.types"


--------------------------
-- Kernel matrix tests: --
--------------------------
local LMesh = L.require "domains.lmesh"
local mesh = LMesh.Load("examples/mesh.lmesh")

mesh.vertices:NewField('tensor_pos', L.mat3d)
mesh.vertices:NewField('posmag', L.double)

------------------
-- Should pass: --
------------------
local populate_tensor = liszt kernel (v : mesh.vertices)
  var p = v.position
  var tensor = {
    { p[0]*p[0], p[0]*p[1], p[0]*p[2] },
    { p[1]*p[0], p[1]*p[1], p[1]*p[2] },
    { p[2]*p[0], p[2]*p[1], p[2]*p[2] }
  }
  v.tensor_pos = tensor
end
populate_tensor(mesh.vertices)

local trace = liszt kernel (v : mesh.vertices)
  var tp = v.tensor_pos
  v.posmag = tp[0,0]*tp[0,0] + tp[1,1]*tp[1,1] + tp[2,2]*tp[2,2]
end
trace(mesh.vertices)

local k = liszt kernel (v : mesh.vertices)
  var x       = { { 5, 5, 5 }, { 4, 4, 4 }, { 5, 5, 5 } }
  v.tensor_pos += x + { { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 } }
end
k(mesh.vertices)

local zero_corner = liszt kernel(v : mesh.vertices)
  v.tensor_pos[0,2] = 0
end
zero_corner(mesh.vertices)

local arith_test = liszt kernel(v : mesh.vertices)
  var id : L.mat3d = {{1,0,0},{0,1,0},{0,0,1}}
  var A  : L.mat3d = {{2,2,2},{3,3,3},{4,4,4}}
  var B  : L.mat3d = {{7,2,2},{3,8,3},{4,4,9}}
  L.assert((A + 5*id) == B)
end
arith_test(mesh.vertices)

------------------
-- Should fail: --
------------------

test.fail_function(function()
  liszt kernel t(v : mesh.vertices)
    var m34 = L.mat3x4f({{1.1,0,2.3},{0.1,0,0},{0,5.2,0}})
    var m33 = L.mat3f(m34)
  end
end,
'Cannot cast between primitives, vectors, matrices of different dimensions')

test.fail_function(function()
  liszt kernel t(v : mesh.vertices)
    var x = {{1, 2, 3},{true,false,true},{1,2,5}}
  end
end, "must be of the same type")

test.fail_function(function()
  liszt kernel t(v : mesh.vertices)
    var x = {{1,2}, {2,3}, {2,3,4}}
  end
end, "matrix literals must contain vectors of the same size")

idx = 3.5
test.fail_function(function()
  liszt kernel t(v : mesh.vertices)
      v.tensor_pos[idx,2] = 5
  end
end, "expected an integer")

test.fail_function(function()
  liszt kernel t(v : mesh.vertices)
      v.tensor_pos[0] = 5
  end
end, "expected vector to index into, not SmallMatrix")

-- Parse error, so not safe to test this way
--test.fail_function(function()
--  liszt kernel(v : mesh.vertices)
--      v.tensor_pos[0,0,1] = 5
--  end
--end, "expected 2 indices")




