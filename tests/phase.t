import "compiler.liszt"
local LMesh = L.require "domains.lmesh"
local M     = LMesh.Load("examples/mesh.lmesh")
local test  = require "tests/test"

M.vertices:NewField('field1', L.float):LoadConstant(0)
M.vertices:NewField('field2', L.float):LoadConstant(0)

test.fail_function(function()
  local kernel = liszt_kernel (v : M.vertices)
    for nv in v.vertices do
      nv.field1 = 3
    end
  end
end, 'Non%-Exclusive WRITE')

test.fail_function(function()
  local kernel = liszt_kernel (v : M.vertices)
    v.field1 = 3
    var sum : L.float = 0
    for nv in v.vertices do
      sum += nv.field1
    end
  end
end, 'READ Phase is incompatible with.* EXCLUSIVE Phase')

test.fail_function(function()
  local kernel = liszt_kernel (v : M.vertices)
    var sum : L.float = 0
    for nv in v.vertices do
      nv.field1 += 1
      sum += nv.field1
    end
  end
end, 'READ Phase is incompatible with.* REDUCE%(%+%) Phase')

test.fail_function(function()
  local kernel = liszt_kernel (v : M.vertices)
    var sum : L.float = 0
    for nv in v.vertices do
      nv.field1 += 1
      nv.field1 *= 2
    end
  end
end, 'REDUCE%(%*%) Phase is incompatible with.* REDUCE%(%+%) Phase')

-- writing and reducing exclusively should be fine
local kernel = liszt_kernel (v : M.vertices)
  v.field1 = 3
  v.field1 += 1
end


-- two different reductions exclusively should be fine
local kernel = liszt kernel (v : M.vertices)
  v.field1 += 2
  v.field1 *= 2
end




local g1 = L.NewGlobal(L.float, 32)

test.fail_function(function()
  local kernel = liszt kernel (v : M.vertices)
    g1 = v.field1
  end
end, 'cannot write to globals inside kernels')

test.fail_function(function()
  local kernel = liszt kernel (v : M.vertices)
    var x = g1
    g1 += 1
  end
end, 'cannot read and reduce a global at the same time')

test.fail_function(function()
  local kernel = liszt kernel (v : M.vertices)
    g1 += v.field1
    g1 *= v.field1
  end
end, 'cannot reduce a global in two different ways')




