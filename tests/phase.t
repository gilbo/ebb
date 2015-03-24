import "compiler.liszt"
local test  = require "tests/test"

local ioOff = L.require 'domains.ioOff'
local M     = ioOff.LoadTrimesh('tests/octa.off')

M.vertices:NewField('field1', L.float):LoadConstant(0)
M.vertices:NewField('field2', L.float):LoadConstant(0)

test.fail_function(function()
  local liszt write_neighbor (v : M.vertices)
    for nv in v.neighbors do
      nv.field1 = 3
    end
  end
  M.vertices:map(write_neighbor)
end, 'Non%-Exclusive WRITE')

test.fail_function(function()
  local liszt read_write_conflict (v : M.vertices)
    v.field1 = 3
    var sum : L.float = 0
    for nv in v.neighbors do
      sum += nv.field1
    end
  end
  M.vertices:map(read_write_conflict)
end, 'READ Phase is incompatible with.* EXCLUSIVE Phase')

test.fail_function(function()
  local liszt read_reduce_conflict (v : M.vertices)
    var sum : L.float = 0
    for nv in v.neighbors do
      nv.field1 += 1
      sum += nv.field1
    end
  end
  M.vertices:map(read_reduce_conflict)
end, 'READ Phase is incompatible with.* REDUCE%(%+%) Phase')

test.fail_function(function()
  local liszt reduce_reduce_conflict (v : M.vertices)
    var sum : L.float = 0
    for nv in v.neighbors do
      nv.field1 += 1
      nv.field1 *= 2
    end
  end
  M.vertices:map(reduce_reduce_conflict)
end, 'REDUCE%(%*%) Phase is incompatible with.* REDUCE%(%+%) Phase')

-- writing and reducing exclusively should be fine
local liszt write_reduce_exclusive (v : M.vertices)
  v.field1 = 3
  v.field1 += 1
end
M.vertices:map(write_reduce_exclusive)


-- two different reductions exclusively should be fine
local liszt reduce_reduce_exclusive (v : M.vertices)
  v.field1 += 2
  v.field1 *= 2
end
M.vertices:map(reduce_reduce_exclusive)




local g1 = L.Global(L.float, 32)

test.fail_function(function()
  local liszt global_write_bad (v : M.vertices)
    g1 = v.field1
  end
  M.vertices:map(global_write_bad)
end, 'Cannot write to globals in functions')

test.fail_function(function()
  local liszt global_read_reduce_conflict (v : M.vertices)
    var x = g1
    g1 += 1
  end
  M.vertices:map(global_read_reduce_conflict)
end, 'REDUCE%(%+%) Phase for Global is incompatible with.*'..
     'READ Phase for Global')

test.fail_function(function()
  local liszt global_reduce_reduce_conflict (v : M.vertices)
    g1 += v.field1
    g1 *= v.field1
  end
  M.vertices:map(global_reduce_reduce_conflict)
end, 'REDUCE%(%*%) Phase for Global is incompatible with.*'..
     'REDUCE%(%+%) Phase for Global')




