import 'ebb'
local dld  = require 'ebb.lib.dld'
local C = terralib.includecstring([[
#include <stdio.h>
  ]])

local cells = L.NewRelation{ name = 'cells', dims = {2,3,1} }

-- print fields
local ebb dump(c, field)
  L.print(c[field])
end

-- fields
cells:NewField('x', L.float)
cells:NewField('y', L.mat2x3i)

--------------------------------------------------------------------------------

-- terra callback to load field x
----------------------------------
-- callback can load field x by performing computation or
-- reading unsupported file formats
-- callback has write only access to requested field
local terra LoadX(dldarray : &dld.ctype, t : float, str : &int8)
  var d    = dldarray[0]
  var s    = d.stride
  var dim  = d.dims

  var ptr : &float   -- data ptr
  var y : float = 0
  for i = 0, dim[0] do
    for j = 0, dim[1] do
      for k = 0, dim[2] do
        ptr = [&float]([&uint8](d.address) + i*s[0] + j*s[1] + k*s[2])
        @[ptr] = y
        y = y + t
      end
    end
  end
  C.printf("%s\n", str)
end

-- invoke LoadX callback
cells.x:LoadTerraFunction(LoadX, {0.02, "hello"})

print("Loaded x values:")
cells:foreach(dump, 'x')

-- terra callback to load field y
----------------------------------
-- callback can load field y by performing computation or
-- reading unsupported file formats
-- callback has write only access to requested field
local terra LoadY(dldarray : &dld.ctype)
  var d    = dldarray[0]
  var s    = d.stride
  var dim  = d.dims
  var st   = d.type.stride
  var dimt = d.type.dims

  var ptr : &int   -- data ptr
  for i = 0, dim[0] do
    for j = 0, dim[1] do
      for k = 0, dim[2] do
          for it = 0, dimt[0] do
            for jt = 0, dimt[1] do
                ptr = [&int]([&uint8](d.address) +
                             i*s[0] + j*s[1] + k*s[2] + 
                             it*st[0] + jt*st[1])
                @[ptr] = (i + j + k) + 10*(it + 1) + 100*(jt + 1)
            end
        end
      end
    end
  end
end

-- invoke LoadY callback
cells.y:LoadTerraFunction(LoadY)

print("Loaded y values:")
cells:foreach(dump, 'y')

--------------------------------------------------------------------------------

-- terra callback to dump x
----------------------------------
-- callback can dump field x (to stdout/ err/ any kind of file)
-- callback has read only access to requested field
local terra DumpX(dldarray : &dld.ctype)
  var d    = dldarray[0]
  var s    = d.stride
  var dim  = d.dims

  var ptr : &float   -- data ptr
  for i = 0, dim[0] do
    for j = 0, dim[1] do
      for k = 0, dim[2] do
        ptr = [&float]([&uint8](d.address) + i*s[0] + j*s[1] + k*s[2])
        C.printf("{ %i, %i, %i } : %f\n", i, j, k, @ptr)
      end
    end
  end
end

-- invoke LoadY callback
print("Dump x:")
cells.x:DumpTerraFunction(DumpX)

--------------------------------------------------------------------------------

-- terra callback to jointly dump x and y
-- callback can dump field x and y (to stdout/ err/ any kind of file)
-- callback has read only access to requested fields
local terra DumpXY(dldarray : &dld.ctype, str : &int8)
  var d0   = dldarray[0]
  var s0   = d0.stride
  var dim  = d0.dims
  var d1   = dldarray[1]
  var s1   = d1.stride
  var st   = d1.type.stride

  var ptr0 : &float   -- data ptr
  var ptr1 : &int     -- data ptr
  for i = 0, dim[0] do
    for j = 0, dim[1] do
      for k = 0, dim[2] do
        ptr0 = [&float]([&uint8](d0.address) + i*s0[0] + j*s0[1] + k*s0[2])
        C.printf("{ %i, %i, %i } : %f, ", i, j, k, @ptr0)
          for it = 0, 2 do
            C.printf("{")
            for jt = 0, 3 do
              ptr1 = [&int]([&uint8](d1.address) +
                             i*s1[0] + j*s1[1] + k*s1[2] + it*st[0] + jt*st[1])
              C.printf(" %i ", @ptr1)
            end
            C.printf("} ")
        end
      end
      C.printf("\n")
    end
  end
  C.printf("%s\n", str)
end

-- invoke DumpXY callback
print("Dump x, y:")
cells:DumpJointTerraFunction(DumpXY, {'x', 'y'}, {"a string"})
