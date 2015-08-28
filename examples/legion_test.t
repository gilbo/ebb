-- This file is to test integration of Liszt with Legion. Add code to test
-- features as they are implemented.

print("* This is a Liszt application *")

import "compiler.liszt"
local DLD = require "compiler.dld"

local C = terralib.includecstring([[ #include <stdio.h> ]])

-- Create relations and fields

local cells = L.NewRelation { name = 'cells_1d', size = 4 }

cells:NewField('mass', L.float):Load({1.2, 1.4, 2.1, 3.2})

local liszt InitMass(c)
  L.print(c.mass)
end

cells:foreach(InitMass)

local terra Printer(darray : &DLD.ctype)
  var d = darray[0]
  var c : float
  var b = d.dims
  var s = d.stride
  for i = 0, b[0] do
    for j = 0, b[1] do
      for k = 0, b[2] do
        var ptr = [&uint8](d.address) + i*s[0] + j*s[1] + k*s[2]
        C.printf("Value = %f\n", @[&float](ptr))
      end
    end
  end
end

cells.mass:DumpTerraFunction(Printer)
