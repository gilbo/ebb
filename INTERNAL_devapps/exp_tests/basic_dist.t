import 'ebb'
local L = require 'ebblib'

local ewrap     = require 'ebb.src.ewrap'
local C = terralib.includecstring [[
  #include <stdio.h>
  #include <stdlib.h>
  #include <unistd.h>
]]


-------------------------------------------------------------------------------
-- Control program
-------------------------------------------------------------------------------

-- Data
local cells = L.NewRelation { name='cells', dims={8,8}, periodic={true,false} }
cells:SetPartitions {2,2}
local mass = cells:NewField('mass', L.float)

local function genfoo()
  local N = ewrap.THIS_NODE
  local Foo = terralib.types.newstruct('foo'..N)
  Foo.entries:insertall{
    { 'x', double },
    { 'y', double },
  }
  for k=1,N do Foo.entries:insert{ 'v'..k, double } end
  Foo:complete()
  return Foo
end
local Foo = genfoo()

-- Init mass task
local terra InitMass(args : ewrap.TaskArgs)
  C.printf('Executing InitMass\n')
  var x_lo = args.bounds[0].lo
  var y_lo = args.bounds[1].lo
  var x_hi = args.bounds[0].hi
  var y_hi = args.bounds[1].hi
  C.printf('Bounds are {%i, %i}, {%i, %i}\n', x_lo, x_hi, y_lo, y_hi)
  var mass = args.fields[0]
  var stride = mass.strides
  C.printf('Elem stride %i, %i\n', stride[0], stride[1])
  C.printf('Ghost adjusted pointer %p\n', mass.ptr)
  var mass_ptr = [&float](mass.ptr)
  var drift = 0.01
  for y = y_lo,y_hi do
    for x = x_lo,x_hi do
      var offset = (x-x_lo)*stride[0] + (y-y_lo)*stride[1]
      mass_ptr[offset] = 2 + drift
      drift = drift + 0.01
      C.printf('x %i, y %i, offset %i, mass %f\n', x, y, offset, mass_ptr[offset])
    end
  end
  var f : Foo
  f.x = 3.4
  f.y = 2.9
  C.printf('init foo print %f %f\n', f.x, f.y)
  C.printf('Completed InitMass\n')
end
local init_mass_field_accesses = {
  ewrap.NewFAccess {
    field = mass._ewrap_field,
    privilege = ewrap.READ_WRITE_PRIVILEGE,
  }
}
local init_mass_etask = ewrap.RegisterNewTask{
  func            = InitMass,
  name            = 'InitMass',
  relation        = cells._ewrap_relation,
  processor       = L.CPU,
  field_accesses  = init_mass_field_accesses,
}

-- Dump mass task
local terra DumpMass(args : ewrap.TaskArgs)
  C.printf('Executing DumpMass\n')
  var x_lo = args.bounds[0].lo
  var y_lo = args.bounds[1].lo
  var x_hi = args.bounds[0].hi
  var y_hi = args.bounds[1].hi
  C.printf('Bounds are {%i, %i}, {%i, %i}\n', x_lo, x_hi, y_lo, y_hi)
  var mass = args.fields[0]
  var stride = mass.strides
  C.printf('Elem stride %i, %i\n', stride[0], stride[1])
  C.printf('Ghost adjusted pointer %p\n', mass.ptr)
  var mass_ptr = [&float](mass.ptr)
  for y = y_lo,y_hi do
    for x = x_lo,x_hi do
      var offset = (x-x_lo)*stride[0] + (y-y_lo)*stride[1]
      C.printf('x %i, y %i, offset %i, mass %f\n', x, y, offset, mass_ptr[offset])
    end
  end
  C.printf('Completed DumpMass\n')
end
local dump_mass_field_accesses = {
  ewrap.NewFAccess {
    field     = mass._ewrap_field,
    privilege = ewrap.READ_ONLY_PRIVILEGE,
  }
}
local dump_mass_etask = ewrap.RegisterNewTask{
  func            = DumpMass,
  name            = 'DumpMass',
  relation        = cells._ewrap_relation,
  processor       = L.CPU,
  field_accesses  = dump_mass_field_accesses,
}

init_mass_etask:exec()
dump_mass_etask:exec()

print("*** Completed control pogram.")
