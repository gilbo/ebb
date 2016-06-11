import 'ebb'
local L = require 'ebblib'

local ewrap     = require 'ebb.src.ewrap'
local stencil   = require 'ebb.src.stencil'

local C = terralib.includecstring [[
  #include <stdio.h>
  #include <stdlib.h>
  #include <unistd.h>
]]


-------------------------------------------------------------------------------
-- Control program
-------------------------------------------------------------------------------

local cells = L.NewRelation { name='cells', dims={8,8} }
cells:SetPartitions {2,2}

local mass = cells:NewField('mass', L.float)

local terra InitMass(task_args : &opaque)
  var args = [&ewrap.TaskArgs](task_args)
  C.printf('Bounds are {%i, %i}, {%i, %i}\n',
           args.bounds[0].lo, args.bounds[1].lo,
           args.bounds[0].hi, args.bounds[1].hi)
end

local init_mass_fields = { mass._ewrap_field.id }
local init_mass_field_accesses = {
  [mass] = stencil.NewCenteredAccessPattern {
    field = mass,
    read  = true,
    write = true,
  }
}

local init_mass_id = ewrap.RegisterNewTask({
  task_func      = InitMass,
  task_name      = 'InitMass',
  rel_id         = cells._ewrap_relation.id,
  processor      = L.CPU,
  fields         = init_mass_fields,
  field_accesses = init_mass_field_accesses,
})

ewrap.SendTaskLaunch(init_mass_id)

print("*** Completed control pogram.")
