local T = {}
package.loaded["compiler.legion_tasks"] = T

local C = require "compiler.c"

-- Legion library
require "legionlib"
local Lc = terralib.includecstring([[
#include "legion_c.h"
]])

local Ld = require "compiler.legion_data"
local Tt = require "compiler.legion_task_types"


-------------------------------------------------------------------------------
--[[                       Kernel Launcher Template                        ]]--
-------------------------------------------------------------------------------

local KernelLauncherTemplate = Tt.KernelLauncherTemplate
local KernelLauncherSize     = Tt.KernelLauncherSize


-------------------------------------------------------------------------------
--[[                         Legion task launcher                          ]]--
-------------------------------------------------------------------------------


-- Creates a task launcher with task region requirements.
-- Implementation details:
--  * This creates a separate region requirement for each accessed field. We
--  can group multiple fields into one region requirement, based on stencil
--  and access privileges.
--  * A region requirement with no fields is created as region req 0. This is
--  for iterating over the index space. We can instead do book-keeping about
--  whcih region can be used for performing iteration, or something else?
function T.CreateTaskLauncher(params)
  local args = params.bran.kernel_launcher:PackToTaskArg()
  -- Simple task that does not return any values
  if params.task_type == Tt.TaskTypes.simple then
    -- task launcher
    local task_launcher = Lc.legion_task_launcher_create(
                             Tt.TID_SIMPLE, args,
                             Lc.legion_predicate_true(), 0, 0)
    local field_use = params.bran.kernel.field_use
    params.bran.field_reg_map = {}
    local field_reg_map = params.bran.field_reg_map
    local relset = params.bran.relset
    -- Add a region requirement for iterating into region req 0
    Lc.legion_task_launcher_add_region_requirement_logical_region(
      task_launcher, relset._logical_region_wrapper.handle,
      Ld.privilege.READ, Ld.coherence.READ,
      relset._logical_region_wrapper.handle, 0, false )
    -- Add region requirements for all fields that kernel accesses
    local reg_req = {}
    local reg_num = 1
    for field, access in pairs(field_use) do
      local rel = field.owner
      print("Region req for " .. tostring(field.name) .. " from relation " ..
            tostring(rel) ..  " : " .. tostring(access))
      reg_req[field] =
        Lc.legion_task_launcher_add_region_requirement_logical_region(
          task_launcher, rel._logical_region_wrapper.handle,
          Ld.privilege[tostring(access)], Ld.coherence[tostring(access)],
          rel._logical_region_wrapper.handle, 0, false )
      field_reg_map[field] = reg_num
      reg_num = reg_num + 1
    end
    for field, access in pairs(field_use) do
      print(tostring(field.name) .. " : " .. tostring(access))
      Lc.legion_task_launcher_add_field(
        task_launcher, reg_req[field], field.fid, true )
      print("Added field to region requirement")
    end
    return task_launcher
  elseif params.task_type == Tt.TaskTypes.fut then
    error("INTERNAL ERROR: Liszt does not handle tasks with future values yet")
  else
    error("INTERNAL ERROR: Unknown task type")
  end
end

-- Launches Legion task and returns.
function T.LaunchTask(p, leg_args)
  print("Launching legion task")
  local f = Lc.legion_task_launcher_execute(leg_args.runtime, leg_args.ctx,
                                            p.task_launcher)
  print("Launched task")
end
