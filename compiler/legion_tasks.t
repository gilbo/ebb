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



function T.CreateTaskLauncher(params)
  local args = params.kernel_launcher:PackToTaskArg()
  -- Simple task that does not return any values
  if params.task_type == Tt.TaskTypes.simple then
    -- task launcher
    local task_launcher = Lc.legion_task_launcher_create(
                             -- TODO: switch to simple once waiting on
                             -- unifinished tasks is resolved 
                             -- Tt.TID_SIMPLE, args,
                             Tt.TID_FUT, args,
                             Lc.legion_predicate_true(), 0, 0)
    local field_use = params.kernel.field_use
    -- Create region requirements
    -- TODO: A separate region for each field access for now. Can group fields
    -- by relation, access mode and partitions to reduce number of physical
    -- regions.
    local reg_req = {}
    for field, access in pairs(field_use) do
      print("Region req for " .. tostring(field.name) .. " : " .. tostring(access))
      local rel = field.owner
      reg_req[field] =
        Lc.legion_task_launcher_add_region_requirement_logical_region(
          task_launcher, rel._logical_region_wrapper.handle,
          Ld.privilege[tostring(access)], Ld.coherence[tostring(access)],
          rel._logical_region_wrapper.handle, 0, false )
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
  -- TODO: no need to wait on future value technically, but Legion exits before
  -- the task is completed, why?
  -- NOTE: This is going to cause the program to crash right now, but we need
  -- it till we implement waiting for all unfinished tasks.
  local res = Lc.legion_future_get_result(f)
end
