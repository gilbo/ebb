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


-- Creates a map of region requirements, to be used when creating region
-- requirements for task launcher, and when codegen-ing task executable.
function T.SetUpTaskArgs(params)
  -- TODO: make this into a class
  local field_use = params.bran.kernel.field_use
  params.bran.arg_layout = {}
  local arg_layout = params.bran.arg_layout
  arg_layout.field_to_index = {}
  arg_layout.index_to_field = {}
  local field_to_index = arg_layout.field_to_index
  local index_to_field = arg_layout.index_to_field
  local reg_num = 0
  for field, access in pairs(field_use) do
    reg_num = reg_num + 1
    field_to_index[field] = reg_num
    index_to_field[reg_num] = {field}
  end
  arg_layout.num_regions = reg_num
end



-- Creates a task launcher with task region requirements.
-- Implementation details:
--  * This creates a separate region requirement for each accessed field. We
--  can group multiple fields into one region requirement, based on stencil
--  and access privileges.
--  * A region requirement with no fields is created as region req 0. This is
--  for iterating over the index space. We can instead do book-keeping about
--  which region can be used for performing iteration, or something else?
function T.CreateTaskLauncher(params)
  print("Create task launches")
  local args = params.bran.kernel_launcher:PackToTaskArg()
  -- Simple task that does not return any values
  if params.task_type == Tt.TaskTypes.simple then
    -- task launcher
    local task_launcher = Lc.legion_task_launcher_create(
                             Tt.TID_SIMPLE, args,
                             Lc.legion_predicate_true(), 0, 0)
    local field_use = params.bran.kernel.field_use
    local relset = params.bran.relset
    local field_to_index = params.bran.arg_layout.field_to_index
    local index_to_field = params.bran.arg_layout.index_to_field
    -- Add a region requirement for iterating into region req 0
    Lc.legion_task_launcher_add_region_requirement_logical_region(
      task_launcher, relset._logical_region_wrapper.handle,
      Ld.privilege.READ, Ld.coherence.READ,
      relset._logical_region_wrapper.handle, 0, false )
    -- Add region requirements for all fields that kernel accesses
    for i = 1, #index_to_field do
      local field = index_to_field[i][1]
      local access = field_use[field]
      local rel = field.owner
      print("Region req for " .. tostring(field.name) .. " from relation " ..
            tostring(rel._name) ..  " : " .. tostring(access) .. " ... ")
      local reg_req =
        Lc.legion_task_launcher_add_region_requirement_logical_region(
          task_launcher, rel._logical_region_wrapper.handle,
          Ld.privilege[tostring(access)], Ld.coherence[tostring(access)],
          rel._logical_region_wrapper.handle, 0, false )
      Lc.legion_task_launcher_add_field(
        task_launcher, reg_req, field.fid, true )
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
