local T = {}
package.loaded["compiler.legion_tasks"] = T

local C = require "compiler.c"

local LW = require "compiler.legionwrap"


-------------------------------------------------------------------------------
--[[                       Kernel Launcher Template                        ]]--
-------------------------------------------------------------------------------

local KernelLauncherTemplate = LW.KernelLauncherTemplate
local KernelLauncherSize     = LW.KernelLauncherSize


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

  -- field to region number
  arg_layout.field_to_rnum = {}
  local field_to_rnum = arg_layout.field_to_rnum

  -- field location
  arg_layout.field_num = {}
  local field_num = arg_layout.field_num
  local num_fields = 1

  -- region metadata
  arg_layout.regions = {}
  local regions = arg_layout.regions

  -- only one region for now
  regions[1] = {}
  regions[1].relation = params.bran.relset
  regions[1].fields = {}
  regions[1].num_fields = 0

  for field, access in pairs(field_use) do
    local reg_num = 1
    field_num[field] = num_fields
    field_to_rnum[field] = reg_num
    regions[reg_num].num_fields = regions[reg_num].num_fields + 1
    regions[reg_num].fields[regions[reg_num].num_fields] = field
    num_fields = num_fields + 1
  end

  arg_layout.num_regions = 1
  arg_layout.num_fields = num_fields - 1
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
  local args = params.bran.kernel_launcher:PackToTaskArg()
  -- Simple task that does not return any values
  if params.task_type == LW.TaskTypes.simple then
    -- task launcher
    local task_launcher = LW.legion_task_launcher_create(
                             LW.TID_SIMPLE, args,
                             LW.legion_predicate_true(), 0, 0)
    local field_use = params.bran.kernel.field_use
    local relset = params.bran.relset
    local field_to_rnum = params.bran.arg_layout.field_to_rnum
    local regions = params.bran.arg_layout.regions
    local num_regions = params.bran.arg_layout.num_regions
    local reg_req = {}
    for r = 1, num_regions do
      local region = regions[r]
      local rel = region.relation
      reg_req[r] =
        LW.legion_task_launcher_add_region_requirement_logical_region(
          task_launcher, rel._logical_region_wrapper.handle,
          LW.READ_WRITE, LW.EXCLUSIVE,
          rel._logical_region_wrapper.handle, 0, false )
      for f = 1, region.num_fields do
        local field = region.fields[f]
        local access = field_use[field]
        local rel = field.owner
        print("In create task launcher, adding field " .. field.fid .. " to region req " .. r)
        LW.legion_task_launcher_add_field(
          task_launcher, reg_req[r], field.fid, true )
      end
    end
    return task_launcher
  elseif params.task_type == LW.TaskTypes.fut then
    error("INTERNAL ERROR: Liszt does not handle tasks with future values yet")
  else
    error("INTERNAL ERROR: Unknown task type")
  end
end

-- Launches Legion task and returns.
function T.LaunchTask(p, leg_args)
  print("Launching legion task")
   LW.legion_task_launcher_execute(leg_args.runtime, leg_args.ctx,
                                   p.task_launcher)
  print("Launched task")
end
