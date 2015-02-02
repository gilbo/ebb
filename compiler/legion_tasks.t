local T = {}
package.loaded["compiler.legion_tasks"] = T

local C = terralib.require "compiler.c"

-- Legion library
terralib.require "legionlib"
local Lc = terralib.includecstring([[
#include "legion_c.h"
]])


-------------------------------------------------------------------------------
--[[                       Kernel Launcher Template                        ]]--
-------------------------------------------------------------------------------
--[[ Kernel laucnher template is a wrapper for a callback function that is
--   passed to a Legion task, when a Liszt kernel is invoked. We pass a
--   function as an argument to a Legion task, because there isn't a way to
--   dynamically register and invoke tasks.
--]]--

struct KernelLauncherTemplate {
  Launch : { &Lc.legion_physical_region_t,
             uint32,
             Lc.legion_context_t,
             Lc.legion_runtime_t } -> bool;
}
T.KernelLauncherTemplate = KernelLauncherTemplate

terra T.NewKernelLauncher(
  kernel_code : { &Lc.legion_physical_region_t, uint32, Lc.legion_context_t,
                  Lc.legion_runtime_t } -> bool )
  var l : KernelLauncherTemplate
  l.Launch = kernel_code
  return l
end

local KernelLauncherSize = terralib.sizeof(KernelLauncherTemplate)

-- Pack kernel launcher into a task argument for Legion.
terra KernelLauncherTemplate:PackToTaskArg()
  var sub_args = Lc.legion_task_argument_t {
    args = [&opaque](self),
    arglen = KernelLauncherSize
  }
  return sub_args
end


-------------------------------------------------------------------------------
--[[                             Legion Tasks                              ]]--
-------------------------------------------------------------------------------
--[[ A simple task is a task that does not have any return value. A fut_task
--   is a task that returns a Legion future, or return value.
--]]--

terra T.simple_task(args : Lc.legion_task_t,
                    regions : &Lc.legion_physical_region_t,
                    num_regions : uint32,
                    ctx : Lc.legion_context_t,
                    runtime : Lc.legion_runtime_t)
  C.printf("Executing simple task\n")
  var arglen = Lc.legion_task_get_arglen(args)
  assert(arglen == KernelLauncherSize)
  var kernel_launcher : &KernelLauncherTemplate =
    [&KernelLauncherTemplate](Lc.legion_task_get_args(args))
  kernel_launcher.Launch(regions, num_regions, ctx, runtime)
end

T.TID_SIMPLE = 200

terra T.fut_task(args : Lc.legion_task_t,
                 regions : &Lc.legion_physical_region_t,
                 num_regions : uint32,
                 ctx : Lc.legion_context_t,
                 runtime : Lc.legion_runtime_t) : Lc.legion_task_result_t
  C.printf("Executing future task\n")
  var dummy : int = 9
  var result = Lc.legion_task_result_create(&dummy, terralib.sizeof(int))
  return result
end

T.TID_FUT = 300

T.TaskTypes = { simple = 'simple', fut = 'fut' }

-------------------------------------------------------------------------------
--[[                         Legion task launcher                          ]]--
-------------------------------------------------------------------------------

function T.CreateTaskLauncher(params)
  local args = params.kernel_launcher:PackToTaskArg()
  if params.task_type == T.TaskTypes.simple then
    local task_launcher = Lc.legion_task_launcher_create(
                             T.TID_SIMPLE, args,
                             Lc.legion_predicate_true(), 0, 0)
    return task_launcher
  elseif params.task_type == T.TaskTypes.fut then
    error("INTERNAL ERROR: Liszt does not handle tasks with future values yet")
  else
    error("INTERNAL ERROR: Unknown task type")
  end
end

-- Launches Legion task and returns.
function T.LaunchTask(p)
  print("Launching legion task")
  local f = Lc.legion_task_launcher_execute(p.runtime, p.ctx, p.task_launcher)
  -- TODO: no need to wait on future value technically, but Legion exits before
  -- the task is completed, why?
  -- NOTE: This is going to cause the program to crash right now, but we need
  -- it till we implement waiting for all unfinished tasks.
  local res = Lc.legion_future_get_result(f)
end
