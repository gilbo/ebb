

local legion_dir    = './legion'
local bindings_dir  = legion_dir..'/bindings/terra'
local runtime_dir   = legion_dir..'/runtime'
local legion_runtime_dir   = legion_dir..'/runtime/legion'
local legion_utils_dir = './legion_utils'

-- Link in a particular library
terralib.linklibrary(bindings_dir..'/liblegion_terra_debug.so')
terralib.linklibrary(legion_utils_dir..'/liblegion_utils_debug.so')
-- Extend the Terra path that looks for C header files
terralib.includepath = terralib.includepath..';'..runtime_dir..';'..legion_runtime_dir..
                                             ';'..bindings_dir..';'..legion_utils_dir
-- Extend the Lua path
package.path = package.path .. ';'..bindings_dir..'/?.t'


local C   = terralib.includecstring([[
#include "stdio.h"
#include "stdlib.h"
]])

local Lg  = terralib.includecstring([[
#include "legion_c.h"
#include "reductions_cpu.h"
]])


local TOP_LEVEL_TASK_ID = 0
local FUTURE_TASK_ID    = 1
local FID               = 2
local REDID             = 3

local Pt1dexplicit = macro(function(val)
  return `Lg.legion_point_1d_t { x = array(val) }
end) 
local Pt1d = macro(function(val)
  return `Lg.legion_domain_point_from_point_1d(
            Lg.legion_point_1d_t { x = array(val) })
end)

-- define the tasks
terra top_level_task(
  task      : Lg.legion_task_t,
  regions   : &Lg.legion_physical_region_t,
  n_regions : uint,
  ctx       : Lg.legion_context_t,
  runtime   : Lg.legion_runtime_t)

  var num_points : int = 4

  var launch_bounds = Lg.legion_rect_1d_t {
    lo = Pt1dexplicit(0),
    hi = Pt1dexplicit(num_points-1),
  }
  var launch_domain = Lg.legion_domain_from_rect_1d(launch_bounds)

  var arg_map = Lg.legion_argument_map_create()
  for i=0,num_points do
    var input : int = i + 22
    Lg.legion_argument_map_set_point(arg_map,
      Pt1d(i),
      Lg.legion_task_argument_t {
        args   = &input,
        arglen = sizeof(int),
      },
      true -- replace = true
    )
  end

  var index_launcher = Lg.legion_index_launcher_create(
    FUTURE_TASK_ID,
    launch_domain,
    Lg.legion_task_argument_t {
      args   = nil,
      arglen = 0,
    },
    arg_map,
    Lg.legion_predicate_true(),
    false,
    0,
    0
  )

  var fm = Lg.legion_index_launcher_execute_reduction(runtime, ctx, index_launcher, REDID)
  var result = Lg.legion_future_get_result(fm)
  var result_val = [&int](result.value)
  C.printf("Result of reduction is %i\n", result_val[0])
end

terra future_task(
  task      : Lg.legion_task_t,
  regions   : &Lg.legion_physical_region_t,
  n_regions : uint,
  ctx       : Lg.legion_context_t,
  runtime   : Lg.legion_runtime_t
)
  var index_point = Lg.legion_task_get_index_point(task)
  assert(index_point.dim == 1)
  C.printf("Executing task %d!\n", index_point.point_data[0])
  assert(Lg.legion_task_get_local_arglen(task) == sizeof(int))
  var input  = @[&int](Lg.legion_task_get_local_args(task))
  var output = 2 * input
  C.printf("Sizes %d, %d\n", sizeof(Lg.legion_task_result_t), sizeof(int))
  C.printf("Output of task %d is %d\n", index_point.point_data[0], output)
  return Lg.legion_task_result_create(&output, sizeof(int))
end

terra main()
  -- register tasks
  Lg.legion_runtime_set_top_level_task_id(TOP_LEVEL_TASK_ID)
  Lg.legion_runtime_register_task_void(
    TOP_LEVEL_TASK_ID,
    Lg.LOC_PROC,
    true,
    false,
    1, -- c.AUTO_GENERATE_ID,
    Lg.legion_task_config_options_t {
      leaf = false, inner = false, idempotent = false},
    "top_level_task",
    top_level_task)
  Lg.legion_runtime_register_task(
    FUTURE_TASK_ID,
    Lg.LOC_PROC,
    true,
    false,
    1, -- c.AUTO_GENERATE_ID,
    Lg.legion_task_config_options_t {
      leaf = true, inner = false, idempotent = false},
    "future_task",
    future_task)
  Lg.legion_runtime_set_top_level_task_id(TOP_LEVEL_TASK_ID)

  -- register reduction
  Lg.register_reduction_global_plus_int32(REDID);

  -- Start the runtime
  Lg.legion_runtime_start(0, [&rawstring](0), false)
end

main()
