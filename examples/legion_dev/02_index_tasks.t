
local C   = terralib.includecstring([[
#include "stdio.h"
#include "stdlib.h"
]])
local Lg  = terralib.includecstring([[
#include "legion_c.h"
]])


local TOP_LEVEL_TASK_ID     = 0
local HELLO_WORLD_INDEX_ID  = 1


-- helpers
local function terra2fp(tfunc)
  local def = tfunc:getdefinitions()[1]
  return def:getpointer()
end

local launch_task = macro(function(TASK_ID, arg, argtype, runtime, ctx)
  return quote
    var launcher = Lg.legion_task_launcher_create(
      TASK_ID,
      Lg.legion_task_argument_t {
        args    = &arg,
        arglen  = sizeof(argtype)
      },
      Lg.legion_predicate_true(),
      0, --Lg.legion_mapper_id_t,
      0  --Lg.legion_mapping_tag_id_t
    )
    var future = Lg.legion_task_launcher_execute(runtime,ctx,launcher)
    Lg.legion_task_launcher_destroy(launcher)
  in
    future
  end
end)

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
  runtime   : Lg.legion_runtime_t
)
  var num_points : int = 4
  var command_args = Lg.legion_runtime_get_input_args()
  if command_args.argc > 1 then
    num_points = C.atoi(command_args.argv[1])
    assert(num_points > 0)
  end
  C.printf("Running hello world redux for %d points...\n", num_points)

  -- Rect<1> launch_bounds(Point<1>(0),Point<1>(num_points-1));
  -- Domain launch_domain = Domain::from_rect<1>(launch_bounds);
  var launch_bounds = Lg.legion_rect_1d_t {
    lo = Pt1dexplicit(0),
    hi = Pt1dexplicit(num_points-1),
  }
  var launch_domain = Lg.legion_domain_from_rect_1d(launch_bounds)

  -- ArgumentMap arg_map;
  -- for (int i = 0; i < num_points; i++)
  -- {
  --   int input = i + 10;
  --   arg_map.set_point(DomainPoint::from_point<1>(Point<1>(i)),
  --       TaskArgument(&input,sizeof(input)));
  -- }
  var arg_map = Lg.legion_argument_map_create()
  for i=0,num_points do
    var input : int = i + 10
    Lg.legion_argument_map_set_point(arg_map,
      Pt1d(i),
      Lg.legion_task_argument_t {
        args   = &input,
        arglen = sizeof(int),
      },
      true -- replace = true
    )
  end

  -- IndexLauncher index_launcher(HELLO_WORLD_INDEX_ID,
  --                              launch_domain,
  --                              TaskArgument(NULL, 0),
  --                              arg_map);
  var index_launcher = Lg.legion_index_launcher_create(
    HELLO_WORLD_INDEX_ID,
    launch_domain,
    Lg.legion_task_argument_t {
      args   = nil,
      arglen = 0,
    },
    arg_map,
    Lg.legion_predicate_true(),
    false, -- must = true
    0, --Lg.legion_mapper_id_t,
    0  --Lg.legion_mapping_tag_id_t
  )

  -- FutureMap fm = runtime->execute_index_space(ctx, index_launcher);
  -- fm.wait_all_results();
  var fm = Lg.legion_index_launcher_execute(runtime, ctx, index_launcher)
  Lg.legion_future_map_wait_all_results(fm)

  -- bool all_passed = true;
  -- for (int i = 0; i < num_points; i++)
  -- {
  --   int expected = 2*(i+10);
  --   int received =
  --     fm.get_result<int>(DomainPoint::from_point<1>(Point<1>(i)));
  --   if (expected != received)
  --   {
  --     printf("Check failed for point %d: %d != %d\n",
  --             i, expected, received);
  --     all_passed = false;
  --   }
  -- }
  -- if (all_passed)
  --   printf("All checks passed!\n");
  var all_passed = true
  for i=0,num_points do
    var expected = 2*(i+10)
    var result   = Lg.legion_future_map_get_result(fm, Pt1d(i))
    var received = @[&int](result.value)
    if expected ~= received then
      C.printf("Check failed for point %d: %d != %d\n", i, expected, received)
      all_passed = false
    end
    -- Cleanup temporary for the return value
    Lg.legion_task_result_destroy(result)
  end
  if all_passed then
    C.printf("All checks passed!\n")
  end

  -- CLEANUP
  Lg.legion_future_map_destroy(fm)
  Lg.legion_index_launcher_destroy(index_launcher)
  -- destroy launch_domain ?
  Lg.legion_argument_map_destroy(arg_map)
end


terra index_space_task(
  task      : Lg.legion_task_t,
  regions   : &Lg.legion_physical_region_t,
  n_regions : uint,
  ctx       : Lg.legion_context_t,
  runtime   : Lg.legion_runtime_t
)
  var index_point = Lg.legion_task_get_index_point(task)
  assert(index_point.dim == 1)
  C.printf("Hello world from task %d!\n", index_point.point_data[0])
  assert(Lg.legion_task_get_local_arglen(task) == sizeof(int))
  var input  = @[&int](Lg.legion_task_get_local_args(task))
  var output = 2 * input
  return Lg.legion_task_result_create(&output, sizeof(int))
end

function main()
  -- must first set the top level task ID
  Lg.legion_runtime_set_top_level_task_id(TOP_LEVEL_TASK_ID)

  -- Before starting the runtime, we need to
  -- register all possible tasks to launch with the runtime.

  -- Register Tasks
  local default_options = global(Lg.legion_task_config_options_t)
  default_options.leaf = false
  default_options.inner = false
  default_options.idempotent = false
  Lg.legion_runtime_register_task_void(
    TOP_LEVEL_TASK_ID,
    Lg.LOC_PROC,
    true,   -- single = true
    false,  -- index = false
    -1,     -- AUTO_GENERATE_ID 
    default_options,
    'top_level_task',
    terra2fp(top_level_task) --Lg.legion_task_pointer_t
  )
  Lg.legion_runtime_register_task(
    HELLO_WORLD_INDEX_ID,
    Lg.LOC_PROC,
    false,  -- single = false
    true,   -- index = true
    -1,     -- AUTO_GENERATE_ID 
    default_options,
    'index_space_task',
    terra2fp(index_space_task) --Lg.legion_task_pointer_t
  )

  -- WARNING: THIS FUNCTION LEAKS MEMORY
  local function arg_convert(args)
    local arguments = global(&&int8, C.malloc(#arg * sizeof(&int8)))
    for i=1,#arg do
      arguments:get()[i-1] = global(&int8, arg[i]):get()
    end
    return arguments:get()
  end

  -- Start the runtime
  Lg.legion_runtime_start(#arg,arg_convert(arg),false)
  -- false means don't run in the background
end

main()

print('exit')