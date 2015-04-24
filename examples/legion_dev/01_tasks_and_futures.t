

local legion_dir    = './legion'
local bindings_dir  = legion_dir..'/bindings/terra'
local runtime_dir   = legion_dir..'/runtime'

-- Link in a particular library
terralib.linklibrary(bindings_dir..'/liblegion_terra_debug.so')
-- Extend the Terra path that looks for C header files
terralib.includepath = terralib.includepath..';'..runtime_dir..
                                             ';'..bindings_dir
-- Extend the Lua path
package.path = package.path .. ';'..bindings_dir..'/?.t'


local C   = terralib.includecstring([[
#include "stdio.h"
#include "stdlib.h"
]])
local Lg  = terralib.includecstring([[
#include "legion_c.h"
]])


local TOP_LEVEL_TASK_ID = 0
local FIBONACCI_TASK_ID = 1
local SUM_TASK_ID       = 2


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


-- define the tasks

terra top_level_task(
  task      : Lg.legion_task_t,
  regions   : &Lg.legion_physical_region_t,
  n_regions : uint,
  ctx       : Lg.legion_context_t,
  runtime   : Lg.legion_runtime_t
)
  var num_fibonacci : int = 7
  var command_args = Lg.legion_runtime_get_input_args()
  if command_args.argc > 1 then
    num_fibonacci = C.atoi(command_args.argv[1])
    assert(num_fibonacci >= 0)
  end
  C.printf("Computing the first %d Fibonacci numbers...\n", num_fibonacci)

  var fib_results = [&Lg.legion_future_t](
    C.malloc(num_fibonacci * sizeof(Lg.legion_future_t)))
  -- launch a bunch of tasks and accumulate results
  for i=0,num_fibonacci do
    fib_results[i] = launch_task(FIBONACCI_TASK_ID, i, int, runtime, ctx)
  end

  -- print out the results and cleanup futures
  for i=0,num_fibonacci do
    var result = Lg.legion_future_get_result(fib_results[i])
    var rval   = @[&int](result.value)
    C.printf("Fibonacci(%d) = %d\n", i, rval)

    Lg.legion_future_destroy(fib_results[i])
  end
  C.free(fib_results)

end

terra fibonacci_task(
  task      : Lg.legion_task_t,
  regions   : &Lg.legion_physical_region_t,
  n_regions : uint,
  ctx       : Lg.legion_context_t,
  runtime   : Lg.legion_runtime_t
)
  assert(Lg.legion_task_get_arglen(task) == sizeof(int))
  var fib_num : int = @([&int](Lg.legion_task_get_args(task)))

  var zero : int = 0
  var one  : int = 1
  if fib_num == 0 then
    return Lg.legion_task_result_create(&zero,sizeof(int))
  end
  if fib_num == 1 then
    return Lg.legion_task_result_create(&one,sizeof(int))
  end

  -- launch fib-1
  var fib1 : int = fib_num-1
  var f1 = launch_task(FIBONACCI_TASK_ID, fib1, int, runtime, ctx)

  -- launch fib-2
  var fib2 : int = fib_num-2
  var f2 = launch_task(FIBONACCI_TASK_ID, fib2, int, runtime, ctx)

  -- launch sum task
  var sum = Lg.legion_task_launcher_create(
    SUM_TASK_ID,
    Lg.legion_task_argument_t {
      args    = nil,
      arglen  = 0
    },
    Lg.legion_predicate_true(),
    0, --Lg.legion_mapper_id_t,
    0  --Lg.legion_mapping_tag_id_t
  )
  Lg.legion_task_launcher_add_future(sum, f1)
  Lg.legion_task_launcher_add_future(sum, f2)
  var f_sum = Lg.legion_task_launcher_execute(runtime, ctx, sum)

  var result  = Lg.legion_future_get_result(f_sum)

  -- cleanup
  Lg.legion_future_destroy(f_sum)
  Lg.legion_future_destroy(f2)
  Lg.legion_future_destroy(f1)
  Lg.legion_task_launcher_destroy(sum)
  -- complete return
  return result
  --return Lg.legion_task_result_create(result.value, sizeof(int))
end

terra sum_task(
  task      : Lg.legion_task_t,
  regions   : &Lg.legion_physical_region_t,
  n_regions : uint,
  ctx       : Lg.legion_context_t,
  runtime   : Lg.legion_runtime_t
)
  assert(Lg.legion_task_get_futures_size(task) == 2)

  var f1 = Lg.legion_task_get_future(task, 0)
  var r1 = Lg.legion_future_get_result(f1)
  var i1 : int = @[&int](r1.value)
  var f2 = Lg.legion_task_get_future(task, 1)
  var r2 = Lg.legion_future_get_result(f2)
  var i2 : int = @[&int](r2.value)

  var sum : int = i1 + i2

  -- cleanup
  Lg.legion_task_result_destroy(r2)
  Lg.legion_task_result_destroy(r1)
  -- complete return
  return Lg.legion_task_result_create(&sum,sizeof(int))
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
    FIBONACCI_TASK_ID,
    Lg.LOC_PROC,
    true,   -- single = true
    false,  -- index = false
    -1,     -- AUTO_GENERATE_ID 
    default_options,
    'fibonacci_task',
    terra2fp(fibonacci_task) --Lg.legion_task_pointer_t
  )
  local leaf_options = global(Lg.legion_task_config_options_t)
  leaf_options.leaf       = true
  leaf_options.inner      = false
  leaf_options.idempotent = false
  Lg.legion_runtime_register_task(
    SUM_TASK_ID,
    Lg.LOC_PROC,
    true,   -- single = true
    false,  -- index = false
    -1,     -- AUTO_GENERATE_ID 
    leaf_options,
    'sum_task',
    terra2fp(sum_task) --Lg.legion_task_pointer_t
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