-- Launch liszt program as a top level legion task.

local C = require "compiler.c"

-- Legion library
require "legionlib-terra"
local Lc = terralib.includecstring([[
#include "legion_c.h"
]])

-- Legion tasks for launching Liszt kernels
local T = require "compiler.legion_tasks"

local terra dereference_legion_context(ctx : &Lc.legion_context_t)
  return @ctx
end

local terra dereference_legion_runtime(runtime : &Lc.legion_runtime_t)
  return @runtime
end

-- Setup tables and constants  for Legion runtime in Liszt.
local function setup_liszt_for_legion(ctx, runtime)
  rawset(_G, '_legion', true)
  local legion_env = { ctx     = dereference_legion_context(ctx),
                       runtime = dereference_legion_runtime(runtime) }
  rawset(_G, '_legion_env', legion_env)
  return true
end
local terra_setup_liszt_for_legion =
  terralib.cast( { &Lc.legion_context_t,
                   &Lc.legion_runtime_t } -> bool, setup_liszt_for_legion )

-- Top level task
TID_TOP_LEVEL = 100

-- Error handler to display stack trace
local function top_level_err_handler(errobj)
  local err = tostring(errobj)
  if string.match(err, "stack traceback:") then
    print(err)
  else
    print(err .. "\n" .. debug.traceback())
  end
  os.exit(1)
end

-- Launch Liszt application
function load_liszt()
  local script_filename = arg[1]
  local success = xpcall( function ()
    assert(terralib.loadfile(script_filename))()
  end, top_level_err_handler)
  if success then
    os.exit(0)
  else
    os.exit(1)
  end
end

-- Run Liszt compiler/ Lua-Terra interpreter as a top level task
local terra top_level_task(task_args : Lc.legion_task_t,
                           regions : &Lc.legion_physical_region_t,
                           num_regions : uint32,
                           ctx : Lc.legion_context_t,
                           runtime : Lc.legion_runtime_t)
  C.printf("Setting up Legion ...\n")
  terra_setup_liszt_for_legion(&ctx, &runtime)
  C.printf("Loading Liszt application ...\n")
  load_liszt()
  C.printf("Finished Liszt application\n")
end

-- Main function that launches Legion runtime
local terra main()
  Lc.legion_runtime_register_task_void(
    TID_TOP_LEVEL, Lc.LOC_PROC, true, false, 1,
    Lc.legion_task_config_options_t {
      leaf = false,
      inner = false,
      idempotent = false },
    'top_level_task', top_level_task)
  Lc.legion_runtime_register_task_void(
    T.TID_SIMPLE, Lc.LOC_PROC, true, true, 1,
    Lc.legion_task_config_options_t {
      leaf = false,
      inner = false,
      idempotent = false },
    'simple_task', T.simple_task)
  Lc.legion_runtime_register_task(
    T.TID_FUT, Lc.LOC_PROC, true, true, 1,
    Lc.legion_task_config_options_t {
      leaf = false,
      inner = false,
      idempotent = false },
    'fut_task', T.fut_task)
  Lc.legion_runtime_set_top_level_task_id(TID_TOP_LEVEL)
  Lc.legion_runtime_start(0, [&rawstring](0), false)
end

main()
