-- Launch liszt program as a top level legion task.

-- set up a global structure to stash legion variables into
rawset(_G, '_legion_env', {})

local C = require "compiler.c"

-- Legion library
local LW = require "compiler.legionwrap"

local terra dereference_legion_context(ctx : &LW.legion_context_t)
  return @ctx
end

local terra dereference_legion_runtime(runtime : &LW.legion_runtime_t)
  return @runtime
end

-- Setup tables and constants  for Legion runtime in Liszt.
local function setup_liszt_for_legion(ctx, runtime)
  local legion_env = rawget(_G, '_legion_env')
  legion_env.ctx      = dereference_legion_context(ctx)
  legion_env.runtime  = dereference_legion_runtime(runtime)
  legion_env.terraargs:get().ctx      = ctx
  legion_env.terraargs:get().runtime  = runtime
  return true
end
local terra_setup_liszt_for_legion =
  terralib.cast( { &LW.legion_context_t,
                   &LW.legion_runtime_t } -> bool, setup_liszt_for_legion )

-- Top level task
TID_TOP_LEVEL = 100

-- Error handler to display stack trace
local function top_level_err_handler(errobj)
  local err = tostring(errobj)
  if not string.match(err, 'stack traceback:') then
    err = err .. '\n' .. debug.traceback()
  end
  print(err)
  os.exit(1)
end

-- Launch Liszt application
function load_liszt()
  local script_filename = arg[1]
  local success = xpcall( function ()
    assert(terralib.loadfile(script_filename))()
  end, top_level_err_handler)
  print("Loaded Liszt with success " .. tostring(success))
  print("Legion tasks may still be executing in the background ...")
end

-- Run Liszt compiler/ Lua-Terra interpreter as a top level task
local terra top_level_task(
  task_args   : LW.legion_task_t,
  regions     : &LW.legion_physical_region_t,
  num_regions : uint32,
  ctx         : LW.legion_context_t,
  runtime     : LW.legion_runtime_t
)
  C.printf("Setting up Legion ...\n")
  terra_setup_liszt_for_legion(&ctx, &runtime)
  C.printf("Loading Liszt application ...\n")
  load_liszt()
  C.printf("Finished Liszt application\n")
end

-- Main function that launches Legion runtime
local terra main()
  LW.legion_runtime_register_task_void(
    TID_TOP_LEVEL, LW.LOC_PROC, true, false, 1,
    LW.legion_task_config_options_t {
      leaf = false,
      inner = false,
      idempotent = false },
    'top_level_task', top_level_task)
  LW.legion_runtime_register_task_void(
    LW.TID_SIMPLE, LW.LOC_PROC, true, false, 1,
    LW.legion_task_config_options_t {
      leaf = false,
      inner = false,
      idempotent = false },
    'simple_task', LW.simple_task)
  LW.legion_runtime_register_task(
    LW.TID_FUTURE, LW.LOC_PROC, true, false, 1,
    LW.legion_task_config_options_t {
      leaf = false,
      inner = false,
      idempotent = false },
    'future_task', LW.future_task)
  LW.legion_runtime_set_top_level_task_id(TID_TOP_LEVEL)
  LW.legion_runtime_start(0, [&rawstring](0), false)
end

main()
