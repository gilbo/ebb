-- Launch ebb program as a top level legion task.

-- set up a global structure to stash legion variables into
rawset(_G, '_legion_env', {})
local LE = rawget(_G, '_legion_env')

-- set up a global structure to stash cluster information into
rawset(_G, '_run_config', {
                            use_ebb_mapper = true,
                            use_partitioning = false,  -- TODO: set this using command line argument
                            num_partitions = { 2 },   -- TODO: set this using command line argument?
                            num_cpus = 0,  -- 0 indicates auomatically find the number of cpus
                          })
local run_config = rawget(_G, '_run_config')

local C = require "ebb.src.c"

-- Legion library
local LW = require "ebb.src.legionwrap"

local terra dereference_legion_context(ctx : &LW.legion_context_t)
  return @ctx
end

local terra dereference_legion_runtime(runtime : &LW.legion_runtime_t)
  return @runtime
end

-- Top level task
TID_TOP_LEVEL = 50

-- Error handler to display stack trace
local function top_level_err_handler(errobj)
  local err = tostring(errobj)
  if not string.match(err, 'stack traceback:') then
    err = err .. '\n' .. debug.traceback()
  end
  print(err)
  os.exit(1)
end

-- Launch Ebb application
function load_ebb()
  local script_filename = arg[0]
  local success = xpcall( function ()
    assert(terralib.loadfile(script_filename))()
    LW.heavyweightBarrier()
  end, top_level_err_handler)
end

-- Run Ebb compiler/ Lua-Terra interpreter as a top level task
local terra top_level_task(
  task_args   : LW.legion_task_t,
  regions     : &LW.legion_physical_region_t,
  num_regions : uint32,
  ctx         : LW.legion_context_t,
  runtime     : LW.legion_runtime_t
)
  LE.legion_env.ctx = ctx
  LE.legion_env.runtime = runtime
  load_ebb()
end

-- Note 4 types of processors

--      TOC_PROC = ::TOC_PROC, // Throughput core
--      LOC_PROC = ::LOC_PROC, // Latency core
--      UTIL_PROC = ::UTIL_PROC, // Utility core
--      PROC_GROUP = ::PROC_GROUP, // Processor group


local function exec(cmd)
  local handle  = io.popen(cmd)
  local out     = handle:read()
  handle:close()
  return out
end

if run_config.num_cpus == 0 then
  local os_type = exec('uname')
  local n_cpu   = 1
  
  if os_type == 'Darwin' then
    n_cpu = tonumber(exec("sysctl -n hw.ncpu"))
  elseif os_type == 'Linux' then
    n_cpu = tonumber(exec("nproc"))
  else
    error('unrecognized operating system: '..os_type..'\n'..
          ' Contact Developers for Support')
  end
  run_config.num_cpus = n_cpu
end

local use_legion_spy  = rawget(_G, 'EBB_LEGION_USE_SPY')
local use_legion_prof = rawget(_G, 'EBB_LEGION_USE_PROF')
local logging_level = 5
if use_legion_prof or use_legion_spy then
  logging_level = 2
end
local logging_cat
if use_legion_prof then
  logging_cat = 'legion_prof'
end
if use_legion_spy then
  logging_cat = 'legion_spy'
end

local legion_args = {}
table.insert(legion_args, "-level")
table.insert(legion_args, tostring(logging_level))
-- # of cpus
table.insert(legion_args, "-ll:cpu")
table.insert(legion_args, tostring(run_config.num_cpus))
-- cpu memory
--table.insert(legion_args, "-ll:csize")
--table.insert(legion_args, "512") -- MB
if terralib.cudacompile then
  -- # of gpus
  table.insert(legion_args, "-ll:gpu")
  table.insert(legion_args, tostring(1))
  -- gpu memory
  --table.insert(legion_args, "-ll:fsize")
  --table.insert(legion_args, "256") -- MB
  -- zero-copy gpu/cpu memory (don't use)
  table.insert(legion_args, "-ll:zsize")
  table.insert(legion_args, "0") -- MB
end
-- stack memory
--table.insert(legion_args, "-ll:stack")
--table.insert(legion_args, "2") -- MB
if logging_cat then
  table.insert(legion_args, "-cat")
  table.insert(legion_args, logging_cat)
end
if logging_cat == 'legion_prof' then
  table.insert(legion_args, "-hl:prof")
  table.insert(legion_args, "1")
end


-- Main function that launches Legion runtime
local terra main()
  var ids_simple_cpu = arrayof(LW.legion_task_id_t, [LW.TID_SIMPLE_CPU])
  var ids_simple_gpu = arrayof(LW.legion_task_id_t, [LW.TID_SIMPLE_GPU])
  var ids_future_cpu = arrayof(LW.legion_task_id_t, [LW.TID_FUTURE_CPU])
  var ids_future_gpu = arrayof(LW.legion_task_id_t, [LW.TID_FUTURE_GPU])
  LW.legion_runtime_register_task_void(
    TID_TOP_LEVEL, LW.LOC_PROC, true, false, 1,
    LW.legion_task_config_options_t {
      leaf = false,
      inner = false,
      idempotent = false },
    'top_level_task', top_level_task)

  for i = 0, LW.NUM_TASKS do
    var simple_cpu : int8[25]
    var simple_gpu : int8[25]
    var future_cpu : int8[25]
    var future_gpu : int8[25]
    C.sprintf(simple_cpu, "simple_task_cpu_%d", ids_simple_cpu[i])
    C.sprintf(simple_gpu, "simple_task_gpu_%d", ids_simple_gpu[i])
    C.sprintf(future_cpu, "future_task_cpu_%d", ids_future_cpu[i])
    C.sprintf(future_gpu, "future_task_gpu_%d", ids_future_gpu[i])
    LW.legion_runtime_register_task_void(
      ids_simple_cpu[i], LW.LOC_PROC, true, false, 1,
      LW.legion_task_config_options_t {
        leaf = true,
        inner = false,
        idempotent = false },
      simple_cpu, LW.simple_task)
    LW.legion_runtime_register_task_void(
      ids_simple_gpu[i], LW.TOC_PROC, true, false, 1,
      LW.legion_task_config_options_t {
        leaf = true,
        inner = false,
        idempotent = false },
      simple_gpu, LW.simple_task)

    LW.legion_runtime_register_task(
      ids_future_cpu[i], LW.LOC_PROC, true, false, 1,
      LW.legion_task_config_options_t {
        leaf = true,
        inner = false,
        idempotent = false },
      future_cpu, LW.future_task)
    LW.legion_runtime_register_task(
      ids_future_gpu[i], LW.TOC_PROC, true, false, 1,
      LW.legion_task_config_options_t {
        leaf = true,
        inner = false,
        idempotent = false },
      future_gpu, LW.future_task)
  end

  LW.legion_runtime_set_top_level_task_id(TID_TOP_LEVEL)

  -- arguments
  var n_args  = [1 + #legion_args]
  var args    = arrayof(rawstring,
    [arg[0]], -- include the Ebb invocation here;
                           -- doesn't matter though
    [legion_args]
  )

  if run_config.use_ebb_mapper then
    LW.register_ebb_mappers()
  end

  LW.legion_runtime_start(n_args, args, false)
end

main()
