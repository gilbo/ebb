-- The MIT License (MIT)
-- 
-- Copyright (c) 2015 Stanford University.
-- All rights reserved.
-- 
-- Permission is hereby granted, free of charge, to any person obtaining a
-- copy of this software and associated documentation files (the "Software"),
-- to deal in the Software without restriction, including without limitation
-- the rights to use, copy, modify, merge, publish, distribute, sublicense,
-- and/or sell copies of the Software, and to permit persons to whom the
-- Software is furnished to do so, subject to the following conditions:
-- 
-- The above copyright notice and this permission notice shall be included
-- in all copies or substantial portions of the Software.
-- 
-- THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
-- IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
-- FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
-- AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
-- LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
-- FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
-- DEALINGS IN THE SOFTWARE.

-- Launch ebb program as a top level legion task.

-- set up a global structure to stash legion variables into
rawset(_G, '_legion_env', {})
local LE = rawget(_G, '_legion_env')

-- set up a global structure to stash cluster information into
rawset(_G, '_run_config', {
                            use_ebb_mapper = true,
                            use_partitioning = false,  -- TODO: remove this once partitioning with legion works
                                                       -- the default with legion then becomes 'one' partition
                            num_partitions = { 2, 2 },   -- TODO: set this from application, default to 1
                            num_cpus = 0,  -- 0 indicates auomatically find the number of cpus
                          })
local run_config = rawget(_G, '_run_config')

local C = require "ebb.src.c"

-- Legion library
local LW = require "ebb.src.legionwrap"

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
  -- register legion tasks
  LW.RegisterTasks()

  -- top level task
  LW.legion_runtime_register_task_void(
    TID_TOP_LEVEL, LW.LOC_PROC, true, false, 1,
    LW.legion_task_config_options_t {
      leaf = false,
      inner = false,
      idempotent = false },
    'top_level_task', top_level_task)
  LW.legion_runtime_set_top_level_task_id(TID_TOP_LEVEL)

  -- register reductions
  LW.RegisterReductions()

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
