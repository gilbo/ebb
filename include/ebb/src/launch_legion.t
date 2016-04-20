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

-- This file launches ebb program as a top level legion task.

local C     = require "ebb.src.c"

-- Check that Legion library is updated and built correctly so that dynamic
-- task registration is available.
local dlfcn = terralib.includec("dlfcn.h")
local terra legion_has_llvm_support() : bool
  return (dlfcn.dlsym([&opaque](0),
          "legion_runtime_register_task_variant_llvmir") ~= [&opaque](0))
end
local use_llvm = legion_has_llvm_support()

-------------------------------------------------------------------------------
--[[  Legion options/ environment                                           ]]--
-------------------------------------------------------------------------------

-- get legion command line options passed to ebb
local use_legion_spy  = rawget(_G, 'EBB_LEGION_USE_SPY')
local use_legion_prof = rawget(_G, 'EBB_LEGION_USE_PROF')
local additional_args = rawget(_G, 'EBB_ADDITIONAL_ARGS')

-- set up a global structure to stash legion variables into
rawset(_G, '_legion_env', {})
local LE = rawget(_G, '_legion_env')

-- set up a global structure to stash cluster information into
rawset(_G, '_run_config', {
                            use_ebb_mapper = true,
                            use_partitioning = rawget(_G, 'EBB_PARTITION'),
                            num_partitions_default = 2,
                            use_llvm = use_llvm
                          })
local run_config = rawget(_G, '_run_config')

-- Load Legion library (this needs run_config to be set up correctly)
local LW = require "ebb.src.legionwrap"

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

-- number of processors
local os_type = exec('uname')
local num_cpus = 1
if os_type == 'Darwin' then
  num_cpus = tonumber(exec("sysctl -n hw.ncpu"))
elseif os_type == 'Linux' then
  num_cpus = tonumber(exec("nproc"))
else
  error('unrecognized operating system: '..os_type..'\n'..
        ' Contact Developers for Support')
end
local util_cpus = 2
local num_gpus  = 1

-- legion logging options
local logging_level = "1"
-- hide warnings for region requirements without any fields
logging_level = logging_level .. ",tasks=5"
if use_legion_prof then
  logging_level = logging_level .. ",legion_prof=2"
end
if use_legion_spy then
  logging_level = logging_level .. ",legion_spy=2"
end

-- set up legion args
local legion_args = {}
table.insert(legion_args, "-level")
table.insert(legion_args, tostring(logging_level))
-- # of cpus
table.insert(legion_args, "-ll:cpu")
table.insert(legion_args, tostring(num_cpus - util_cpus))
table.insert(legion_args, "-ll:util")
table.insert(legion_args, tostring(util_cpus))
-- cpu memory
table.insert(legion_args, "-ll:csize")
table.insert(legion_args, "8000") -- MB
-- message buffer memory
table.insert(legion_args, "-ll:lmbsize")
table.insert(legion_args, "2048")
if terralib.cudacompile then
  -- # of gpus
  table.insert(legion_args, "-ll:gpu")
  table.insert(legion_args, tostring(num_gpus))
  -- gpu memory
  table.insert(legion_args, "-ll:fsize")
  table.insert(legion_args, "4000") -- MB
  -- zero-copy gpu/cpu memory (don't use)
  table.insert(legion_args, "-ll:zsize")
  table.insert(legion_args, "0") -- MB
end
if use_legion_prof then
  table.insert(legion_args, "-hl:prof")
  table.insert(legion_args, "4")
end
table.insert(legion_args, "-logfile")
table.insert(legion_args, "legion_ebb_%.log")
if additional_args then
    for word in additional_args:gmatch("%S+") do
        table.insert(legion_args, word)
    end
end


-------------------------------------------------------------------------------
--[[  Top Level Task                                                       ]]--
-------------------------------------------------------------------------------

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
load_ebb = terralib.cast({}->{},load_ebb)

-- Run Ebb compiler/ Lua-Terra interpreter as a top level task
local terra top_level_task(data : & opaque, datalen : C.size_t,
                           userdata : &opaque, userlen : C.size_t,
                           proc_id : LW.legion_lowlevel_id_t)
  -- legion preamble
  var task_args : LW.TaskArgs
  LW.legion_task_preamble(data, datalen, proc_id, &task_args.task,
                          &task_args.regions, &task_args.num_regions,
                          &task_args.lg_ctx, &task_args.lg_runtime)
  -- set global variables ctx and runtime                        
  LE.legion_env.ctx     = task_args.lg_ctx
  LE.legion_env.runtime = task_args.lg_runtime
  load_ebb()

  -- legion postamble
  LW.legion_task_postamble(task_args.lg_runtime, task_args.lg_ctx,
                           [&opaque](0), 0)
end
local TID_TOP_LEVEL = LW.get_new_task_id()


-------------------------------------------------------------------------------
--[[  Launch Legion and top level control task                             ]]--
-------------------------------------------------------------------------------
-- Main function that launches Legion runtime
local terra main()

  -- preregister top level task
  LW.legion_runtime_preregister_task_variant_fnptr(
    TID_TOP_LEVEL, LW.LOC_PROC,
    LW.legion_task_config_options_t { leaf = false, inner = false, idempotent = false },
    "top_level_task", [&opaque](0), 0, top_level_task) 
  LW.legion_runtime_set_top_level_task_id(TID_TOP_LEVEL)

  -- register reductions
  LW.RegisterReductions()

  -- arguments
  var n_args  = [1 + #legion_args]
  var args    = arrayof(rawstring,
    [arg[0]], -- include the Ebb invocation here;
    [legion_args]
  )

  if run_config.use_ebb_mapper then
    LW.register_ebb_mappers()
  end

  LW.legion_runtime_start(n_args, args, false)
end

main()
