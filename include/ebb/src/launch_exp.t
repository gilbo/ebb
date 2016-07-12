-- The MIT License (MIT)
-- 
-- Copyright (c) 2016 Stanford University.
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


local use_gpu         = rawget(_G,'EBB_USE_GPU_SIGNAL')
local cmdline_n_nodes = tonumber(rawget(_G,'EBB_EXPERIMENTAL_N_NODES'))
--print('OK JUST RECEIVED N_NODES', cmdline_n_nodes)
--for k,v in pairs(arg) do print('',k,v) end


rawset(_G,'GASNET_PRELOADED',true)
local gas     = require 'gasnet'
local gaswrap = require 'gaswrap'


-------------------------------------------------------------------------------
--[[  Initialize Networking Subsystem                                      ]]--
-------------------------------------------------------------------------------

local N_THREADS = 1

-- In order to make gasnet initialization work we need to invoke it
-- with a particular argument structure
--    BASE_CMD <number_of_nodes> [...ARGS...]
-- The result is that <number_of_nodes> copies of the command will be
-- launched across the system.  The invoked copies however, will appear
-- as if they were launched with argument structure
--    BASE_CMD [...ARGS...]
-- This is slightly complicated by the Ebb executable shifting pre-script
-- arguments into the negative range of the 'arg' table.
local arg_list = {}
local min_idx  = 0
for idx,_ in pairs(arg) do min_idx = math.min(min_idx, idx) end
table.insert(arg_list, arg[min_idx])
for k=min_idx+1,#arg do table.insert(arg_list, arg[k]) end
--print('args:')
--for k,a in pairs(arg) do print('',k,a) end
--print('mod args:')
--for k,a in pairs(arg_list) do print('',k,a) end

local terra call_init()
  var argc = [#arg_list]
  var arg_array = arrayof(rawstring, [arg_list])
  var argv = &(arg_array[0])
  gaswrap.initGasnet(argc, argv, cmdline_n_nodes, N_THREADS)
end
print('*** Before GASNet initialization.')
call_init()
print('*** After GASNet initialization.')

local N_NODES   = tonumber(gas.nodes())
local THIS_NODE = tonumber(gas.mynode())


-------------------------------------------------------------------------------
-- E Wrap Include...
-------------------------------------------------------------------------------

-- We wait until after the network has been initialized to execute the
-- E-wrap script.  Doing so ensures that the E-Wrap script is able to
-- introspect about which node it is being executed on.

local ewrap   = require 'ebb.src.ewrap'


-------------------------------------------------------------------------------
-- Register Events
-------------------------------------------------------------------------------

gaswrap.registerLuaEvent('Shutdown',function()
  print('['..THIS_NODE..'] AT THE END')
  gaswrap.shutdownGasnet()
  print('ok') -- will never run
end)


-------------------------------------------------------------------------------
--[[  Main Entry Point                                                     ]]--
-------------------------------------------------------------------------------

local function top_level_err_handler ( errobj )
  local err = tostring(errobj)
  if not string.match(err, 'stack traceback:') then
    err = err .. '\n' .. debug.traceback()
  end
  print(err)
  gaswrap.shutdownGasnetOnError()
end

script_filename = arg[0]


if THIS_NODE == 0 then
  xpcall( function ()
    -- run the script
    assert(terralib.loadfile(script_filename))()

    if use_gpu then
      -- make sure all CUDA computations have finished
      -- before we exit
      local errcode = require('ebb.src.gpu_util').device_sync()
      if errcode ~= 0 then
        error('saw CUDA error code when exiting: '..errcode)
      end
    end

    gaswrap.broadcastLuaEvent('Shutdown')
    gaswrap.pollLuaEvents(0,0)

  end, top_level_err_handler)
else
  gaswrap.startEventLoop()
end





