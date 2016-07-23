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


local Exports = {}
package.loaded["ebb.src.machine"] = Exports

local use_legion = not not rawget(_G, '_legion_env')
local use_exp    = not not rawget(_G, 'EBB_USE_EXPERIMENTAL_SIGNAL')
local use_single = not use_legion and not use_exp

local LE, legion_env, LW, use_partitioning
if use_legion then
  LE                = rawget(_G, '_legion_env')
  legion_env        = LE.legion_env[0]
  LW                = require 'ebb.src.legionwrap'
  use_partitioning  = rawget(_G, '_run_config').use_partitioning
end

local ewrap         = use_exp and require 'ebb.src.ewrap'

local Util              = require 'ebb.src.util'
local C                 = require 'ebb.src.c'
local ffi               = require 'ffi'

------------------------------------------------------------------------------

local newlist = terralib.newlist

-- a 64-bit value converted to hex should be...
local hex_str_buffer = global(int8[24])
local terra do_hex_conversion( val : uint64 ) : &int8
  C.snprintf(hex_str_buffer, 24, "%lx", val)
  return hex_str_buffer
end
local function tohexstr(obj)
  return ffi.string(do_hex_conversion(obj),16) -- exactly 16*4=64 bits
end

-------------------------------------------------------------------------------
--[[ Node Types:                                                           ]]--
-------------------------------------------------------------------------------

-- would be nice to define these here...
local NodeType    = {}
NodeType.__index  = NodeType

local all_node_types = newlist()

-- I expect to add more parameters and details here
-- as we get more specific about how we want to model the machine
local CreateNodeType = Util.memoize_named({
  'n_cpus', 'n_gpus',
},function(args)
  local nt = setmetatable({
    n_cpus = args.n_cpus,
    n_gpus = args.n_gpus,
  }, NodeType)
  all_node_types:insert(nt)
  return nt
end)

function Exports.GetAllNodeTypes() return all_node_types end

---- a simple default node type that should always work
---- though it severely under-estimates the compute power of a node
--local SingleCPUNode = setmetatable({},NodeType)
--Exports.SingleCPUNode = SingleCPUNode
--


-------------------------------------------------------------------------------
--[[ Defining a Machine                                                    ]]--
-------------------------------------------------------------------------------

local Machine   = {}
Machine.__index = Machine
local TheMachine

if use_exp then
  local function initialize_machine_model()
    local n_nodes     = ewrap.N_NODES
    local this_node   = ewrap.THIS_NODE

    -- for now just assume each machine has one CPU
    local nodes = newlist()
    for i=1,n_nodes do
      local n_threads = 1

      local node = {
        node_id = i-1,
        node_type = CreateNodeType{
          n_cpus = 1,
          n_gpus = 0,
        },
      }
    end

    TheMachine = setmetatable({
      nodes = nodes,
    }, Machine)
  end

  initialize_machine_model()
end

