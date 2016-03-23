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
package.loaded["ebb.src.planner"] = Exports

local use_legion = not not rawget(_G, '_legion_env')
local use_single = not use_legion
local LE, legion_env, LW
if use_legion then
  LE            = rawget(_G, '_legion_env')
  legion_env    = LE.legion_env[0]
  LW            = require 'ebb.src.legionwrap'
end


local Util              = require 'ebb.src.util'
local P                 = require 'ebb.src.partitions'

-------------------------------------------------------------------------------

local newlist = terralib.newlist

-------------------------------------------------------------------------------
--[[ The Planner:                                                          ]]--
-------------------------------------------------------------------------------

-- The planner is a singleton object.  I declare it as an object here in
-- case we have a change of heart later about how it should work.
-- The alternative would be to scatter its internals throughout this
-- file, which I'd rather not do...

local Planner   = {}
Planner.__index = Planner

-- only designed to be called this one time right now
local function NewPlanner()

  return setmetatable({
    -- sets of active objects
      active_local_partitions   = {},
      active_node_types         = {},
      active_funcs              = {},

    -- statistics & other observed data:
      f_call_count    = {}, -- set of invoked funcs with invocation counts

    -- maintained indices
      partition_index       = Util.new_named_cache({
        'typedfunc', 'relation', 'node_type', 'partition_strategy',
      }), -- stores RelLocalPartition objects
      local_ghost_index     = Util.new_named_cache {
        'rel_local_partition', 'ghost_strategy',
      }, -- stores local_ghost_partition
      legion_data_index     = Util.new_named_cache {
        'local_ghost_partition', 'node_id', 'proc_id',
      }, -- stores legion_stuff...?

    -- "state_flags" (used to signal potentially invalid indices)
      new_func_queue            = newlist(),
      new_local_partition_queue = newlist(),
  },Planner)
end
local ThePlanner = NewPlanner()

-------------------------------------------------------------------------------
--[[ Planner Interface:                                                    ]]--
-------------------------------------------------------------------------------

-- MAYBE this should be perfomed automatically on startup instead of
-- being exposed at the module interface? dunno
--    e.g. 1) simply do this work in-line
--    e.g. 2) call this on the first note_launch() call
function Exports.register_node_types(node_types)
  local self = ThePlanner

  self.active_node_types = newlist()
  for i,d in ipairs(ndescs) do self.active_node_types[i] = d end
end

-----------------------------------------
--  This function should be called every time
--  the system is preparing to launch a given
--  function.  It provides a hook for providing
--  any useful information about the launch so
--  that the planner can observe and adapt
--[[
{
  typedfunc
  ...
}
--]]
function Exports.note_launch(args)
  local self = ThePlanner

  -- note statistics
  local call_count = self.f_call_count[args.typedfunc]
  self.f_call_count[args.typedfunc] = (call_count or 0) + 1

  -- and invalidate indices if needed
  if not self.active_funcs[args.typedfunc] then
    self.active_funcs[ args.typedfunc ] = true
    self.new_func_queue:insert( args.typedfunc )
  end
end

-----------------------------------------------
--  This function is called to get necessary
--  launch parameters: partition data so that
--  a Legion launch can be performed.
--
-- do we need to query for node and proc id?
function Exports.query_for_partitions(typedfunc, node_desc, node_id, proc_id)
  local self = ThePlanner

  -- THIS IS A HACK.  Do we even want to supply node and proc ids?
  --                  Would it be better to just supply node and processor
  --                  type?
  --                  This depends a lot on how the actual legion launch is
  --                  done...
  node_id = node_id or 1
  node_id = proc_id or 1

  -- make sure all indices are fresh before satisfying a query
  if #self.new_func_queue > 0 then
    self:refresh_local_ghost_index()
  end
  if #self.new_local_partition_queue > 0 then
    self:rebuild_partitions()
  end

  -- make / get partitioning decision
  local partition_strategy  =
    self:choose_partition_strategy(typedfunc, node_desc)

  -- make / get ghost region decision
  local ghost_strategies =
    self:choose_ghost_strategies(typedfunc, typedfunc:all_accesses(),
                                 node_desc, partition_strategy)

  -- do second index lookup, on per-access basis
  local per_access_data = {}
  local field_accesses  = typedfunc:all_accesses()
  for f,access in pairs(field_accesses) do
    local relation = f:Relation()
    -- index 1 (apply node-local partitioning strategy)
    local local_partition = self.local_ghost_index:lookup {
      typedfunc           = typedfunc,
      relation            = relation,
      node_type           = node_desc,
      partition_strategy  = partition_strategy,
    }
    -- index 2 (apply ghost strategy)
    local lpart_ghost = self.local_ghost_index:lookup {
      rel_local_partition   = local_partition,
      ghost_strategy        = ghost_strategies[access],
    }
    -- index 3 (translate to legion data)
    local legion_data = self.local_ghost_index:lookup {
      local_ghost_partition = lpart_ghost,
      node_id               = node_id,
      proc_id               = proc_id,
    }
    per_access_data[access] = legion_data
  end

  return per_access_data
end



-------------------------------------------------------------------------------
--[[ Planning Objects                                                      ]]--
-------------------------------------------------------------------------------

-- Represents the choice of how to partition data within a node
local PartitionStrategy   = {}
PartitionStrategy.__index = PartitionStrategy
local GPU_Only = setmetatable({},PartitionStrategy)
local CPU_Only = setmetatable({},PartitionStrategy)

-- Represents choice of strategy for handling the ghost-cells
local GhostStrategy   = {}
GhostStrategy.__index = GhostStrategy
local GhostDepth = Util.memoize(function(k)
  return setmetatable({ depth = k },GhostStrategy)
end)

-------------------------------------------------------------------------------
--[[ Planner Implementation:                                               ]]--
-------------------------------------------------------------------------------

function Planner:choose_partition_strategy(typedfunc, node_desc)
  -- dumb default for development
  return CPU_Only
end

function Planner:choose_ghost_strategies(
  typedfunc, field_accesses, node_desc, partition_strategy
)
  -- CAN'T QUITE HAVE A DEFAULT;
  -- but I bet this works for plumbing through to begin with
  -- TODO: have a better policy here
  local choices = {}
  for f,access in pairs(field_accesses) do
    choices[access] = GhostDepth(2)
  end
  return choices
end

local all_partition_strategies = { CPU_Only }--, GPU_Only }
function Planner:refresh_local_ghost_index()
  -- handle all newly registered functions
  --  iterate over all possible types of nodes
  --  and strategies, and derive a local partition for each

  -- PER-FUNC,RELATION,NODE-TYPE,PARTITION-STRATEGY
  for _,typedfunc in ipairs(self.new_func_queue) do
    -- unpack all relations mentioned by the function
    local all_relations   = { [typedfunc:relation()] = true }
    local field_accesses  = typedfunc:all_accesses()
    for f,access in pairs(field_accesses) do
      all_relations[f:Relation()] = true
    end
  for relation,_ in pairs(all_relations) do
    local global_partition  = relation:_GetGlobalPartition()
  for _,node_type in ipairs(self.active_node_types) do
  for _,part_strategy in ipairs(all_partition_strategies) do
    -- compute parameters from the strategy
    -- TODO: COMPUTE PARAMETERS HERE

    -- create and cache a local partition for this combination
    -- of keys and strategy
    local local_partition = P.RelLocalPartition {
      global_partition  = global_partition,
      node_type         = node_type,
      -- some other parameters...
    }
    self.local_ghost_index:insert( local_partition, {
      typedfunc           = typedfunc,
      relation            = relation,
      node_type           = node_type,
      partition_strategy  = part_strategy,
    })

    -- potentially invalidate the partitions
    if not self.active_local_partitions[ local_partition ] then
      self.active_local_partitions[ local_partition ] = true
      self.new_local_partition_queue:insert( local_partition )
    end
  end end end end

  self.new_func_queue = newlist() -- re-validate the index
end

function Planner:rebuild_partitions()
  -- Code here should do the following
    -- * potentially reconstruct the partition tree
    -- * update the index caches

  -- TODO: Definitely needs code here

  self.new_local_partition_queue = newlist() -- re-validate the index
end




















