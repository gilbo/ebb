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


local Util  = require 'ebb.src.util'
local P     = require 'ebb.src.partitions'
local M     = require 'ebb.src.machine'

-------------------------------------------------------------------------------

local newlist = terralib.newlist

-------------------------------------------------------------------------------

--**--**--**--**--**--**--**--**--
--  How Does the Planner Work?  --
--**--**--**--**--**--**--**--**--
--[[

  The planner has two key external interfaces:

    1) note_launch(...) is called to inform the planner that some function
            is going to be executed.  By making this a separate function from
            the second (query) we can allow callers to notify the planner
            about a sequence of functions it is about to call before it
            starts trying to actually do so.
    2) query_for_partitions(...) is called to retreive the Legion runtime
            data that is necessary to actually execute a task launch across
            the machine.  Implicit in this retreieved data is a number of
            decisions about *how* the task should be executed.

  The planner abstracts the question of *HOW* a task should be launched into
  special objects called STRATEGIES.  There are currently two strategic
  decisions the planner must make when launching a function:

    1) PartitionStrategy -- how data should be partitioned within a given
            node of the supercomputer.  Examples are to place all the data
            (and hence all the computation) on the GPU or all the
            data/compute on the CPU, or to balance between the two.
    2) GhostStrategy -- how latency buffers, commonly called "ghost cells"
            for data on neighboring nodes should be managed.  Examples are to
            keep ghost cells for all cells 1,2,... hops away, or perhaps
            to only keep ghost cells for a single hop in cardinal directions.

    (note: these strategies neither encode the details of how they should be
            achieved, nor how a decision should be made about which strategy
            to pursue in any given case.  Instead they represent the
            interface between those two concerns. )

  Given these decisions, the planner's implementation becomes structured
  around a "pipeline" of *INDICES* that progressively combine strategic
  decisions with input to derive the appropriate launch data.
    
    Query Pipeline Input)
      function & node_type (to launch on)

    Decomposition 1)
      extract all relations accessed by the function

    Decomposition 2)
      extract all field-accesses performed by the function

    Decision 1)
      given a function & node_type, decide what PartitionStrategy to use

    Decision 2)
      given a function, all field-accesses, node_type, and the resulting
              strategy of the first decision, decide what GhostStrategy
              to use.

    Index 1)
      for each  (function, relation, node_type, PartitionStrategy)
      produce   RelLocalPartition -- an object expressing a node-local
                                     partition of a relation

    Index 2)
      for each  (RelLocalPartition, GhostStrategy)
      produce   LocalGhostPattern -- an object expressing a node-local
                                     template for sets of ghost-cells

    Index 3)
      for each  (LocalGhostPattern, ...(query_params)...)
      produce   the relevant legion data to return
  

  Making Decisions: The planner chooses how to make decisions by collecting
    statistics about the launches that are made.  By decoupling these
    decisions into the choice of & execution of strategies, we hope that
    the decision-making logic can be decoupled and simplified.  Early
    implementations will keep decision making limited to simple policies.

  Maintaining Indices: By decoupling the pipeline into a series of indices
    we introduce 2 key points / objects (RelLocalPartition &
    LocalGhostPartition) which we can memoize, thereby removing/reducing
    dependency.  Since the actual partition of data on the machine as
    observed by Legion only needs to be updated when Index 3 changes, this
    also serves to reduce the frequency with which that partition needs
    to change.

--]]

-------------------------------------------------------------------------------
--[[ The Planner: (Data Representation)                                    ]]--
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
      _is_initialized = false,

  -- sets of active objects
      active_node_types         = {},
      active_funcs              = {},
      -- the following two sets correspond to memoized intermediaries
      -- in the index pipeline
      active_local_partitions   = {},
      active_ghost_patterns     = {},

  -- statistics & other observed data:
      f_call_count    = {}, -- set of invoked funcs with invocation counts

  -- maintained indices (the query pipeline)
      partition_index       = Util.new_named_cache({
        'typedfunc', 'relation', 'node_type', 'partition_strategy',
      }), -- stores RelLocalPartition objects
      ghost_pattern_index     = Util.new_named_cache {
        'rel_local_partition', 'ghost_strategy',
      }, -- stores LocalGhostPattern objects
      legion_data_index     = Util.new_named_cache {
        'local_ghost_pattern', 'node_id', 'proc_id',
      }, -- stores legion_stuff...?

  -- "state_flags" (used to signal/control updates to indices)
      new_func_queue            = newlist(),
      new_local_partition_queue = newlist(),
      new_ghost_pattern_queue   = newlist(),
  },Planner)
end
local ThePlanner = NewPlanner()

-------------------------------------------------------------------------------
--[[ Planner Interface:                                                    ]]--
-------------------------------------------------------------------------------

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

  if not self._is_initialized then self:init() end

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
function Exports.query_for_partitions(typedfunc, node_type, node_id, proc_id)
  local self = ThePlanner

  -- THIS IS A HACK.  Do we even want to supply node and proc ids?
  --                  Would it be better to just supply node and processor
  --                  type?
  --                  This depends a lot on how the actual legion launch is
  --                  done...
  node_id = node_id or 1
  proc_id = proc_id or 1
  node_type = node_type or M.SingleCPUNode

  -- make sure all indices are fresh before satisfying a query
  self:update_indices()

  -- make / get partitioning decision
  local partition_strategy  =
    self:choose_partition_strategy(typedfunc, node_type)

  -- make / get ghost region decision
  local ghost_strategies =
    self:choose_ghost_strategies(typedfunc, typedfunc:all_accesses(),
                                 node_type, partition_strategy)

  -- do second index lookup, on per-access basis
  local per_access_data = {}
  local field_accesses  = typedfunc:all_accesses()
  for f,access in pairs(field_accesses) do
    local relation = f:Relation()
    -- index 1 (apply node-local partitioning strategy)
    local local_partition = self.partition_index:lookup {
      typedfunc           = typedfunc,
      relation            = relation,
      node_type           = node_type,
      partition_strategy  = partition_strategy,
    }
    assert(local_partition, 'no local partition found')
    -- index 2 (apply ghost strategy)
    local local_ghost_pattern = self.ghost_pattern_index:lookup {
      rel_local_partition   = local_partition,
      ghost_strategy        = ghost_strategies[access],
    }
    if not local_ghost_pattern:supports( access:getstencil() ) then
      error('INTERNAL: ghost pattern does not support required stencil '..
            ' access pattern.')
    end
    assert(local_ghost_pattern, 'no ghost pattern found')
    -- index 3 (translate to legion data)
    local legion_data = self.legion_data_index:lookup {
      local_ghost_pattern = local_ghost_pattern,
      node_id             = node_id,
      proc_id             = proc_id,
    }
    assert(legion_data, 'no legion data found')
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


-- MAYBE this should be perfomed automatically on startup instead of
-- being exposed at the module interface? dunno
--    e.g. 1) simply do this work in-line
--    e.g. 2) call this on the first note_launch() call
function Exports.register_node_types(node_types)
  local self = ThePlanner

  self.active_node_types = newlist()
  for i,d in ipairs(node_types) do self.active_node_types[i] = d end
end

function Planner:init()
  self.active_node_types = newlist{ M.SingleCPUNode }

  self._is_initialized = true
end



function Planner:choose_partition_strategy(typedfunc, node_type)
  -- dumb default for development
  return CPU_Only
end

function Planner:choose_ghost_strategies(
  typedfunc, field_accesses, node_type, partition_strategy
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

function Planner:update_indices()
  if #self.new_func_queue > 0 then
    self:refresh_partition_index()
  end
  if #self.new_local_partition_queue > 0 then
    self:refresh_ghost_pattern_index()
  end
  if #self.new_ghost_pattern_queue > 0 then
    self:rebuild_partitions()
  end
end

local all_partition_strategies = { CPU_Only }--, GPU_Only }
function Planner:refresh_partition_index()
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
  for _,partition_strategy in ipairs(all_partition_strategies) do
    -- compute parameters from the strategy
    -- TODO: COMPUTE PARAMETERS HERE

    -- create and cache a local partition for this combination
    -- of keys and strategy
    local local_partition = P.RelLocalPartition {
      rel_global_partition  = global_partition,
      node_type             = node_type,
      -- some other parameters...
    }
    self.partition_index:insert( local_partition, {
      typedfunc           = typedfunc,
      relation            = relation,
      node_type           = node_type,
      partition_strategy  = partition_strategy,
    })

    -- potentially invalidate the partitions
    if not self.active_local_partitions[ local_partition ] then
      self.active_local_partitions[ local_partition ] = true
      self.new_local_partition_queue:insert( local_partition )
    end
  end end end end

  self.new_func_queue = newlist() -- re-validate the index
end

local all_ghost_strategies = {
  GhostDepth(0), GhostDepth(1), GhostDepth(2)
}
function Planner:refresh_ghost_pattern_index()
  -- handle all newly registered local_ghost_patterns

  -- PER-REL_LOCAL_PARTITION,GHOST_STRATEGY
  for _,local_partition in ipairs(self.new_local_partition_queue) do
  for _,ghost_strategy in ipairs(all_ghost_strategies) do
    -- TODO: COMPUTE PARAMETERS HERE

    local local_ghost_pattern = P.LocalGhostPattern {
      rel_local_partition = local_partition,
      -- params...
    }
    self.ghost_pattern_index:insert( local_ghost_pattern, {
      rel_local_partition   = local_partition,
      ghost_strategy        = ghost_strategy,
    })

    -- invalidate the partitions
    -- b/c of construction we're guaranteed this is new
    self.active_ghost_patterns[ local_ghost_pattern ] = true
    self.new_ghost_pattern_queue:insert( local_ghost_pattern )
  end end

  self.new_local_partition_queue = newlist() -- re-validate the index
end

-------------------------------------------------------------------------------

-- initialization of private Legion partition data
local function get_partition_store(planner)
  if planner._partition_storage then return planner._partition_storage end

  planner._partition_storage = {
    trees = {} -- maps RelGlobalPartitions to partition objects
  }
  return planner._partition_storage
end

function Planner:rebuild_partitions()
  -- Code here should do the following
    -- * potentially reconstruct the partition tree
    -- * update the final index appropriately

  local pstore = get_partition_store(self)

  -- writing as a single-time attempt to partition...
  -- needs to be edited into better shape
  for local_partition,_ in pairs(self.active_local_partitions) do
    local_partition:execute_partition()
  end

  -- TODO: Definitely needs code here
  --local relations


  -- TODO: STUB FOR DEVELOPMENT
  for _,local_ghost_pattern in ipairs(self.new_ghost_pattern_queue) do
    local data = {}
    self.legion_data_index:insert( data, {
      local_ghost_pattern = local_ghost_pattern,
      node_id             = 1,
      proc_id             = 1,
    })
  end

  self.new_ghost_pattern_queue = newlist() -- re-validate the index
end




















