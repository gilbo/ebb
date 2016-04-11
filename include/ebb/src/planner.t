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
   
   Assumption 1) Number of nodes in global partitioning equals number of machine
     nodes, so that there is a one-to-one mapping between them.
     (May not be true in the beginnning, but planner makes decisions for
      partition nodes rather than machine nodes, and we assume Legion mapper
      makes reasonable decisions.)

   Assumption 2) Legion mapper does a reasonable job of mapping global partition
     nodes to machine nodes. Planner does not deal with this decision making
     right now.

    Index 1) partition_index
      for each  (function, relation, node_type, PartitionStrategy)
      produce   RelLocalPartition -- an object expressing a node-local
                                     partition of a relation

    Index 2) ghost_pattern_index
      for each  (RelLocalPartition, GhostStrategy)
      produce   LocalGhostPattern -- an object expressing a node-local
                                     template for sets of ghost-cells

    Index 3) legion_data_index
      for each  (LocalGhostPattern, ...(query_params)...)
      produce the relevant legion data (indexed by node id) to return
      NOTE: Might have to redesign this when we start using index partitions.
  

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

  FOR NOW: Assume a single node type to simplify agglomerating partitions, and
    one cpu per partition node for simplifying local partitioning..

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
      }, -- stores LocalGhostPattern
      legion_data_index     = Util.new_named_cache {
        'local_ghost_pattern', --'node_id', 'proc_id',
      }, -- stores legion_stuff...?

  -- "state_flags" (used to signal/control updates to indices)
      new_func_queue            = newlist(),
      new_local_partition_queue = newlist(),
      new_ghost_pattern_queue   = newlist(),
  },Planner)
end
local ThePlanner = NewPlanner()


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

-------------------------------------------------------------------------------

--  This function is called to get necessary launch parameters:
--  partition data so that a Legion launch can be performed.
--  FOR NOW: This assumes that there is only one node type.
--  Returns data first indexed by field accesses, then by nodes.
--  There is no processor indexing yet (see refresh_partition_index/
--  rebuild_partitions/ execute_partition in partitions.t).
--  Returned data is structured as:
--[[
--{
--  access_1 : {
--    partition : {per node regions}
--  },
--  access_2 : {
--    partition : {per node regions}
--  }
--}
--]]
function Exports.query_for_partitions(typedfunc)
  local self = ThePlanner

  -- make sure all indices are fresh before satisfying a query
  self:update_indices()

  -- make / get ghost region decision
  local per_access_data = {}

  -- NOTE: handle only one node type right now.
  -- If there are multiple node types, we need to figure out how to make
  -- decisions for each node type, and agglomerate the results.
  assert(#M.GetAllNodeTypes() == 1)
  local node_type = M.GetAllNodeTypes()[1]

  -- make / get partitioning decision
  local partition_strategy  =
    self:choose_partition_strategy(typedfunc, node_type)

  local per_access_data = {}

  -- choose what ghost strategies to use for each access
  local ghost_strategies =
    self:choose_ghost_strategies(typedfunc, typedfunc:all_accesses(),
                                 node_type, partition_strategy)
  -- do second index lookup, on per-access basis
  local field_accesses  = typedfunc:all_accesses()
  -- for each access, figure out what partitions to use
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
      --node_id             = node_id,
      --proc_id             = proc_id,
    }
    assert(legion_data, 'no legion data found')
    per_access_data[access] = legion_data
  end

  -- also add primary partition
  local prim_relation = typedfunc:relation()
  local prim_local_partition = self.partition_index:lookup {
    typedfunc                = typedfunc,
    relation                 = prim_relation,
    node_type                = node_type,
    partition_strategy       = partition_strategy,
  }
  assert(prim_local_partition, 'no primary local partition found')
  local prim_local_ghost_pattern = self.ghost_pattern_index:lookup {
    rel_local_partition          = prim_local_partition,
    ghost_strategy               = GhostDepth(0)
  }
  local prim_legion_data = self.legion_data_index:lookup {
    local_ghost_pattern  = prim_local_ghost_pattern
  }
  assert(prim_legion_data ~= nil)
  per_access_data.primary = prim_legion_data

  return per_access_data
end


-------------------------------------------------------------------------------
--[[ Planner Implementation:                                               ]]--
-------------------------------------------------------------------------------


function Planner:init()
  self.active_node_types = M.GetAllNodeTypes()

  self._is_initialized = true
end

-- Return a partition strategy to use given a typedfunction and nodetype
function Planner:choose_partition_strategy(typedfunc, node_type)
  -- dumb default for development
  return CPU_Only
end

-- Return a ghost strategy to use given a typedfunction, nodetype,
-- field accesses and partition strategy
function Planner:choose_ghost_strategies(
  typedfunc, field_accesses, node_type, partition_strategy
)
  -- returns only two ghost strategies, depth 0 for centered accesses and
  -- depth 2 for others (hopefully this works for now)
  local choices = {}
  for f,access in pairs(field_accesses) do
    if access:isCentered() then
      choices[access] = GhostDepth(0)
    else
      choices[access] = GhostDepth(2)
    end
  end
  return choices
end

-- If there are new typedfunctions (functions on relations), refresh partition
-- index.
-- If there are new local partitions (such as from new typedfunctions), refresh
-- ghost partition index that produces local ghost patterns.
-- If there are new ghost patterns (which happens if there are new local
-- partitions right now), rebuild partition trees over logical regions and
-- index space, and update legion data index.
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

-- Refresh partition index for all new typed functions, according to enumerated
-- partition strategies.
-- FOR NOW: This assumes that there is just one cpu per node (each cpu is a
-- different node).
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
          -- FOR NOW: assume we'll only use one cpu per-node

          -- FOR NOW: also assume this local partition is over all the nodes
          -- in global partition, since there is only one node type.

          -- create and cache a local partition for this combination of keys
          -- strategy, and record which nodes use this local partition
          local nodes    = terralib.newlist()  -- a list of node ids
          local blocking = global_partition:get_blocking()
          local ndims    = #blocking
          if ndims == 3 then
            for x = 1, blocking[1] do
              for y = 1, blocking[2] do
                for z = 1, blocking[3] do
                  nodes:insert({x, y, z})
            end end end
          else
            for x = 1, blocking[1] do
              for y = 1, blocking[2] do
                nodes:insert({x, y})
            end end
            assert(ndims == 2)
          end
          local local_partition = P.RelLocalPartition {
            rel_global_partition  = global_partition,
            node_type             = node_type,
            nodes                 = nodes
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
        end
      end
    end
  end

  self.new_func_queue = newlist() -- re-validate the index
end

local all_ghost_strategies = {
  GhostDepth(0), GhostDepth(1), GhostDepth(2)
}
-- Refresh ghost pattern index for all new RelLocalPartitions (such as from
-- refresh_partition_index), according to enumerated ghost strategies.
function Planner:refresh_ghost_pattern_index()
  -- handle all newly registered local_ghost_patterns

  -- PER-REL_LOCAL_PARTITION,GHOST_STRATEGY
  for _,local_partition in ipairs(self.new_local_partition_queue) do
  for _,ghost_strategy in ipairs(all_ghost_strategies) do
    -- TODO: COMPUTE PARAMETERS HERE

    local local_ghost_pattern = P.LocalGhostPattern {
      rel_local_partition = local_partition,
      -- Gonna need to change this parameter around in the future...
      uniform_depth       = ghost_strategy.depth,
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

local function get_legion_ghost_regions()
end

-- Rebuild legion region trees and legion_data_index, for newly computed
-- LocalGhostPatterns (such as in refresh_ghost_pattern_index).
function Planner:rebuild_partitions()
  -- Code here should do the following
    -- * potentially reconstruct the partition tree
    -- * update the final index appropriately
  
  -- make sure that legion data is built for these local partitions
  for local_partition,_ in pairs(self.active_local_partitions) do
    local_partition:execute_partition()
  end

  -- make sure that legion data is built for these local ghost patterns
  for _,local_ghost_pattern in ipairs(self.new_ghost_pattern_queue) do
    local_ghost_pattern:execute_partition()
  end

  -- get legion ghost regions (for all nodes), for each local ghost pattern
  for _,local_ghost_pattern in ipairs(self.new_ghost_pattern_queue) do
    local data = {
      partition = local_ghost_pattern:get_ghost_legion_subregions(),
    }
    self.legion_data_index:insert(data,
      { local_ghost_pattern = local_ghost_pattern }
    )
  end

  self.new_ghost_pattern_queue = newlist() -- re-validate the index
end
