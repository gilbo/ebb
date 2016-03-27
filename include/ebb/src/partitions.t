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
package.loaded["ebb.src.partitions"] = Exports

local use_legion = not not rawget(_G, '_legion_env')
local use_single = not use_legion


local LE, legion_env, LW
if use_legion then
  LE            = rawget(_G, '_legion_env')
  legion_env    = LE.legion_env[0]
  LW            = require 'ebb.src.legionwrap'
end

local R   = require 'ebb.src.relations'

local Util            = require 'ebb.src.util'
local Machine         = require 'ebb.src.machine'
local SingleCPUNode   = Machine.SingleCPUNode

------------------------------------------------------------------------------


-------------------------------------------------------------------------------
--[[ SplitTrees                                                            ]]--
-------------------------------------------------------------------------------

local SplitTree   = {}
SplitTree.__index = SplitTree

local SplitNode   = setmetatable({},SplitTree)
SplitNode.__index = SplitNode

local SplitLeaf   = setmetatable({},SplitTree)
SplitLeaf.__index = SplitLeaf

local function is_splitleaf(obj) return getmetatable(obj) == SplitLeaf end
local function is_splitnode(obj) return getmetatable(obj) == SplitNode end
local function is_splittree(obj)
  return is_splitleaf(obj) or is_splitnode(obj)
end
Exports.is_splitleaf = is_splitleaf
Exports.is_splitnode = is_splitnode
Exports.is_splittree = is_splittree

Exports.NewSplitNode = Util.memoize(function(
  axis_i, split_percent, leftnode, rightnode
)
  return setmetatable({
    axis  = axis_i,
    frac  = split_percent,
    left  = leftnode,
    right = rightnode,
  }, SplitNode)
end)

-- could be extended?
Exports.NewSplitLeaf = Util.memoize(function(proc_type)
  return setmetatable({ proc = proc_type }, SplitLeaf)
end)





-------------------------------------------------------------------------------
--[[ Relation Local / Global Partitions:                                   ]]--
-------------------------------------------------------------------------------

local RelationGlobalPartition   = {}
RelationGlobalPartition.__index = RelationGlobalPartition

local RelationLocalPartition    = {}
RelationLocalPartition.__index  = RelationLocalPartition

Exports.RelGlobalPartition = Util.memoize(function(
  relation, nX, nY, nZ
)
  assert(R.is_relation(relation))
  assert(nX and nY)
  return setmetatable({
    _n_nodes    = nX * (nY or 1) * (nZ or 1),
    _dims       = { nX, nY, nZ },
    _n_x        = nX,
    _n_y        = nY,
    _n_z        = nZ,
    _lreg       = relation._logical_region_wrapper,
    --_relation   = relation,
  },RelationGlobalPartition)
end)


Exports.RelLocalPartition = Util.memoize_named({
  'rel_global_partition', 'node_type'--, 'split_tree',
}, function(args)
  assert(args.rel_global_partition)
  assert(args.node_type)
  return setmetatable({
    _global_part  = args.rel_global_partition,
    _node_type    = args.node_type,
    -- TODO: ADD WAYS OF USING MULTIPLE PROCESSORS ON A NODE
    --_split_tree   = args.split_tree,
  },RelationLocalPartition)
end)

function RelationGlobalPartition:execute_partition()
  if self._lpart then return end -- make idempotent
  local lreg  = self._lreg
  local lpart = lreg:CreateDisjointBlockPartition(self._dims)
  self._lpart = lpart
end

function RelationGlobalPartition:get_legion_partition()
  return self._lpart
end

function RelationGlobalPartition:subregions()
  return self._lpart:subregions()
end

function RelationLocalPartition:execute_partition()
  if self._lpart then return end -- make idempotent

  self._global_part:execute_partition()
  --local gpart = self._global_part:get_legion_partition()
  -- don't do sub-partition right now...
--  for _,p in ipairs(gpart:subregions()) do
--    local i,j,k = unpack(p.idx)
--    if true then -- if node-type matches... TODO
--
--    end
--  end
end
function RelationLocalPartition:get_legion_partition()
  return self._global_part:get_legion_partition()
end


-------------------------------------------------------------------------------
--[[ Global / Local Partition Ghosts:                                      ]]--
-------------------------------------------------------------------------------

local GlobalGhostPattern    = {}
GlobalGhostPattern.__index  = GlobalGhostPattern

local LocalGhostPattern     = {}
LocalGhostPattern.__index   = LocalGhostPattern


local NewGlobalGhostPattern = Util.memoize_named({
  'rel_global_partition',
  'uniform_depth',
}, function(args)
  assert(args.rel_global_partition)
  assert(args.uniform_depth)
  return setmetatable({
    _rel_global_partition = args.rel_global_partition,
    _depth                = args.uniform_depth,
  },GlobalGhostPattern)
end)

Exports.LocalGhostPattern   = Util.memoize_named({
  'rel_local_partition', --'params',
  -- NEED TO CHANGE THIS IN THE FUTURE
  'uniform_depth',
}, function(args)
  assert(args.rel_local_partition)
  assert(args.uniform_depth)
  local global_pattern = NewGlobalGhostPattern{
    rel_global_partition  = args.rel_local_partition._global_part,
    uniform_depth         = args.uniform_depth,
  }
  return setmetatable({
    _rel_local_partition  = args.rel_local_partition,
    _depth                = args.uniform_depth,
    _global_pattern       = global_pattern,
  },LocalGhostPattern)
end)

function LocalGhostPattern:supports(stencil)
  return true -- TODO: implement actual check
end



-- set up ghost regions for each node
function GlobalGhostPattern:execute_partition()
  if self._ghost_partitions then return end -- make idempotent

  local regs = self._rel_global_partition:subregions()
  for _,p in ipairs(regs) do
    local reg, i,j,k = p.region, unpack(p.idx)
    
  end
end

-- set up ghost regions internal to a node
function LocalGhostPattern:execute_partition()
  -- basically do nothing for now...
  self._global_pattern:execute_partition()

  --if self._lpart then return end -- make idempotent
  --local lreg  = self._lreg
  --local lpart = lreg:CreateDisjointBlockPartition(self._dims)
  --self._lpart = lpart
end

















