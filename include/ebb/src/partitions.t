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

local newlist = terralib.newlist

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

-- TODO: check that _n_nodes matches number of machine nodes
Exports.RelGlobalPartition = Util.memoize(function(
  relation, nX, nY, nZ
)
  assert(R.is_relation(relation))
  assert(nX and nY)
  return setmetatable({
    _n_nodes     = nX * (nY or 1) * (nZ or 1),
    _blocking    = { nX, nY, nZ },
    _n_x         = nX,
    _n_y         = nY,
    _n_z         = nZ,
    _lreg        = relation._logical_region_wrapper,
    _rel_dims    = relation:Dims(),
    _lpart       = nil,  -- legion logical partition
    --_relation    = relation,
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
    _nodes        = args.nodes,  -- list of node ids this paritioning is used for
    _lregs        = nil,         -- list of regions for nodes
    --_split_tree   = args.split_tree,
  },RelationLocalPartition)
end)

local function linearize_idx_3d(i,j,k,nx,ny,nz)
  return i*ny*nz + j*nz + k
end
local function linearize_idx_2d(i,j,nx,ny)
  return i*ny + j
end

function RelationGlobalPartition:get_n_nodes()
  return self._n_nodes
end

function RelationGlobalPartition:nDims()
  return #self._rel_dims
end

function RelationGlobalPartition:get_subrects()
  local nX, nY, nZ    = self._n_x, self._n_y, self._n_z
  local dX, dY, dZ    = unpack(self._rel_dims)
  local is3d          = #self._rel_dims == 3

  local subrects   = newlist()
  -- loop to fill out the partition coloring
  for i=1,nX do
    local xrange  = { math.floor( dX * (i-1) / nX ),
                      math.floor( dX * i / nX ) - 1 }
    for j=1,nY do
      local yrange  = { math.floor( dY * (j-1) / nY ),
                        math.floor( dY * j / nY ) - 1 }
      if not is3d then
        subrects:insert( Util.NewRect2d( xrange,yrange ) )
      else for k=1,nZ do
        local zrange  = { math.floor( dZ * (k-1) / nZ ),
                          math.floor( dZ * k / nZ ) - 1 }
        subrects:insert( Util.NewRect3d( xrange,yrange,zrange ) )
      end end
    end
  end

  return subrects
end

function RelationGlobalPartition:execute_partition()
  if self._lpart then return end  -- make idempotent
  self._lpart       = newlist()
  local lreg        = self._lreg
  self._lpart       = lreg:CreateDisjointPartition(self:get_subrects())
end

function RelationGlobalPartition:get_legion_partition()
  return self._lpart
end

function RelationGlobalPartition:subregions()
  return self._lpart:subregions()
end

function RelationGlobalPartition:TEMPORARY_get_subregion_for_node(node)
  return self:subregions()[node]
end

function RelationLocalPartition:get_global_partition()
  return self._global_part
end

function RelationLocalPartition:nDims()
  return self:get_global_partition():nDims()
end

function RelationLocalPartition:execute_partition()
  if self._lregs then return end  -- make idempotent

  self._global_part:execute_partition()

  self._lregs = {}
  -- FOR NOW: Assume there is just one processor partition per node.
  -- Copy over partitions for all supported nodes.
  for i, nid in ipairs(self._nodes) do
    self._lregs[i] = self._global_part:TEMPORARY_get_subregion_for_node(i)
  end
end

function RelationLocalPartition:get_legion_partition()
  error("Local partitions implemented with subregions right now " ..
        "that emulate index space. Please use " ..
        "TEMPORARY_get_legion_subregions()", 2)
end

-- Returns a list of partition regions, indexed by node.
-- No porcessor indexing yet.
function RelationLocalPartition:TEMPORARY_get_subregions()
  return self._lregs
end


-------------------------------------------------------------------------------
--[[ Global / Local Partition Ghosts:                                      ]]--
-------------------------------------------------------------------------------

local GlobalGhostPattern    = {}
GlobalGhostPattern.__index  = GlobalGhostPattern

local LocalGhostPattern     = {}
LocalGhostPattern.__index   = LocalGhostPattern


-- TODO: QUESTION: Why is this there?
local NewGlobalGhostPattern = Util.memoize_named({
  'rel_global_partition',
  'uniform_depth',
}, function(args)
  assert(args.rel_global_partition)
  assert(args.uniform_depth)
  return setmetatable({
    _rel_global_partition = args.rel_global_partition,
    _depth                = args.uniform_depth,
    _lpart                = nil,
  },GlobalGhostPattern)
end)

Exports.LocalGhostPattern   = Util.memoize_named({
  'rel_local_partition', --'params',
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
    _lregs                = nil,  -- first indexed by node, then by ghost number
  },LocalGhostPattern)
end)

-- set up ghost regions for each node
function GlobalGhostPattern:execute_partition()
  -- do nothing
end

function LocalGhostPattern:get_local_partition()
  return self._rel_local_partition
end

-- Return a list of ghost regions first indexed by node and
-- then by ghost position
function LocalGhostPattern:get_all_subrects()
  local depth = self._depth

  local local_partition = self:get_local_partition()

  local all_rects = newlist()
  local all = {-math.huge,math.huge}
  local is3d = local_partition:nDims() == 3

  if is3d then
    for i,reg in ipairs(local_partition:TEMPORARY_get_subregions()) do
      local node_rects = newlist()
      if depth ~= 0 then
        local rect = reg:get_rect()
        local xlo,ylo,zlo,xhi,yhi,zhi = rect:mins_maxes()
        local b = { 
                     { 
                        Util.NewRect3d({-math.huge,xlo+depth},all,all),
                        Util.NewRect3d({ xlo+depth,xhi-depth},all,all),
                        Util.NewRect3d({ xhi-depth,math.huge},all,all)
                     },
                     { 
                        Util.NewRect3d(all,{-math.huge,ylo+depth},all),
                        Util.NewRect3d(all,{ ylo+depth,yhi-depth},all),
                        Util.NewRect3d(all,{ yhi-depth,math.huge},all)
                     },
                     { 
                        Util.NewRect3d(all,all,{-math.huge,zlo+depth}),
                        Util.NewRect3d(all,all,{ zlo+depth,zhi-depth}),
                        Util.NewRect3d(all,all,{ zhi-depth,math.huge})
                     },
                  }
        for x = 1,3 do for y = 1,3 do for z = 1,3 do
              node_rects:insert(rect:clip(b[1][x]):clip(b[2][y]):clip(b[3][z]))
        end end end  -- x, y, z
      end
      all_rects:insert(node_rects)
    end
  else
    for i,reg in ipairs(local_partition:TEMPORARY_get_subregions()) do
      local node_rects = newlist()
      if depth ~= 0 then
        local rect = reg:get_rect()
        local xlo,ylo,xhi,yhi = rect:mins_maxes()
        local b = {
                     {
                        Util.NewRect2d({-math.huge,xlo+depth},all),
                        Util.NewRect2d({ xlo+depth,xhi-depth},all),
                        Util.NewRect2d({ xhi-depth,math.huge},all)
                     },
                     {
                        Util.NewRect2d(all,{-math.huge,ylo+depth}),
                        Util.NewRect2d(all,{ ylo+depth,yhi-depth}),
                        Util.NewRect2d(all,{ yhi-depth,math.huge})
                      }
                    }
        for x = 1,3 do for y = 1,3 do
              node_rects:insert(rect:clip(b[1][x]):clip(b[2][y]))
        end end  -- x, y
      end
      all_rects:insert(node_rects)
    end
  end
  return all_rects
end

function LocalGhostPattern:supports(stencil)
  return true -- TODO: implement actual check
end

function LocalGhostPattern:get_legion_partition()
  error("Local partitions implemented with subregions right now " ..
        "that emulate index space. Please use " ..
        "TEMPORARY_get_legion_subregions()", 2)
  -- return legion partitions for each node
  if self._depth == 0 then
    return self._rel_local_partition:get_legion_partition()
  else
    -- TODO: HACK: for now, just return nil for non-zero stencils
    -- This will make legionwrap add entire logical region instead of
    -- a partition.
    -- We should use computed ghost regions instead
    return nil
  end
end

-- set up ghost regions internal to a node
function LocalGhostPattern:execute_partition()
  if self._lregs then return end  -- make idempotent

  self._global_pattern:execute_partition()

  local all_subrects = self:get_all_subrects()

  local disjoint_regions = self:get_local_partition():TEMPORARY_get_subregions()
  self._lregs  = newlist()
  -- FOR NOW: Assume there is just one processor partition per node.
  -- Set up ghost regions for the one partition for every node.
  for n,ghosts in ipairs(all_subrects) do
    local lregs = false
    if #ghosts ~= 0 then
      local subrects = all_subrects[n]
      lregs          = false
      -- TODO: for now, return false for non-zero ghosts, so legionwrap adds
      -- the entire logical region.
      --lregs          = disjoint_regions[n]:CreateOverlappingPartition(subrects)
    else
      lregs          = disjoint_regions[n] 
    end
    self._lregs:insert(lregs)
  end
end

-- Returns a 2 level list of aliased partition regions first indexed by node
-- and then indexed by ghost number.
function LocalGhostPattern:TEMPORARY_get_legion_subregions()
  return self._lregs
end
