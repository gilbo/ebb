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
  assert(args.nodes)
  local lin_nodes = terralib.newlist()
  for i, nid in args.nodes do
    lin_nodes[i] = args.rel_global_partition:get_linearized_node_id(nid)
  end
  return setmetatable({
    _global_part  = args.rel_global_partition,
    _node_type    = args.node_type,
    -- list of node ids this paritioning is used for
    _nodes        = terralib.newlist({unpack(nodes)}),  -- make a copy
    _lin_nodes    = lin_nodes,
    -- list of regions for nodes, indexed by linearized node id
    _lregs        = nil,
  },RelationLocalPartition)
end)

function RelationGlobalPartition:get_n_nodes()
  return self._n_nodes
end

function RelationGlobalPartition:get_blocking()
  return {self._n_x, self._n_y, self._n_z}
end

function RelationGlobalPartition:nDims()
  return #self._rel_dims
end

-- wrap around for global boundaries
function RelationGlobalPartition:get_neighbor_id(nid, offset)
  local ndims  = self:nDims()
  local nbr_id = {}
  for d = 1,ndims do
    nbr_id[d] = (nid[d] + offset[d]) % self._blocking[d]
  end
  return nbr_id
end

function RelationGlobalPartition:get_linearized_node_id(nid)
  local ndims = self:nDims()
  if ndims == 3 then
    return (nid[3]-1)*self._ny_*self._n_z + (nid[2]-1)*self._n_z + nid[3]
  else
    assert(ndims == 2)
    return (nid[1]-1)*self._n_y + nid[2]
  end
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

function RelationGlobalPartition:get_subregion_for_node(node)
  local lin_id = self:get_linearized_nod_id(node)
  return self:subregions()[lin_id]
end

function RelationLocalPartition:get_global_partition()
  return self._global_part
end

function RelationLocalPartition:nDims()
  return self:get_global_partition():nDims()
end

-- caller should not mutate this list
function RelationLocalPartition:get_node_ids()
  return self._nodes
end

-- caller should not mutate this list
function RelationLocalPartition:get_linearized_node_ids()
  return self._lin_nodes
end

function RelationLocalPartition:get_neighbor_id(nid, offset)
  return self:get_global_partition():get_neighbor_id()
end

function RelationLocalPartition:execute_partition()
  if self._lregs then return end  -- make idempotent

  self._global_part:execute_partition()

  self._lregs = {}
  -- FOR NOW: Assume there is just one processor partition per node.
  -- Copy over partitions for all supported nodes.
  for i, nid in ipairs(self._nodes) do
    self._lregs[nid] = self._global_part:get_subregion_for_node(nid)
  end
end

function RelationLocalPartition:get_legion_partition()
  error("Local partitions implemented with subregions right now " ..
        "that emulate index space. Please use " ..
        "get_legion_subregions()", 2)
end

-- Returns a list of partition regions, indexed by linearized node id.
-- No porcessor indexing yet.
function RelationLocalPartition:get_subregions()
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
    rel_global_partition  = args.rel_local_partition:get_global_partition(),
    uniform_depth         = args.uniform_depth,
  }
  return setmetatable({
    _rel_local_partition  = args.rel_local_partition,
    _depth                = args.uniform_depth,
    _global_pattern       = global_pattern,
    -- following two are indexed first by node number and then by ghost number
    _aliased_lregs         = nil,  -- aliased, but within home local partition
  },LocalGhostPattern)
end)

-- set up ghost regions for each node
function GlobalGhostPattern:execute_partition()
  -- do nothing
end

function LocalGhostPattern:get_local_partition()
  return self._rel_local_partition
end

function LocalGhostPattern:get_node_ids()
  return selfLget_local_partition():get_node_ids()
end

local ghost_regions_layout = {
  dim_2 = {9, 3, 1},
  dim_3 = {3, 1}
}
Exports.ghost_regions_layout = ghost_regions_layout

-- aliased ghost regions that belong inside a disjoint local partition
function Exports.ComputeAliasedGhostRegionNum(id)
  if #id == 3 then
    return (id[1]+1)*ghost_regions_layout.dim_3[1] +
           (id[2]+1)*ghost_regions_layout.dim_3[2] +
           (id[3]+1)*ghost_regions_layout.dim_3[3] + 1

  else
    return (id[1]+1)*ghost_regions_layout.dim_2[1] +
           (id[2]+1)*ghost_regions_layout.dim_2[2] + 1
  end
end
local ComputeAliasedGhostRegionNum = Exports.ComputeAliasedGhostRegionNum
-- ghost regions with a non-zero stencil, from neighboring partitions
Exports.ComputeGhostRegionNum      = Exports.ComputeAliasedGhostRegionNum

-- Return a list of ghost regions first indexed by node and
-- then by ghost position
function LocalGhostPattern:get_all_subrects()
  local depth = self._depth

  local local_partition = self:get_local_partition()

  local all_rects = newlist()
  local all = {-math.huge,math.huge}
  local is3d = local_partition:nDims() == 3

  if is3d then
    for nid,reg in pairs(local_partition:get_subregions()) do
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
              local rnum = ComputeAliasedGhostRegionNum {x-2, y-2, z-2 }
              node_rects[rnum] = rect:clip(b[1][x]):clip(b[2][y]):clip(b[3][z])
        end end end  -- x, y, z
      end
      all_rects[nid] = node_rects
    end
  else
    for nid,reg in pairs(local_partition:get_subregions()) do
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
            local rnum = ComputeAliasedGhostRegionNum {x-2, y-2}
            node_rects[rnum] = rect:clip(b[1][x]):clip(b[2][y])
        end end  -- x, y
      end
      all_rects[nid] = node_rects
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
        "get_legion_subregions()", 2)
end

-- set up ghost regions internal to a node
function LocalGhostPattern:execute_partition()
  if self._ghost_regs then return end  -- make idempotent

  self._global_pattern:execute_partition()

  local all_subrects  = self:get_all_subrects()

  local disjoint_regions = self:get_local_partition():get_subregions()
  self._aliased_lregs = {}
  -- FOR NOW: Assume there is just one processor partition per node.
  -- Set up ghost regions for the one partition for every node.
  for nid,ghosts in pairs(all_subrects) do
    local aliased_lregs = nil
    if #ghosts ~= 0 then
      local subrects = all_subrects[nid]
      aliased_lregs  = disjoint_regions[nid]:CreateOverlappingPartition(subrects)
    else
      aliased_lregs  = disjoint_regions[nid] 
    end
    self._aliased_lregs[nid] = aliased_lregs
  end
end

function LocalGhostPattern:get_aliased_legion_subregions(node_id)
  assert(self._aliased_lregs)
  local global_partition = self:get_local_partition():get_global_partition()
  local lin_id           = global_partition:get_linearized_id(node_id)
  return self._aliased_lregs[lin_id]
end

function LocalGhostPattern:get_aliased_legion_subregion(node_id, ghost_id)
  assert(self._aliased_lregs)
  local global_partition = self:get_local_partition():get_global_partition()
  local lin_id           = global_partition:get_linearized_id(node_id)
  local gnum             = ComputeAliasedGhostRegionNum(ghost_id)
  return self._aliased_lregs[lin_id][gnum]
end

-- FOR NOW: assume that there is no need to look up RelLocalPartition for
-- a neighbor node id. This is true as long as there is only one node type.
-- NOTE:
-- With multiple node types, we need to do a lookup the correct LocalPartition
-- and GhostPattern given a neighbor node id,
-- and we also need to update execute partitions to  generate correct ghosts
-- for neighbors.
function LocalGhostPattern:get_ghost_legion_subregions()
  local data             = terralib.newlist()
  local local_partition  = self:get_local_partition()
  local node_ids         = local_partition:get_node_ids()
  local is3d             = local_partition:nDims() == 3
  if is3d then
    for i, nid in ipairs(node_ids) do
      local ghost_lregs      = terralib.newlist()
      if self._depth == 0 then
        ghost_lregs[1]      = self:get_aliased_legion_subregions(nid)[1]
      else
        for x = -1,1 do for y = -1,1 do for z = -1,1 do
          local nbr_id      = local_partition:get_neighbor_id(nid, {x, y, z})
          local rnum        = ComputeGhostRegionNum({x, y, z})
          ghost_lregs[rnum] =
            self:get_aliased_legion_subregion(nbr_id, {-x, -y, -z})
        end end end
      end
    end
    data[nid]             = ghost_lregs 
  else
    for i, nid in ipairs(node_ids) do
      local ghost_lregs      = terralib.newlist()
      if self._depth == 0 then
        ghost_lregs[1]      = self:get_aliased_legion_subregions(nid)[1]
      else
        for x = -1,1 do for y = -1,1 do
          local nbr_id      = local_partition:get_neighbor_id(nid, {x, y})
          local rnum        = ComputeGhostRegionNum({x, y})
          ghost_lregs[rnum] =
            self:get_aliased_legion_subregion(nbr_id, {-x, -y})
        end end
      end
    end
    data[nid]             = ghost_lregs 
  end
  return data
end
