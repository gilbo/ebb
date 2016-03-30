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

Exports.RelGlobalPartition = Util.memoize(function(
  relation, nX, nY, nZ
)
  assert(R.is_relation(relation))
  assert(nX and nY)
  return setmetatable({
    _n_nodes    = nX * (nY or 1) * (nZ or 1),
    _blocking   = { nX, nY, nZ },
    _n_x        = nX,
    _n_y        = nY,
    _n_z        = nZ,
    _lreg       = relation._logical_region_wrapper,
    _rel_dims   = relation:Dims()
    --_relation   = relation,
  },RelationGlobalPartition)
end)

function RelationGlobalPartition:get_n_blocks()
  return self._n_nodes
end

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

function RelationLocalPartition:get_global_partition()
  return self._global_part
end

local function linearize_idx_3d(i,j,k,nx,ny,nz)
  return i*ny*nz + j*nz + k
end
local function linearize_idx_2d(i,j,nx,ny)
  return i*ny + j
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
  if self._lpart then return end -- make idempotent
  local lreg  = self._lreg
  local lpart = lreg:CreateDisjointPartition(self:get_subrects())
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

-- TODO: HACK: for now, just return global partition.
-- Global partition = local partition till we are on a single node.
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

function LocalGhostPattern:get_local_partition()
  return self._rel_local_partition
end

local ghost_nums_2d = {
  [-1]  = {
    [-1]  = 0,
    [0]   = 1,
    [1]   = 2,
  },
  [0] = {
    [-1]  = 3,
    --[0]   = 1,
    [1]   = 4,
  },
  [1] = {
    [-1]  = 5,
    [0]   = 6,
    [1]   = 7,
  },
}
local n_2d_ghost_rects = 8

local ghost_nums_3d = {
  [-1]  = {
    [-1]  = { [-1] =  0,  [0] =  1,  [1] =  2 },
    [0]   = { [-1] =  3,  [0] =  4,  [1] =  5 },
    [1]   = { [-1] =  6,  [0] =  7,  [1] =  8 },
  },
  [0] = {
    [-1]  = { [-1] =  9,  [0] = 10,  [1] = 11 },
    [0]   = { [-1] = 12, --[[0] =,]] [1] = 13 },
    [1]   = { [-1] = 14,  [0] = 15,  [1] = 16 },
  },
  [1] = {
    [-1]  = { [-1] = 17,  [0] = 18,  [1] = 19 },
    [0]   = { [-1] = 20,  [0] = 21,  [1] = 22 },
    [1]   = { [-1] = 23,  [0] = 24,  [1] = 25 },
  },
}
local n_3d_ghost_rects = 26

function GlobalGhostPattern:get_all_subrects()
  local is3d  = #self._rel_global_partition._rel_dims == 3
  local depth = self._depth

  if depth == 0 then return newlist() end

  local subrects = newlist()
  local all = {-math.huge,math.huge}
  -- TODO: QUESTION: was the intention for the following rectangles to be overlapping?
  -- TODO: QUESTION: what happens when regions are so tiny that these don't make sense?
  if is3d then
    for i,reg in ipairs(self._rel_global_partition:subregions()) do
      local rect = reg.rect
      local xlo,ylo,zlo,xhi,yhi,zhi = rect:mins_maxes()
      local xn, xm, xp  = Util.NewRect3d({-math.huge,xlo+depth},all,all),
                          Util.NewRect3d({ xlo+depth,xhi-depth},all,all),
                          Util.NewRect3d({ xhi-depth,math.huge},all,all)
      local yn, ym, yp  = Util.NewRect3d(all,{-math.huge,ylo+depth},all),
                          Util.NewRect3d(all,{ ylo+depth,yhi-depth},all),
                          Util.NewRect3d(all,{ yhi-depth,math.huge},all)
      local zn, zm, zp  = Util.NewRect3d(all,all,{-math.huge,zlo+depth}),
                          Util.NewRect3d(all,all,{ zlo+depth,zhi-depth}),
                          Util.NewRect3d(all,all,{ zhi-depth,math.huge})
      subrects:insert( newlist {
        rect:clip(xn):clip(yn):clip(zn), -- -1,-1,-1
        rect:clip(xn):clip(yn):clip(zm), -- -1,-1, 0
        rect:clip(xn):clip(yn):clip(zp), -- -1,-1, 1
        rect:clip(xn):clip(xm):clip(zn), -- -1, 0,-1
        rect:clip(xn):clip(xm):clip(zm), -- -1, 0, 0
        rect:clip(xn):clip(xm):clip(zp), -- -1, 0, 1
        rect:clip(xn):clip(yp):clip(zn), -- -1, 1,-1
        rect:clip(xn):clip(yp):clip(zm), -- -1, 1, 0
        rect:clip(xn):clip(yp):clip(zp), -- -1, 1, 1

        rect:clip(xm):clip(yn):clip(zn), --  0,-1,-1
        rect:clip(xm):clip(yn):clip(zm), --  0,-1, 0
        rect:clip(xm):clip(yn):clip(zp), --  0,-1, 1
        rect:clip(xm):clip(ym):clip(zn), --  0, 0,-1
                                         --  0, 0, 0
        rect:clip(xm):clip(ym):clip(zp), --  0, 0, 1
        rect:clip(xm):clip(yp):clip(zn), --  0, 1,-1
        rect:clip(xm):clip(yp):clip(zm), --  0, 1, 0
        rect:clip(xm):clip(yp):clip(zp), --  0, 1, 1

        rect:clip(xp):clip(yn):clip(zn), --  1,-1,-1
        rect:clip(xp):clip(yn):clip(zm), --  1,-1, 0
        rect:clip(xp):clip(yn):clip(zp), --  1,-1, 1
        rect:clip(xp):clip(ym):clip(zn), --  1, 0,-1
        rect:clip(xp):clip(ym):clip(zm), --  1, 0, 0
        rect:clip(xp):clip(ym):clip(zp), --  1, 0, 1
        rect:clip(xp):clip(yp):clip(zn), --  1, 1,-1
        rect:clip(xp):clip(yp):clip(zm), --  1, 1, 0
        rect:clip(xp):clip(yp):clip(zp), --  1, 1, 1
      })
    end
  else
    for i,reg in ipairs(self._rel_global_partition:subregions()) do
      local rect = reg.rect
      local xlo,ylo,xhi,yhi = rect:mins_maxes()
      local xn, xm, xp  = Util.NewRect2d({-math.huge,xlo+depth},all),
                          Util.NewRect2d({ xlo+depth,xhi-depth},all),
                          Util.NewRect2d({ xhi-depth,math.huge},all)
      local yn, ym, yp  = Util.NewRect2d(all,{-math.huge,ylo+depth}),
                          Util.NewRect2d(all,{ ylo+depth,yhi-depth}),
                          Util.NewRect2d(all,{ yhi-depth,math.huge})
      subrects:insert( newlist {
        rect:clip(xn):clip(yn), -- -1,-1
        rect:clip(xn):clip(ym), -- -1, 0
        rect:clip(xn):clip(yp), -- -1, 1
        rect:clip(xm):clip(yn), --  0,-1
                                --  0, 0
        rect:clip(xm):clip(yp), --  0, 1
        rect:clip(xp):clip(yn), --  1,-1
        rect:clip(xp):clip(ym), --  1, 0
        rect:clip(xp):clip(yp), --  1, 1
      })
    end
  end
  return subrects
end

-- set up ghost regions for each node
function GlobalGhostPattern:execute_partition()
  if self._ghost_partition then return end -- make idempotent

  local all_subrects = self:get_all_subrects()
  if #all_subrects == 0 then
    return
  end

  self._ghost_partitions = newlist()
  for i,reg in ipairs(self._rel_global_partition:subregions()) do
    local subrects  = all_subrects[i]
    -- TODO: QUESTION: why overlapping?
    local lpart     = reg:CreateOverlappingPartition(subrects)
    self._ghost_partitions:insert(lpart)
  end
end

-- set up ghost regions internal to a node
function LocalGhostPattern:execute_partition()
  -- just defer for now...
  self._global_pattern:execute_partition()

  --if self._lpart then return end -- make idempotent
  --local lreg  = self._lreg
  --local lpart = lreg:CreateDisjointBlockPartition(self._blocking)
  --self._lpart = lpart
end

function LocalGhostPattern:get_legion_partition()
  if self._depth == 0 then
    return self._rel_local_partition:get_legion_partition()
  else
    -- TODO: HACK: for now, just return nil for non-zero stencils
    -- This will make legionwrap add entire logical region isntead of
    -- a partition.
    -- We should use computed ghost regions instead
    return nil
  end
end
