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
  'rel_global_partition', 'node_type', --'params',
}, function(args)
  --error('no good right now')
  assert(args.rel_global_partition)
  assert(args.node_type)
  return setmetatable({
    _global_part  = args.rel_global_partition,
    _node_type    = args.node_type,
  },RelationLocalPartition)
end)

function RelationGlobalPartition:execute_partition()
  if self._lpart then return end -- make idempotent
  local lreg  = self._lreg
  local lpart = lreg:CreateDisjointBlockPartition(self._dims)
  self._lpart = lpart
end


function RelationLocalPartition:execute_partition()
  if self._lpart then return end -- make idempotent

  self._global_part:execute_partition()
  local gpart = self._global_part._lpart
  for _,p in ipairs(gpart:subregions()) do
    local i,j,k = unpack(p.idx)
    if true then -- if node-type matches... TODO

    end
  end
end


-------------------------------------------------------------------------------
--[[ Global / Local Partition Ghosts:                                      ]]--
-------------------------------------------------------------------------------

--local GlobalGhostPattern    = {}
--GlobalGhostPattern.__index  = GlobalGhostPattern

local LocalGhostPattern     = {}
LocalGhostPattern.__index   = LocalGhostPattern

Exports.LocalGhostPattern   = Util.memoize_named({
  'rel_local_partition', --'params',
}, function(args)
  --error('no good right now')
  return setmetatable({},LocalGhostPattern)
end)

function LocalGhostPattern:supports(stencil)
  return true -- TODO: implement actual check
end




















