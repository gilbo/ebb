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

local T                 = require 'ebb.src.types'
local Stats             = require 'ebb.src.stats'
local Util              = require 'ebb.src.util'

local Pre               = require 'ebb.src.prelude'
local R                 = require 'ebb.src.relations'
local specialization    = require 'ebb.src.specialization'
local semant            = require 'ebb.src.semant'
local phase             = require 'ebb.src.phase'
local stencil           = require 'ebb.src.stencil'

------------------------------------------------------------------------------

-------------------------------------------------------------------------------
--[[ Node Types:                                                           ]]--
-------------------------------------------------------------------------------

-- would be nice to define these here...


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
  return setmetatable({
    n_nodes   = nX * nY * nZ,
    dims      = { nX, nY, nZ },
    n_x       = nX,
    n_y       = nY,
    n_z       = nZ,
    relation  = relation,
  })
end)

Exports.RelLocalPartition = Util.memoize(function(
  rel_global_partition, node_type, params
)
  error('no good right now')
  return setmetatable({
    n_nodes   = nX * nY * nZ,
    dims      = { nX, nY, nZ },
    n_x       = nX,
    n_y       = nY,
    n_z       = nZ,
    relation  = relation,
  })
end)


-------------------------------------------------------------------------------
--[[ Global / Local Partition Ghosts:                                      ]]--
-------------------------------------------------------------------------------

--local GlobalPartitionGhost    = {}
--GlobalPartitionGhost.__index  = GlobalPartitionGhost

local LocalPartitionGhost     = {}
LocalPartitionGhost.__index   = LocalPartitionGhost

Exports.LocalPartitionGhost = Util.memoize(function(
  rel_local_partition, params
)
  error('no good right now')
end)























