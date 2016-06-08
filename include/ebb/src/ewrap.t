-- The MIT License (MIT)
-- 
-- Copyright (c) 2015 Stanford University.
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
package.loaded["ebb.src.ewrap"] = Exports

local use_exp = rawget(_G,'EBB_USE_EXPERIMENTAL_SIGNAL')
if not use_exp then return Exports end

local C     = require "ebb.src.c"
local DLD   = require "ebb.lib.dld"
local Util  = require 'ebb.src.util'

-- signal to gasnet library not to try to find the shared lib or headers
rawset(_G,'GASNET_PRELOADED',true)
local gas       = require 'gasnet'
local gaswrap   = require 'gaswrap'
local distdata  = require 'distdata'

local newlist = terralib.newlist


-------------------------------------------------------------------------------
--            Helper Functions
-------------------------------------------------------------------------------

local function is_2_nums(obj)
  return type(obj) == 'table' and #obj == 2
     and type(obj[1]) == 'number' and type(obj[2]) == 'number'
end
local function is_3_nums(obj)
  return type(obj) == 'table' and #obj == 3
     and type(obj[1]) == 'number' and type(obj[2]) == 'number'
                                  and type(obj[3]) == 'number'
end


-------------------------------------------------------------------------------
--            Basic Setup  
-------------------------------------------------------------------------------

local N_NODES                   = gas.nodes()
local THIS_NODE                 = gas.mynode()

local BASIC_SAFE_GHOST_WIDTH    = 2


-------------------------------------------------------------------------------
--            Relations
-------------------------------------------------------------------------------

local EGridRelation   = {}
EGridRelation.__index = EGridRelation
local function is_egrid_relation(obj)
  return getmetatable(obj) == EGridRelation
end

local EField   = {}
EField.__index = EField
local function is_efield(obj) return getmetatable(obj) == EField end



--[[
  name = 'string'
  dims = {1d,2d,3d}
--]]
local relation_id_counter = 1
local function NewGridRelation(args)
  assert(type(args)=='table','expected table')
  assert(type(args.name) == 'string',"expected 'name' string arg")
  assert(is_2_nums(args.dims) or is_3_nums(args.dims),
         "expected 'dims' arg: list of 2 or 3 numbers")

  local rel_id        = relation_id_counter
  relation_id_counter = rel_id + 1

  distdata.broadcastNewRelation(rel_id, args.name, 'GRID', args.dims)

  return setmetatable({
    id    = rel_id,
    dims  = args.dims
  }, EGridRelation)
end

--[[
  name = 'string'
  rel  = EGridRelation
  type = TerraType
--]]
local field_id_counter = 1
local function NewField(args)
  assert(type(args)=='table','expected table')
  assert(type(args.name) == 'string',"expected 'name' string arg")
  assert(is_egrid_relation(args.rel),"expected 'rel' relation arg")
  assert(terralib.types.istype(args.type),"expected 'type' terra type arg")

  local f_id        = field_id_counter
  field_id_counter  = f_id + 1

  -- create the field
  distdata.broadcastNewField(f_id, args.rel.id, args.name,
                             terralib.sizeof(args.type))
  -- allocate memory to back the field
  local ghosts = {}
  for i,_ in ipairs(args.rel.dims) do ghosts[i] = BASIC_SAFE_GHOST_WIDTH end
  distdata.remoteAllocateField(args.rel.id, f_id, ghosts)

  local f = setmetatable({
    id      = f_id,
    rel_id  = args.rel.id,
  }, EField)

  return f
end




--[[
  blocking = {#,#,?} -- dimensions of grid of blocks
--]]
function EGridRelation:partition(args)
  assert(type(args)=='table','expected table')
  assert(is_2_nums(args.blocking) or is_3_nums(args.blocking),
         "expected 'blocking' arg: list of 2 or 3 numbers")
  local dims    = self.dims
  local blocks  = args.blocking
  assert(#dims == #blocks, 'dimension of blocking did not '..
                           'match dimension of grid')
  local bounds  = {}
  local map     = {}
  for i=1,#dims do assert(dims[i] > blocks[i], '# cells < # blocks') end

  -- generate bounds and map
  local nX,nY,nZ = unpack(blocks)
  if #dims == 2 then
    local dx      = math.floor(dims[1] / nX)
    local xlo,xhi = 0, dims[1] - dx*(nX-1)
    for i=1,nX do
      bounds[i], map[i] = {},{}

      local dy      = math.floor(dims[2] / nY)
      local ylo,yhi = 0, dims[2] - dy*(nY-1)
      for j=1,nY do
        local node_id = (i-1)*nY + (j-1) + 1
        map[i][j]     = node_id
        bounds[i][j]  = {
          lo = {xlo,ylo},
          hi = {xhi-1,yhi-1}, -- inclusive bound
        }
        ylo,yhi = yhi,yhi+dy
      end
      xlo,xhi = xhi,xhi+dx
    end
  else assert(#dims == 3)
    local dx      = math.floor(dims[1] / nX)
    local xlo,xhi = 0, dims[1] - dx*(nX-1)
    for i=1,nX do
      bounds[i], map[i] = {},{}

      local dy      = math.floor(dims[2] / nY)
      local ylo,yhi = 0, dims[2] - dy*(nY-1)
      for j=1,nY do
        bounds[i][j], map[i][j] = {},{}

        local dz      = math.floor(dims[3] / nZ)
        local zlo,zhi = 0, dims[3] - dz*(nZ-1)
        for k=1,nZ do
          local node_id   = (i-1)*nY*nZ + (j-1)*nZ + (k-1) + 1
          map[i][j][k]    = node_id
          bounds[i][j][k] = {
            lo = {xlo,ylo,zlo},
            hi = {xhi-1,yhi-1,zhi-1}, -- inclusive bound
          }
          zlo,zhi = zhi,zhi+dz
        end
        ylo,yhi = yhi,yhi+dy
      end
      xlo,xhi = xhi,xhi+dx
    end
  end

  self._partition_map     = map
  self._partition_bounds  = bounds
  distdata.broadcastGlobalGridPartition(self.id, blocks, bounds, map)
end


-------------------------------------------------------------------------------
--            Yadda
-------------------------------------------------------------------------------

--[[
local struct FieldData {
  array      : distdata.Array;
  field_name : uint8[64];
  rel_name   : uint8[64];
}

--[[
local terra dumpFloatFieldData(args : &opaque)
  var worker_id : uint = [gas.mynode()]
  var field_data = [&FieldData](args)
  var ptr : &float = [&float](field_data.array:DataPtr())
  C.printf('[%u] Dumping field data for relation %s, field %s\n',
           worker_id, field_data.rel_name, field_data.field_name)
  var lo     = field_data.array:LowerBounds()
  var hi     = field_data.array:HigherBounds()
  var stride = field_data.array:Stride()
  for j = lo[1], hi[1], 1 do
    for i = lo[0], hi[0], 1 do
      C.printf('[%u] %u, %u : %f\n', worker_id, i, j,
                                     ptr[i*stride[0] + j*stride[1] ])
    end
  end
end

-- print list of relations, fields, and data over the fields
local function dumpAllFields()
  os.execute('sleep 2')
  for _, rel in pairs(distdata._TESTING_relation_metadata) do
    local bounds = rel:GetPartitionBounds()
    local bounds_str = '{' .. gaswrap.stringify_list(bounds.lo) ..
                       ',' ..
                       gaswrap.stringify_list(bounds.hi) .. '}'
    local map_str   = gaswrap.stringify_list(rel:GetPartitionMap())
    print('[' .. tostring(gas.mynode()) ..
          '] Relation ' .. tostring(rel:Name()) .. ', bounds ' .. bounds_str ..
          ', map ' .. map_str .. ':')
    for _, field in pairs(rel:Fields()) do
      print('  [' .. tostring(gas.mynode()) ..
            '] Field ' .. tostring(field:Name()))
      local field_data = terralib.cast(&FieldData,
                            C.malloc(terralib.sizeof(FieldData)))
      field_data.array = field:GetArray()
      field_data.field_name = field:Name()
      field_data.rel_name   = rel:Name()
      gaswrap.acquireScheduler()
      local ws = field:GetPreviousWriteSignal()
      local a_out = ws:exec(0, dumpFloatFieldData:getpointer(),
                               field_data)
      field:RecordRead(a_out)
      gaswrap.releaseScheduler()
    end
  end
end

gaswrap.registerLuaEvent('dumpAllFields', dumpAllFields)
--]]







-------------------------------------------------------------------------------
-- Extra Events
-------------------------------------------------------------------------------



-------------------------------------------------------------------------------
-- Exports
-------------------------------------------------------------------------------

Exports.N_NODES                       = N_NODES
Exports.THIS_NODE                     = THIS_NODE


-- functions
Exports.broadcastNewRelation          = distdata.broadcastNewRelation
Exports.broadcastGlobalGridPartition  = distdata.broadcastGlobalGridPartition
Exports.broadcastNewField             = distdata.broadcastNewField
Exports.remoteAllocateField           = distdata.remoteAllocateField
Exports.remoteLoadFieldConstant       = distdata.remoteLoadFieldConstant

Exports.NewGridRelation               = NewGridRelation
Exports.NewField                      = NewField

-- structs
Exports.Array                         = distdata.Array







