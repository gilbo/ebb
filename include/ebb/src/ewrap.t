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
--[[            Data Accessors - Fields, Futures Declarations              ]]--
-------------------------------------------------------------------------------


local N_NODES     = gas.nodes()
local THIS_NODE   = gas.mynode()


-------------------------------------------------------------------------------
-- Check metadata/field
-------------------------------------------------------------------------------


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

-- structs
Exports.Array                        = distdata.Array







