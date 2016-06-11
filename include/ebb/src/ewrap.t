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

local C       = require "ebb.src.c"
local DLD     = require "ebb.lib.dld"
local Util    = require 'ebb.src.util'

-- Following modules are probably not going to depdend (directly or indirectly)
-- on ewrap since they define basic types.
local Pre     = require 'ebb.src.prelude'
local Rawdata = require 'ebb.src.rawdata'
local DynamicArray = Rawdata.DynamicArray
--local Types   = require 'ebb.src.types'

-- signal to gasnet library not to try to find the shared lib or headers
rawset(_G,'GASNET_PRELOADED',true)
local gas       = require 'gasnet'
local gaswrap   = require 'gaswrap'

local newlist = terralib.newlist


-------------------------------------------------------------------------------
-- Basic Setup  
-------------------------------------------------------------------------------

local N_NODES                   = gas.nodes()
local THIS_NODE                 = gas.mynode()
local CONTROL_NODE              = 0

local BASIC_SAFE_GHOST_WIDTH    = 2

-- constants/modes
local GRID                      = 'GRID'
local CPU                       = tostring(Pre.CPU)
local GPU                       = tostring(Pre.GPU)


-------------------------------------------------------------------------------
-- Helper methods
-------------------------------------------------------------------------------

-- assumption: 0 is control node, remaining are compute nodes
local function numComputeNodes()  -- get total number of compute nodes
  return gas.nodes() - 1
end
local function BroadcastLuaEventToComputeNodes(event_name, ...)
  print('*** DEBUG INFO: Sending ' .. event_name)
  for i = 1,numComputeNodes() do
    gaswrap.sendLuaEvent(i, event_name, ...)
  end
end

local default_align_max_pow = 32 -- this should accomodate diff architectures
local function pow2align(N,max_pow)
  max_pow = max_pow or default_align_max_pow
  if N > max_pow then -- smallest multiple of max_pow that works
    return math.ceil(N/max_pow)*max_pow
  else
    -- find smallest power of 2 >= N
    while max_pow / 2 >= N do max_pow = max_pow / 2 end
    return max_pow
  end
end

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
-- Relation and Field objects at compute nodes
-------------------------------------------------------------------------------

--[[
array,
dld,
ghost_width
--]]
local FieldInstanceTable = {}
FieldInstanceTable.__index = FieldInstanceTable

--[[
params {
  bounds,
  ghost_width,
  type_size
}
--]]
local function NewFieldInstanceTable(params)
  local range        = params.bounds:getranges()
  local ghost_width  = params.ghost_width
  local n_elems      = 1
  local dim_size     = {}
  local dim_stride   = {}
  for d = 1,3 do
    dim_stride[d]  = n_elems   -- column major?
    if not ghost_width[d] then ghost_width[d] = 0 end
    dim_size[d]    = range[d][2] - range[d][1] + 1
    n_elems        = n_elems * (dim_size[d] + 2*ghost_width[d])
  end
  local elem_size   = params.type_size
  local elem_stride = pow2align(elem_size)
  local array = DynamicArray.New {
    size      = elem_size * n_elems,
    type      = uint8[elem_size],
    processor = Pre.CPU,
  }
  local dld = DLD.NewDLD {
    base_type   = DLD.UINT_8,
    location    = DLD.CPU,
    type_stride = elem_stride,
    address     = array:_raw_ptr(),
    dim_size    = dim_size,
    dim_stride  = dim_stride,
  }
  return setmetatable ({
    array       = array,
    dld         = dld,
    ghost_width = ghost_width,
  }, FieldInstanceTable)
end

function FieldInstanceTable:DataPtr()
  return self.array:_raw_ptr()
end

function FieldInstanceTable:DataPtrGhostAdjusted()
  local ptr = terralib.cast(&uint8, self:DataPtr())
  local offset = 0
  for d = 1,3 do
    offset = offset + self.ghost_width[d] *
                      self.dld.dim_stride[d] * self.dld.type_stride
  end
  return ptr - offset
end

function FieldInstanceTable:DLD()
  return self.dld
end

local relation_metadata = {}
local field_metadata    = {}

--[[
FieldMetada : {
  type, { base_type, n_rows, n_cols } (pull this out from ebb types?)
  name,
  relation,
  array, { ghost_width, n_elems, ptr } (pull this out from rawdata?)
  rw_signal, (signal representing previous read/write, triggers when done)
}
--]]

local FieldData = {}
FieldData.__index = FieldData

local function NewFieldData(relation, name, typ_size)
  return setmetatable({
                        type_size    = typ_size,
                        name         = name,
                        relation     = relation,
                        instance     = nil,
                        last_read    = nil,
                        last_write   = nil,
                      },
                      FieldData)
end

function FieldData:Name()
  return self.name
end

function FieldData:isAllocated()
  return (self.array ~= nil)
end

function FieldData:AllocateInstance(gw)
  assert(self.array == nil, 'Trying to allocate already allocated array.')
  assert(self.relation:isPartitioned(),
         'Relation ' .. self.relation:Name() .. 'not partitioned. ' ..
         'Cannot allocate data over it.')
  self.instance   = NewFieldInstanceTable {
                      bounds      = self.relation:GetPartitionBounds(),
                      ghost_width = gw,
                      type_size   = self:GetTypeSize(),
                    }
  self.last_read  = gaswrap.newSignalSource():trigger()
  self.last_write = gaswrap.newSignalSource():trigger()
end

function FieldData:GetTypeSize()
  return self.type_size
end

function FieldData:GetInstance()
  return self.instance
end

function FieldData:GetDLD()
  return self.instance:DLD()
end

function FieldData:GetDataPtr()
  return self.instance:DataPtr()
end

function FieldData:GetPreviousReadSignal()
  return self.last_read
end
function FieldData:GetPreviousWriteSignal()
  return self.last_write
end
function FieldData:ForkPreviousWriteSignal()
  local signals = terralib.new(gaswrap.Signal[2])
  self.last_write:fork(2, signals)
  self.last_write = signals[1]
  return signals[0]
end
function FieldData:RecordRead(signal)
  self.last_read = signal
end
function FieldData:RecordReadWrite(signal)
  local signals = terralib.new(gaswrap.Signal[2])
  signal:fork(2, signals)
  self.last_read  = signals[0]
  self.last_write = signals[1]
end

--[[
RelationData : {
  name,
  mode,
  dims,  (global dimensions, might not need this)
  partition, { blocking, block_id, bounds, map },
  fields,
}
--]]

local RelationData = {}
RelationData.__index = RelationData

local function NewRelationData(name, mode, dims)
  assert(mode == 'GRID', 'Relations must be of grid type on GASNet.')
  return setmetatable({
                        name      = name,
                        mode      = mode,
                        dims      = dims,
                        partition = nil,
                        fields    = newlist(),
                      }, RelationData)
end

function RelationData:Name()
  return self.name
end

function RelationData:Dims()
  return self.dims
end

function RelationData:Fields()
  return self.fields
end

function RelationData:RecordPartition(blocking, block_id, bounds, map)
  if self.partition ~= nil then
    print(
          'Only one partition per relation supported. Node ' ..
          tostring(gas.mynode()) .. ' already has partition ' ..
          gaswrap.lson_stringify(self.partition.bounds) .. ',' ..
          'Cannot add ' .. gaswrap.lson_stringify(bounds)
         )
    assert(false)
  end
  assert(Util.isrect2d(bounds) or Util.isrect3d(bounds))
  self.partition = {
    blocking = blocking,
    block_id = block_id,
    bounds   = bounds,
    map      = map,
  }
end

function RelationData:isPartitioned()
  return self.partition
end

function RelationData:RecordField(f_id, name, typ_size)
  assert(self.fields[f_id] == nil, "Recording already recorded field '"..
                                   name.."' with id # "..f_id)
  self.fields:insert(f_id)
  field_metadata[f_id] = NewFieldData(self, name, typ_size)
end

function RelationData:GetPartitionBounds()
  return self.partition.bounds
end

function RelationData:GetPartitionMap()
  return self.partition.map
end

local function RecordNewRelation(unq_id, name, mode, dims)
  assert(relation_metadata[unq_id] == nil,
         "Recordling already recorded relation '"..name..
         "' with id # "..unq_id)
  relation_metadata[unq_id] = NewRelationData(name, mode, dims)
end

local function GetRelationData(unq_id)
  return relation_metadata[unq_id]
end

function GetFieldData(f_id)
  return field_metadata[f_id]
end


-------------------------------------------------------------------------------
-- Relations, fields, partitions
-------------------------------------------------------------------------------

------------------------------------
-- EVENT BROADCASTS/ EVENT HANDLERS
------------------------------------

-- send relation metadata
local function BroadcastNewRelation(unq_id, name, mode, dims)
  assert(mode == 'GRID', 'Unsupported relation mode ' .. mode)
  assert(type(dims) == 'table')
  local rel_size = gaswrap.lson_stringify(dims)
  BroadcastLuaEventToComputeNodes( 'newRelation', unq_id, name,
                                                  mode, rel_size)
end
-- event handler for relation metadata
local function CreateNewRelation(unq_id, name, mode, dims)
  RecordNewRelation(tonumber(unq_id), name, mode, gaswrap.lson_eval(dims))
end

-- disjoint partition blocking over relation
-- options: send subregions here or let nodes construct it from the blocking
-- assumption: we'll have only one disjoint partitioning per relation
--             works for static relations, revisit later for dynamic relations
local function SendGlobalGridPartition(
  nid, rel_id, blocking, bid, partition, map_str
)
  local blocking_str  = gaswrap.lson_stringify(blocking)
  local block_id_str  = gaswrap.lson_stringify(bid) 
  assert(Util.isrect2d(partition) or Util.isrect3d(partition))
  local partition_str = gaswrap.lson_stringify(partition:getranges())
  gaswrap.sendLuaEvent(nid, 'globalGridPartition', rel_id, blocking_str,
                       block_id_str, partition_str, map_str)
end
local function BroadcastGlobalGridPartition(
  rel_id, blocking, partitioning, map
)
  -- map: How node/partitions are arranged into a block.
  --      So could be {{1,2}, {3,4}}.
  -- Lets controller decide which partitions go where.
  local num_partitions = 1
  for d = 1,#blocking do
    num_partitions = num_partitions * blocking[d]
  end
  assert(num_partitions == numComputeNodes(),
         'Number of partitions ' .. num_partitions ..
         ' ~= number of nodes ' .. numComputeNodes())
  local map_str = gaswrap.lson_stringify(map)
  if #blocking == 2 then
    for xid, m in ipairs(map) do
      for yid, nid in ipairs(m) do
        assert(nid >= 1 and nid <= numComputeNodes())
        SendGlobalGridPartition(nid, rel_id, blocking, {xid, yid},
                                partitioning[xid][yid], map_str)
      end
    end
  else
    assert(#blocking == 3, 'Expected 2 or 3 dimensional grid.')
    for xid, mt in ipairs(map) do
      for yid, m in ipairs(mt) do
        for zid, nid in ipairs(mt) do
          assert(nid >= 1 and nid <= numComputeNodes())
          sendGlobalGridPartition(nid, rel_id, blocking, {xid, yid, zid},
                                  partitioning[xid][yid][zid], map_str)
        end
      end
    end
  end
end
-- event handler for partition metadata
-- assumption: only one partition gets mapped to this node
-- question: how do we do local partititons in a node?
local function CreateGlobalGridPartition(rel_id, blocking_str,
                                         blocking_id_str,
                                         partition_str, map_str)
  local relation = GetRelationData(tonumber(rel_id))
  assert(relation,
         'Relation #' .. rel_id .. ' to partition is not defined.')
  assert(relation.block_id == nil,
         'Already recorded a partition. ' ..
         'Are multiple partitions mapped to this node?')
  local blocking  = gaswrap.lson_eval(blocking_str)
  local block_id  = gaswrap.lson_eval(blocking_id_str)
  local range     = gaswrap.lson_eval(partition_str)
  if not range[3] then range[3] = {1,1} end
  local bounds = Util.NewRect3d(unpack(range))
  local map       = gaswrap.lson_eval(map_str)
  relation:RecordPartition(blocking, block_id, bounds, map)
end

-- record a new field over a relation
local function BroadcastNewField(f_id, rel_id, field_name, type_size)
  BroadcastLuaEventToComputeNodes('recordNewField', f_id, rel_id, field_name,
                                                    type_size)
end
local function RecordNewField(f_id, rel_id, field_name, type_size)
  local relation = GetRelationData(tonumber(rel_id))
  relation:RecordField(tonumber(f_id), field_name, tonumber(type_size))
end

-- allocate field over remote nodes
-- shared memory, one array across threads for now
local function RemoteAllocateField(f_id, ghost_width)
  BroadcastLuaEventToComputeNodes('allocateField', f_id,
                                  gaswrap.lson_stringify(ghost_width))
end
-- event handler to allocate array for a field
local function AllocateField(f_id, ghost_width_str)
  local field = GetFieldData(tonumber(f_id))
  assert(field, 'Attempt to allocate unrecorded field #'..f_id)
  local ghost_width = gaswrap.lson_eval(ghost_width_str)
  field:AllocateInstance(ghost_width)
end

--[[
-- just constant for now
-- can probably convert other kinds of loads to tasks
-- value should be a 2 level list right now, can relax this once we start using
-- luatoebb conversions from ebb.
local function RemoteLoadFieldConstant(f_id, rel_id, value)
  assert(false, 'INTERNAL ERROR: Load is broken currently.')
  local value_ser = gaswrap.lson_stringify(value)
  BroadcastLuaEventToComputeNodes('loadFieldConstant', f_id, rel_id, value_ser)
end
local struct ArrayLoadConst{
  array : Array,
  val   : &opaque,
}
local terra load_field_constant(args : &opaque)
  var array_data  = [&ArrayLoadConst](args)
  var ptr         = [&uint8](array_data.array:DataPtr())
  var type_size   = array_data.array:TypeSize()
  var type_stride = array_data.array:TypeStride()
  for i = 0, array_data.array:NumElems() do
  -- write over ghost values too as it doesn't matter
    C.memcpy(ptr, array_data.val, type_size)
    ptr = ptr + type_stride
  end
  C.free(array_data.val)
  C.free(array_data)
end
local function LoadFieldConstant(f_id, rel_id, value_ser)
  assert(false, 'INTERNAL ERROR: Load is broken currently.')
  local rel      = GetRelationData(tonumber(rel_id))
  assert(rel, 'Attempt to load into a field over unrecorded relation ' ..
               rel_id)
  local field    = rel:GetFieldData(tonumber(f_id))
  assert(field, 'Attempt to load into unrecorded field ' .. f_id ..
                ' over ' .. rel_id .. '.')
  assert(field:isAllocated(), 'Attempt to load into unallocated field ' ..
                              field:Name() ..  ' over ' .. rel:Name() .. '.')
  local array_data  = terralib.cast(&ArrayLoadConst,
                                    C.malloc(terralib.sizeof(ArrayLoadConst)))
  array_data.array  = field:GetArray()
  local typ         = field:GetType()
  local val_data    = terralib.new(typ.base_type[typ.n_cols][typ.n_cols],
                                   gaswrap.lson_eval(value_ser))
  local elem_size   = terralib.sizeof(typ.base_type) *
                      typ.n_rows * typ.n_cols
  array_data.val    = C.malloc(elem_size)
  C.memcpy(array_data.val, val_data, elem_size)
  -- single threaded right now
  gaswrap.acquireScheduler()
  field:GetPreviousWriteSignal():sink()
  local a_in  = field:ForkPreviousReadSignal(1)[0]
  local a_out = a_in:exec(0, load_field_constant:getpointer(),
                          array_data)
  field:RecordReadWrite(a_out)
  gaswrap.releaseScheduler()
end
--]]

-----------------------------------
-- HELPER METHODS FOR CONTROL NODE
-----------------------------------

--[[
  id,
  dims,
  _partition_map,
  _partition_bounds,
--]]
local EGridRelation   = {}
EGridRelation.__index = EGridRelation
local function is_egrid_relation(obj)
  return getmetatable(obj) == EGridRelation
end

--[[
  rel_id,
  id
--]]
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

  BroadcastNewRelation(rel_id, args.name, 'GRID', args.dims)

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
  BroadcastNewField(f_id, args.rel.id, args.name, terralib.sizeof(args.type))
  -- allocate memory to back the field
  local ghosts = {}
  for i,_ in ipairs(args.rel.dims) do ghosts[i] = BASIC_SAFE_GHOST_WIDTH end
  RemoteAllocateField(f_id, ghosts)

  local f = setmetatable({
    id      = f_id,
    rel_id  = args.rel.id,
  }, EField)

  return f
end

--[[
  blocking = {#,#,?} -- dimensions of grid of blocks
--]]
function EGridRelation:partition_across_nodes(args)
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
        bounds[i][j]  = Util.NewRect2d({xlo,xhi-1},{ylo,yhi-1})
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
          bounds[i][j][k] = Util.NewRect3d({xlo,xhi-1},{ylo,yhi-1},{zlo,zhi-1})
          zlo,zhi = zhi,zhi+dz
        end
        ylo,yhi = yhi,yhi+dy
      end
      xlo,xhi = xhi,xhi+dx
    end
  end

  self._partition_map     = map
  self._partition_bounds  = bounds
  BroadcastGlobalGridPartition(self.id, blocks, bounds, map)
end

-- TODO:
-- local partitions (for partitioning task across threads)
function EGridRelation:partition_within_nodes()
end


-------------------------------------------------------------------------------
-- Handle actions over sets of fields and privileges
-------------------------------------------------------------------------------

-- **** THIS SECTION IS VERY UNSTABLE AND INCOMPLETE ****

--[[
Each task is associated with:
  * terra code for the task
  * relation the task is compiled for
  * processor (CPU or GPU)
  * field_accesses
  * .. globals not yet
Each ufversion maps to a unique task.
The controller node is responsible for managing mapping between functions
and ufversions, making launch decisions, and invoking the correct task with
a selected partitioning.
If the controller decides to invoke a function across cpus + gpus (in rare
cases?), it is responsible for invoking two tasks, the cpu version and the gpu
version, with the correct partitioning.
Compute nodes should not be making these decisions.

A task can be invoked with different partitionings. This partitioning only
determines how a task is run in parallel all the local processors, and over
what parts of the relation.
A task partitioning over a relation must be registered separately before a task
with that partitioning is invoked.

For now, we assume that the controller node has allocated data correctly on
worker nodes, to support all the tasks (with uncentered accesses), and
that there is only one instance of data for every {relation, field}.
--]]

local privileges = {
  read_only  = 1,
  read_write = 2,
  reduce     = 4
}
--[[
  privilege,
  ghost_width/uncentered to indicate communication?
--]]
local Access = {}
Access.__index = Access
local function NewAccess(privilege)
  return setmetatable({
    privilege = privilege,
  }, Access)
end
function Access:__tostring()
  return '{' ..
            tostring(self.privilege) ..
         '}'
end
local function AccessFromString(str)
  return NewAccess(unpack(gaswrap.lson_eval(str)))
end

local used_task_id    = 0   -- last used task id, for generating ids
local task_table      = {}  -- map task id to terra code and other task metadata

--[[
{
  id,
  name, (optional)
  func, (args : bounds, ordered list of fields, 0 indexed, return : void)
  rel_id,
  processor,
  fields, (ordered list, 1 indexed)
  field_accesses
}
  ** TASK IS RESPONSIBLE FOR CLEANING UP ARGS **
--]]
local Task   = {}
Task.__index = Task

local struct BoundsStruct { lo : uint64, hi : uint64 }
local struct FieldInstance {
  ptr : &uint8;
  dld : DLD.C_DLD;
}
local struct TaskArgs {
  bounds : BoundsStruct[3];
  fields : &FieldInstance;
}

local function NewTaskMessage(task_id, params)
  assert(params.task_func and params.rel_id and
         params.processor and params.fields and params.field_accesses,
         'One or more arguments necessary to define a new task missing.')
  assert(terralib.isfunction(params.task_func),
         'Invalid task function to NewTask.')
  assert(params.processor == Pre.CPU, 'Only CPU tasks supported right now.')
  local fields = newlist(params.fields)
  local field_accesses = {}
  for field,access in pairs(params.field_accesses) do
    local privilege = access:isReadOnly() and privileges.read_only or
                      access:requiresExclusive() and privileges.read_write or
                      privileges.reduction
    field_accesses[field._ewrap_field.id] = NewAccess(privilege)
  end
  local task_name = params.name or params.task_func:getname()
  local bitcode   = terralib.saveobj(nil, 'bitcode',
                                    {[task_name]=params.task_func})
  local t = {
              task_id   = task_id,
              name      = task_name,
              rel_id    = params.rel_id,
              processor = tostring(params.processor),
              fields    = fields,
              field_accesses = field_accesses,
            }
  BroadcastLuaEventToComputeNodes('newTask', bitcode, gaswrap.lson_stringify(t))
end

local function TaskFromString(bitcode, msg)
  local t         = gaswrap.lson_eval(msg)
  assert(t.processor == CPU)
  local blob      = terralib.linkllvmstring(bitcode)
  local task_func = blob:extern(t.name, {&opaque} -> {})
  local task = setmetatable({
    id             = t.task_id,
    name           = t.name,
    func           = task_func:compile(),
    rel_id         = t.rel_id,
    processor      = Pre.CPU,
    fields         = t.fields,
    field_accesses = t.field_accesses,
  }, Task)
  return task
end

-- Event/handler for registering a new task. Returns a task id.
--[[
params : {
  task_func,
  task_name,
  rel_id,
  processor,
  fields,
  field_accesses
}
returns task_id
--]]
local function RegisterNewTask(params)
  assert(gas.mynode() == CONTROL_NODE,
         'Can send tasks from control node only.')
  assert(params.processor == Pre.CPU, 'Tasks over ' .. GPU .. ' not supported yet.')
  used_task_id  = used_task_id + 1
  local task_id = used_task_id
  local msg     = NewTaskMessage(task_id, params)
  return task_id
end
local function ReceiveNewTask(bitcode, msg)
  local task = TaskFromString(bitcode, msg)
  task_table[task.id] = task
end

-- Invoke a single task.
local function SendTaskLaunch(task_id, partition_id)
  BroadcastLuaEventToComputeNodes('launchTask', task_id)
end
local function LaunchTask(task_id_ser, partition_id_ser)
  local task = task_table[tonumber(task_id_ser)]
  assert(task, 'Task ' .. task_id_ser ..  ' is not registered.')
  -- allocate signal arrays
  local num_fields  = #task.fields
  local signals_in  = terralib.new(gaswrap.Signal[num_fields])
  local a_out_forked = terralib.new(gaswrap.Signal[num_fields])
  -- can we/ should we reuse args across launches?
  -- ** TASK IS RESPONSIBLE FOR CLEANING UP ARGS **
  local args  = terralib.cast(&TaskArgs, C.malloc(terralib.sizeof(TaskArgs)))
  args.fields = terralib.cast(&FieldInstance,
                              C.malloc(num_fields * terralib.sizeof(FieldInstance)))
  local bounds = GetRelationData(task.rel_id):GetPartitionBounds()
  local range  = bounds:getranges()
  for d = 1,3 do
    args.bounds[d-1].lo = range[d][1]
    args.bounds[d-1].hi = range[d][2]
  end
  for n, fid in ipairs(task.fields) do
    local field = GetFieldData(fid)
    args.fields[n-1].ptr = field:GetInstance():DataPtrGhostAdjusted()
    args.fields[n-1].dld = field:GetDLD():toTerra()
  end
  -- ACQUIRE SCHEDULER
  gaswrap.acquireScheduler()
  for n, fid in ipairs(task.fields) do
    local field  = GetFieldData(fid)
    local access = task.field_accesses[fid]
    if access.privilege == privileges.read_only then
      local f_read     = field:GetPreviousReadSignal()
      local f_write    = field:ForkPreviousWriteSignal()
      local signals    = terralib.new(gaswrap.Signal[2], {f_read, f_write})
      signals_in[n-1]  = gaswrap.mergeSignals(2, signals)
    else
      local f_read     = field:GetPreviousReadSignal()
      local f_write    = field:GetPreviousWriteSignal()
      local signals    = terralib.new(gaswrap.Signal[2], {f_read, f_write})
      signals_in[n-1]  = gaswrap.mergeSignals(2, signals)
    end
  end
  local signals_in_merge = terralib.new(gaswrap.Signal[num_fields], signals_in)
  local a_in = nil
  if num_fields ~= 0 then
    a_in = gaswrap.mergeSignals(num_fields, signals_in_merge)
  else
    a_in = gaswrap.newSignalSource():trigger()
  end
  -- TODO: use partition id to run task across worker threads
  local a_out        = a_in:exec(0, task.func, args)
  -- record output signal
  if num_fields == 0 then
    a_out:sink()
  elseif num_fields == 1 then
    -- avoid assertion error when forking to just 1 signal
    a_out_forked[0] = a_out
  else
    a_out:fork(num_fields, a_out_forked)
  end
  for n, fid in ipairs(task.fields) do
    local field  = GetFieldData(fid)
    local access = task.field_accesses[fid]
    if access.privilege == privileges.read_only then
      field:RecordRead(a_out_forked[n-1])
    else
      field:RecordReadWrite(a_out_forked[n-1])
    end
  end
  -- RELEASE SCHEDULER
  gaswrap.releaseScheduler()
end

-- Task sequences can be done using:
--   1. update single task launch to support a sequence
--      pros: reduce communication/lua event overhead
--      not sure if we actually need this
--   2. register a new terra task that performs a series of tasks
--      pros: can reorder/transform code when fusing tasks
--      cons: probably not trivial to support any data exchanges within the task


-------------------------------------------------------------------------------
-- Extra Events
-------------------------------------------------------------------------------


-------------------------------------------------------------------------------
-- Register event handlers 
-------------------------------------------------------------------------------

-- relational data
gaswrap.registerLuaEvent('newRelation',         CreateNewRelation)
gaswrap.registerLuaEvent('globalGridPartition', CreateGlobalGridPartition)
gaswrap.registerLuaEvent('recordNewField',      RecordNewField)
gaswrap.registerLuaEvent('allocateField',       AllocateField)
--gaswrap.registerLuaEvent('loadFieldConstant', LoadFieldConstant)

-- task code and data
gaswrap.registerLuaEvent('newTask',             ReceiveNewTask)
gaswrap.registerLuaEvent('launchTask',          LaunchTask)


-------------------------------------------------------------------------------
-- Exports for testing
-------------------------------------------------------------------------------

Exports._TESTING_VroadcastNewRelation         = BroadcastNewRelation
Exports._TESTING_BroadcastGlobalGridPartition = BroadcastGlobalGridPartition
Exports._TESTING_BroadcastNewField            = BroadcastNewField
Exports._TESTING_RemoteAllocateField          = RemoteAllocateField
--Exports._TESTING_remoteLoadFieldConstant      = remoteLoadFieldConstant
Exports._TESTING_BroadcastLuaEventToComputeNodes =
  BroadcastLuaEventToComputeNodes

-------------------------------------------------------------------------------
-- Exports
-------------------------------------------------------------------------------

Exports.N_NODES                       = N_NODES
Exports.THIS_NODE                     = THIS_NODE

Exports.NewGridRelation               = NewGridRelation
Exports.NewField                      = NewField
Exports.RegisterNewTask               = RegisterNewTask

Exports.TaskArgs                      = TaskArgs
Exports.FieldInstance                 = FieldInstance
Exports.RegisterNewTask               = RegisterNewTask
Exports.SendTaskLaunch                = SendTaskLaunch
