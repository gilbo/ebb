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
local CPU                       = 'CPU'
local GPU                       = 'GPU'


-------------------------------------------------------------------------------
-- Helper methods
-------------------------------------------------------------------------------

-- assumption: 0 is control node, remaining are compute nodes
local function numComputeNodes()  -- get total number of compute nodes
  return gas.nodes() - 1
end
local function broadcastLuaEventToComputeNodes(event_name, ...)
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
dld
--]]
local DataInstance = {}
DataInstance.__index = DataInstance

--[[
params {
  bounds,
  ghost_width,
  type_size
}
--]]
local function NewDataInstance(params)
  local range        = params.bounds:getranges()
  local ghost_width  = params.ghost_width
  local n_elems      = 1
  local dim_size     = {}
  local dim_stride   = {}
  for d = 1,3 do
    dim_stride[d]  = n_elems  -- column major?
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
    array = array,
    dld   = dld,
  }, DataInstance)
end

function DataInstance:DataPtr()
  return self.array:_raw_ptr()
end

function DataInstance:DLD()
  return self.dld
end

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
                        array        = nil,
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
  self.instance   = NewDataInstance {
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

-- n+1th signal should not be used by caller
-- it is used here for later depdendencies
function FieldData:ForkPreviousReadSignal(n)
  local signals = terralib.new(gaswrap.Signal[n+1])
  self.last_read:fork(n+1, signals)
  self.last_read = signals[n]
  return signals
end
function FieldData:GetPreviousWriteSignal()
  return self.last_write
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
                        fields    = {},
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
  self.fields[f_id] = NewFieldData(self, name, typ_size)
end

function RelationData:GetFieldData(f_id)
  return self.fields[f_id]
end

function RelationData:GetPartitionBounds()
  return self.partition.bounds
end

function RelationData:GetPartitionMap()
  return self.partition.map
end

-- relation_name -> relation_metadata
local relation_metadata = {}

local function RecordNewRelation(unq_id, name, mode, dims)
  assert(relation_metadata[unq_id] == nil,
         "Recordling already recorded relation '"..name..
         "' with id # "..unq_id)
  relation_metadata[unq_id] = NewRelationData(name, mode, dims)
end

local function GetRelationData(unq_id)
  return relation_metadata[unq_id]
end


-------------------------------------------------------------------------------
-- Relations, fields, partitions
-------------------------------------------------------------------------------

------------------------------------
-- EVENT BROADCASTS/ EVENT HANDLERS
------------------------------------

-- send relation metadata
local function broadcastNewRelation(unq_id, name, mode, dims)
  assert(mode == 'GRID', 'Unsupported relation mode ' .. mode)
  assert(type(dims) == 'table')
  local rel_size = gaswrap.lson_stringify(dims)
  broadcastLuaEventToComputeNodes( 'newRelation', unq_id, name,
                                                  mode, rel_size)
end
-- event handler for relation metadata
local function createNewRelation(unq_id, name, mode, dims)
  RecordNewRelation(tonumber(unq_id), name, mode, gaswrap.lson_eval(dims))
end

-- disjoint partition blocking over relation
-- options: send subregions here or let nodes construct it from the blocking
-- assumption: we'll have only one disjoint partitioning per relation
--             works for static relations, revisit later for dynamic relations
local function sendGlobalGridPartition(
  nid, rel_id, blocking, bid, partition, map_str
)
  local blocking_str  = gaswrap.lson_stringify(blocking)
  local block_id_str  = gaswrap.lson_stringify(bid) 
  assert(Util.isrect2d(partition) or Util.isrect3d(partition))
  local partition_str = gaswrap.lson_stringify(partition:getranges())
  gaswrap.sendLuaEvent(nid, 'globalGridPartition', rel_id, blocking_str,
                       block_id_str, partition_str, map_str)
end
local function broadcastGlobalGridPartition(
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
        sendGlobalGridPartition(nid, rel_id, blocking, {xid, yid},
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
local function createGlobalGridPartition(rel_id, blocking_str,
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
local function broadcastNewField(f_id, rel_id, field_name, type_size)
  broadcastLuaEventToComputeNodes('recordNewField', f_id, rel_id, field_name,
                                                    type_size)
end
local function recordNewField(f_id, rel_id, field_name, type_size)
  local relation = GetRelationData(tonumber(rel_id))
  relation:RecordField(tonumber(f_id), field_name, tonumber(type_size))
end

-- allocate field over remote nodes
-- shared memory, one array across threads for now
local function remoteAllocateField(f_id, rel_id, ghost_width)
  broadcastLuaEventToComputeNodes('allocateField', f_id, rel_id,
                                  gaswrap.lson_stringify(ghost_width))
end
-- event handler to allocate array for a field
local function allocateField(f_id, rel_id, ghost_width_str)
  local rel   = GetRelationData(tonumber(rel_id))
  assert(rel, 'Attempt to allocate for field of '..
              'unrecorded relation #'..rel_id)
  local field = rel:GetFieldData(tonumber(f_id))
  assert(field, 'Attempt to allocate unrecorded field #'..f_id..
                ' on relation '..rel:Name()..' #'..rel_id)
  local ghost_width = gaswrap.lson_eval(ghost_width_str)
  assert(#ghost_width == #rel:Dims(),
         'Relation dimensions ' .. #rel:Dims() ..
         ' do not match ghost width dimensions ' .. #ghost_width)
  field:AllocateInstance(ghost_width)
end

--[[
-- just constant for now
-- can probably convert other kinds of loads to tasks
-- value should be a 2 level list right now, can relax this once we start using
-- luatoebb conversions from ebb.
local function remoteLoadFieldConstant(f_id, rel_id, value)
  assert(false, 'INTERNAL ERROR: Load is broken currently.')
  local value_ser = gaswrap.lson_stringify(value)
  broadcastLuaEventToComputeNodes('loadFieldConstant', f_id, rel_id, value_ser)
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
local function loadFieldConstant(f_id, rel_id, value_ser)
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

  broadcastNewRelation(rel_id, args.name, 'GRID', args.dims)

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
  broadcastNewField(f_id, args.rel.id, args.name, terralib.sizeof(args.type))
  -- allocate memory to back the field
  local ghosts = {}
  for i,_ in ipairs(args.rel.dims) do ghosts[i] = BASIC_SAFE_GHOST_WIDTH end
  remoteAllocateField(args.rel.id, f_id, ghosts)

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
  broadcastGlobalGridPartition(self.id, blocks, bounds, map)
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

local used_task_id    = 0   -- last used task id, for generating ids
local task_id_to_name = {}  -- task id to name
local task_table      = {}  -- map task id to terra code and other task metadata

--[[
A task is defined by:
  * name (optional)
  * task id (auto-generated)
  * terra code
  * rel_id
  * processor
  * field_accesses
This is designed for serializing/deserializing.
--]]
local Task   = {}
Task.__index = Task
local Access = {}
Access.__index = Access

local function NewTask(params)
  assert(params.task_func and params.rel_id and
         params.processor and params.accesses,
         'One or more arguments necessary to define a new task missing.')
  assert(terralib.isfunction(params.task_func) and
         params.task_func:gettype() == ({&opaque} -> {}),
         'Invalid task function to NewTask.')
  assert(params.processor == CPU, 'Only CPU tasks supported right now.')
  used_task_id = used_task_id + 1
  local field_accesses = {}
  return setmetatable({
                        name      = params.name or params.task_func:getname(),
                        task_id   = used_task_id,
                        rel_id    = params.rel_id,
                        processor = params.processor,
                        accesses  = accesses
                      }, Task)
end

-- Event/handler for registering a new task. Returns a task id.
--[[
params : {
  task,
  task_name,
  rel_id,
  processor,
  accesses
}
--]]
local function sendNewTask(params)
  assert(gas.mynode() == CONTROL_NODE,
         'Can send tasks from control node only.')
  local task_name    = params.task_name or params.task:getname()
  local bitcode_ser  = terralib.saveobj(nil, 'bitcode', {[task_name]=params.task})
  used_task_id       = used_task_id + 1
  local task_id      = used_task_id
  assert(params.processor == CPU, 'Tasks over ' .. GPU .. ' not supported yet.')
  local accesses_ser = lson_stringify(accesses)
  broadcastLuaEventToComputeNodes('newTask', task_id, bitcode_ser, task_name,
                                  params.rel_id, params.processor, accesses_ser)
  return task_id
end
local function receiveNewTask(task_id_ser, bitcode_ser, task_name, rel_ser,
                              proc_ser, accesses_ser)
  local bitcode   = terralib.linkllvmstring(bitcode_ser)
  local task_code = bitcode:extern(task_name, {&opaque} -> {})
  task_code:setname(task_name)
  local task_id   = tonumber(task_id_ser)
  assert(not task_id_to_name[task_id],
         'Received task ' .. task_id_ser .. ' : ' .. task_name ..
         ', but already recorded ' .. ' a task with the same task id.')
  task_id_to_name[task_id] = task_name
  task_table[task_id] = task_code
end

-- Send task partitioning over a relation.
local function sendLocalGridPartition(relation, local_partitioning)
end
local function recordLocalGridPartition(rel_ser, local_partitioning_ser)
end

-- Invoke a single task.
-- Task sequences can be done using:
--   1. update this to support a sequence
--      pros: reduce communication/lua event overhead
--      not sure if we actually need this
--   2. register a new terra task that performs a series of tasks
--      pros: can reorder/transform code when fusing tasks
--      cons: probably not trivial to support any data exchanges within the task
local function sendTaskLaunch(task_id, partition_id)
  broadcastLuaEventToComputeNodes('launchTask', task_id)
end
local function launchTask(task_id_ser, partition_id_ser)
  local task_id = tonumber(task_id_ser)
  assert(task_table[task_id], 'Task ' .. task_id_ser ..  ' is not registered.')
  -- TODO: add dependencies, set up args, and enqueue this action
  task_table[task_id](nil)
end

-- TODO: Task sequences




-------------------------------------------------------------------------------
-- Extra Events
-------------------------------------------------------------------------------


-------------------------------------------------------------------------------
-- Register event handlers 
-------------------------------------------------------------------------------

-- relational data
gaswrap.registerLuaEvent('newRelation', createNewRelation)
gaswrap.registerLuaEvent('globalGridPartition', createGlobalGridPartition)
gaswrap.registerLuaEvent('recordNewField', recordNewField)
gaswrap.registerLuaEvent('allocateField', allocateField)
--gaswrap.registerLuaEvent('loadFieldConstant', loadFieldConstant)

-- task code and data
gaswrap.registerLuaEvent('newTask', receiveNewTask)
gaswrap.registerLuaEvent('launchTask', launchTask)


-------------------------------------------------------------------------------
-- Exports for testing
-------------------------------------------------------------------------------

Exports._TESTING_broadcastNewRelation         = broadcastNewRelation
Exports._TESTING_broadcastGlobalGridPartition = broadcastGlobalGridPartition
Exports._TESTING_broadcastNewField            = broadcastNewField
Exports._TESTING_remoteAllocateField          = remoteAllocateField
--Exports._TESTING_remoteLoadFieldConstant      = remoteLoadFieldConstant
Exports._TESTING_broadcastLuaEventToComputeNodes =
  broadcastLuaEventToComputeNodes

-------------------------------------------------------------------------------
-- Exports
-------------------------------------------------------------------------------

Exports.N_NODES                       = N_NODES
Exports.THIS_NODE                     = THIS_NODE

Exports.NewGridRelation               = NewGridRelation
Exports.NewField                      = NewField
