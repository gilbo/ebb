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
local Util    = require 'ebb.src.util'
local ffi     = require 'ffi'

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
-- Basic Prepare  
-------------------------------------------------------------------------------

local N_NODES                   = gas.nodes()
local THIS_NODE                 = gas.mynode()
local CONTROL_NODE              = 0

local BASIC_SAFE_GHOST_WIDTH    = 2

-- constants/modes
local GRID                      = 'GRID'
local CPU                       = tostring(Pre.CPU)
local GPU                       = tostring(Pre.GPU)

local GHOST_DIGITS              = 3
local NODE_DIGITS               = 4

local USE_CONSERVATIVE_GHOSTS   = false
local POLL_USLEEP               = 2


-------------------------------------------------------------------------------
-- Helper methods
-------------------------------------------------------------------------------

-- assumption: 0 is control node, remaining are compute nodes
local function numComputeNodes()  -- get total number of compute nodes
  return gas.nodes() - 1
end
local function BroadcastLuaEventToComputeNodes(event_name, ...)
  --print('*** DEBUG INFO: Sending ' .. event_name)
  for i = 1,numComputeNodes() do
    gaswrap.sendLuaEvent(i, event_name, ...)
  end
end
local function on_control_node() return THIS_NODE == CONTROL_NODE end

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
-- Various Structures
-------------------------------------------------------------------------------

local struct BoundsStruct { lo : int64, hi : int64 }

--[[
_array,
_ghost_width,
_ ... see below
--]]
local FieldInstanceTable = {}
FieldInstanceTable.__index = FieldInstanceTable

local struct FieldInstance {
  ptr       : &uint8;
  strides   : uint64[3];  -- in terms of elements for now
  elem_size : uint64;
}

local struct GlobalInstance {
  ptr       : &uint8;
}

local WorkerField = {}
WorkerField.__index = WorkerField

--[[
params {
  bounds,
  ghost_width,
  type_size
}
--]]
local function NewFieldInstanceTable(params)
  local widths      = params.bounds:getwidths()
  local n_elems     = 1
  local ghost_width = {}
  local dim_size    = {}
  local strides     = {}
  for d = 1,#widths do
    strides[d]      = n_elems   -- column major?
    ghost_width[d]  = params.ghost_width[d] or 0
    dim_size[d]     = widths[d] + 1 -- adjust for hi/lo inclusive convention
    n_elems         = n_elems * (dim_size[d] + 2*ghost_width[d])
  end
  if #widths == 2 then
    strides[3]      = 0
    ghost_width[3]  = 0
    dim_size[3]     = 0
  end

  local elem_size   = params.type_size
  local elem_stride = pow2align(elem_size,4)
  assert(elem_stride == elem_size)
  local array = DynamicArray.New {
    size      = n_elems,
    type      = uint8[elem_size],
    processor = Pre.CPU,
  }
  return setmetatable ({
    _array        = array,
    _strides      = strides,
    _elem_size    = elem_size,
    _ghost_width  = ghost_width,
    _n_elems      = n_elems,
  }, FieldInstanceTable)
end

function FieldInstanceTable:DataPtr()
  return self._array:_raw_ptr()
end

function FieldInstanceTable:DataPtrGhostAdjusted()
  local ptr = terralib.cast(&uint8, self:DataPtr())
  local offset = 0
  for d = 1,#self._strides do
    offset = offset + self._ghost_width[d] * self._strides[d]
  end
  return ptr + offset * self._elem_size
end

function FieldInstanceTable:NElems() return self._n_elems end
function FieldInstanceTable:Strides() return self._strides end
function FieldInstanceTable:ElemSize() return self._elem_size end



-------------------------------------------------------------------------------
-- GHOSTS
-------------------------------------------------------------------------------


local struct GhostInstance  {
  union {
    src : &gaswrap.AsyncBufSrcChannel;
    dst : &gaswrap.AsyncBufDstChannel;
  }
  bounds    : BoundsStruct[3];
  ptr       : &uint8;
  strides   : uint64[3];  -- in element counts, matching field instances
  elem_size : uint64;
}
local terra printGI( g : &GhostInstance, prefix : rawstring )
  var prefix_skip = ""
  if prefix ~= nil then prefix_skip = "\n    "
                   else prefix = "" end
  C.printf(["[%d] %s%s"..
                 "chan, buf, elem_size:   %p %p %d\n"..
             "    bds ; strides:          %d,%d %d,%d %d,%d ; %d,%d,%d\n"
           ], THIS_NODE, prefix, prefix_skip, g.src, g.ptr, g.elem_size,
           g.bounds[0].lo, g.bounds[0].hi,
           g.bounds[1].lo, g.bounds[1].hi,
           g.bounds[2].lo, g.bounds[2].hi,
           g.strides[0], g.strides[1], g.strides[2]
           )
end


local controller_field_ghost_ready_semaphores = {}

local function InitFieldGhostSemaphore(f_id)
  assert(on_control_node(), 'only for controller')
  controller_field_ghost_ready_semaphores[f_id] = numComputeNodes()
end

local function MarkFieldGhostsReady(f_id, hid_base)
  f_id = tonumber(f_id)
  local semaphore_val = controller_field_ghost_ready_semaphores[f_id]
  controller_field_ghost_ready_semaphores[f_id] = semaphore_val - 1
  --print(THIS_NODE, 'DECREMENT f#'..f_id..': '..(semaphore_val-1))
end
gaswrap.registerLuaEvent('MarkFieldGhostsReady', MarkFieldGhostsReady)

local function WaitOnFieldGhostReady(f_id)
  assert(on_control_node(), 'only for controller')
  local semaphore_val = controller_field_ghost_ready_semaphores[f_id]
  assert(semaphore_val, 'tried to wait on uninitialized field '..f_id)
  while semaphore_val > 0 do
    C.usleep(POLL_USLEEP)
    gaswrap.pollLuaEvents(0, 0)
    semaphore_val = controller_field_ghost_ready_semaphores[f_id]
  end
  --print(THIS_NODE, 'DONE waiting on f#'..f_id)
end


-- Allocate incoming and outgoing buffers for data exchange, set up channels
-- with neighboring nodes, and send a confirmation to control node when all
-- channels are set up.
local ghost_listing_2d = newlist()
local ghost_listing_3d = newlist()
if USE_CONSERVATIVE_GHOSTS then
  for x = -1,1 do
    for y = -1,1 do
      if x ~= 0 or y ~= 0 then
        ghost_listing_2d:insert({x,y})
      end
      for z = -1,1 do
        if x ~= 0 or y ~= 0 or z ~= 0 then
          ghost_listing_3d:insert({x,y,z})
        end
      end
    end
  end
  assert(#ghost_listing_2d == 8)
  assert(#ghost_listing_3d == 26)
else
  for d = 1,2 do
    local neg, pos = {0,0}, {0,0}
    neg[d] = -1
    pos[d] =  1
    ghost_listing_2d:insertall({neg, pos})
  end
  for d = 1,3 do
    local neg, pos = {0,0,0}, {0,0,0}
    neg[d] = -1
    pos[d] =  1
    ghost_listing_3d:insertall({neg, pos})
  end
  assert(#ghost_listing_2d == 4)
  assert(#ghost_listing_3d == 6)
end

local worker_ghost_channel_semaphores = {} -- keyed by field id
local function ProcessAllGhostChannels(field, handshake_unq_id, ghost_width)
  local relation      = field.relation
  local ndims         = #relation.dims
  local off_patterns  = ndims == 2 and ghost_listing_2d or ghost_listing_3d
  worker_ghost_channel_semaphores[field.id] = 2 * #off_patterns
  local function dec_semaphore()
    local semaphore_val = worker_ghost_channel_semaphores[field.id] - 1
    worker_ghost_channel_semaphores[field.id] = semaphore_val
    if semaphore_val == 0 then
      gaswrap.sendLuaEvent(CONTROL_NODE, 'MarkFieldGhostsReady', field.id)
    end
  end

  local function gen_handshake_id(src_node, off, flip_off)
    local h_id  = handshake_unq_id
    h_id        = src_node + N_NODES * h_id
    local o     = { off[1], off[2], off[3] or 0 }
    if flip_off then for d=1,3 do o[d] = -o[d] end end
    h_id        = (3*3)*(o[1]+1) + 3*(o[2]+1) + (o[3]+1) + (3*3*3)*h_id
    return h_id
  end

  local send_ghosts   = C.safemalloc(GhostInstance, #off_patterns)
  local recv_ghosts   = C.safemalloc(GhostInstance, #off_patterns)
  for i,off in ipairs(off_patterns) do
    -- Compute:
    --    node to connect to
    --    index bounds for the ghost cell region (send and recv)
    --    strides for the ghost cell region
    --    number of elements in the ghost cell region
    local other_node
    do
      local blockDims   = relation:GetPartitionBlockDims()
      local bid         = relation:GetPartitionBlockId()
      local nid         = {}
      for d=1,ndims do
        nid[d] = (bid[d]+off[d]-1) % blockDims[d] + 1
      end
      other_node        = relation:GetNodeIdFromMap(nid)
    end
    local gsend_bounds, grecv_bounds, gstrides, n_elems
    do
      local bds         = relation:GetPartitionBounds():getranges()
      gsend_bounds      = {}
      grecv_bounds      = {}
      gstrides          = {}
      n_elems           = 1
      for d=1,ndims do
        local gw      = ghost_width[d]
        gstrides[d]   = n_elems
        if off[d] == 0 then
          gsend_bounds[d] = bds[d]
          grecv_bounds[d] = bds[d]
          n_elems         = n_elems * (bds[d][2]-bds[d][1]+1)
        elseif off[d] == -1 then
          gsend_bounds[d] = { bds[d][1], bds[d][1] + gw - 1 }
          grecv_bounds[d] = { bds[d][1] - gw, bds[d][1] - 1 }
          n_elems         = gw * n_elems
        elseif off[d] == 1 then
          gsend_bounds[d] = { bds[d][2] - gw + 1, bds[d][2] }
          grecv_bounds[d] = { bds[d][2] + 1, bds[d][2] + gw }
          n_elems         = gw * n_elems
        else assert(false,'bad offset pattern') end
      end
    end

    -- Compute other derived data
    local elem_size   = field:GetTypeSize()
    local buf_size    = n_elems * elem_size

    -- Create Send Channel
    local sendg     = send_ghosts[i-1]
    if other_node == THIS_NODE then
      sendg.ptr = nil
      dec_semaphore()
    else
      sendg.ptr       = terralib.cast(&uint8, C.malloc(buf_size))
      sendg.elem_size = elem_size
      for d = 1,ndims do
        sendg.bounds[d-1].lo  = gsend_bounds[d][1]
        sendg.bounds[d-1].hi  = gsend_bounds[d][2]
        sendg.strides[d-1]    = gstrides[d]
      end
      if ndims == 2 then
        sendg.bounds[2].lo  = 0
        sendg.bounds[2].hi  = 0
        sendg.strides[2]    = 0
      end
      gaswrap.CreateAsyncBufSrcChannel(
        other_node, gen_handshake_id(THIS_NODE, off, false),
        sendg.ptr, buf_size,
        function(chan)  sendg.src = chan; dec_semaphore() end)
    end
    if field.id == 11 then
      --printGI(sendg,'process send f#'..field.id..' g '..i)
    end


    -- Create Recv Channel
    local recvg     = recv_ghosts[i-1]
    if other_node == THIS_NODE then
      recvg.ptr = nil
      dec_semaphore()
    else
      recvg.ptr       = terralib.cast(&uint8, C.malloc(buf_size))
      recvg.elem_size = elem_size
      for d = 1,ndims do
        recvg.bounds[d-1].lo  = grecv_bounds[d][1]
        recvg.bounds[d-1].hi  = grecv_bounds[d][2]
        recvg.strides[d-1]    = gstrides[d]
      end
      if ndims == 2 then
        recvg.bounds[2].lo  = 0
        recvg.bounds[2].hi  = 0
        recvg.strides[2]    = 0
      end
      gaswrap.CreateAsyncBufDstChannel(
        other_node, gen_handshake_id(other_node, off, true),
        recvg.ptr, buf_size,
        function(chan)  recvg.dst = chan; dec_semaphore() end)
    end
    if field.id == 11 then
      --printGI(recvg,'process recv f#'..field.id..' g '..i)
    end
  end
  field.send_ghosts         = send_ghosts
  field.recv_ghosts         = recv_ghosts
  field.num_send_ghosts     = #off_patterns
  field.num_recv_ghosts     = #off_patterns
end



-------------------------------------------------------------------------------
-- Relation and Field objects at compute nodes
-------------------------------------------------------------------------------

local worker_relation_metadata = {}
local worker_field_metadata    = {}

--[[
WorkerField : {
  id          = #,
  type_size   = #, (bytes)
  name        = 'string',
  relation    = WorkerRelation,
  instance    = FieldInstance,
  last_read   = Signal,
  last_write  = Signal,
}
--]]

local function NewWorkerField(relation, params)
  return setmetatable({
    id                = params.id,
    type_size         = params.type_size,
    name              = params.name,
    relation          = relation,
    instance          = nil,
    last_read         = nil,
    last_write        = nil,
    send_ghosts       = nil,
    recv_ghosts       = nil,
    num_send_ghosts   = 0,
    num_recv_ghosts   = 0,
  }, WorkerField)
end

function WorkerField:Name()
  return self.name
end

function WorkerField:isAllocated()
  return (self.instance ~= nil)
end

function WorkerField:AllocateInstance(gw)
  assert(self.array == nil, 'Trying to allocate already allocated array.')
  assert(self.relation:isPartitioned(),
         'Relation ' .. self.relation:Name() .. ' not partitioned. ' ..
         'Cannot allocate data over it.')
  self.instance   = NewFieldInstanceTable {
                      bounds      = self.relation:GetPartitionBounds(),
                      ghost_width = gw,
                      type_size   = self:GetTypeSize(),
                    }
  self.last_read  = gaswrap.newSignalSource():trigger()
  self.last_write = gaswrap.newSignalSource():trigger()
end

function WorkerField:GetTypeSize()
  return self.type_size
end
function WorkerField:GetTypeStride()
  return self.instance:ElemSize()
end

function WorkerField:GetInstance()
  return self.instance
end

function WorkerField:GetDataPtr()
  return self.instance:DataPtr()
end
function WorkerField:GetElemCount() -- includes ghosts
  return self.instance:NElems()
end
function WorkerField:GetStrides()
  return self.instance:Strides()
end

function WorkerField:nDims()  return #self.relation.dims  end

function WorkerField:GetReadWriteSignal()
  --local signals = C.safemalloc(gaswrap.Signal, 2)
  --signals[0] = self.last_read
  --signals[1] = self.last_write
  --print('['..THIS_NODE..'] Getting '..self.id..': '..self.last_read.id..
  --                                              ' '..self.last_write.id)
  --local signal = gaswrap.mergeSignals(2, signals)
  local signal = gaswrap.luaMergeSignals{ self.last_read, self.last_write }
  --print('['..THIS_NODE..'] Got '..self.id)
  return signal
end
function WorkerField:GetReadSignal()
  return self.last_read
end
function WorkerField:GetWriteSignal()
  return self.last_write
end
function WorkerField:ForkReadSignal()
  --print('['..THIS_NODE..'] fork read: '..self.id)
  --local signals = C.safemalloc(gaswrap.Signal, 2)
  local signals = gaswrap.luaForkSignals(self.last_read, 2)
  --self.last_read:fork(2, signals)
  self.last_read = signals[2]
  return signals[1]
end
function WorkerField:ForkWriteSignal()
  --print('['..THIS_NODE..'] fork write: '..self.id)
  --local signals = C.safemalloc(gaswrap.Signal, 2)
  local signals = gaswrap.luaForkSignals(self.last_write, 2)
  --self.last_write:fork(2, signals)
  self.last_write = signals[2]
  return signals[1]
end
function WorkerField:MergeReadSignal(signal)
  --print('['..THIS_NODE..'] merge read: '..self.id)
  --local signals = C.safemalloc(gaswrap.Signal, 2)
  self.last_read = gaswrap.luaMergeSignals{ self.last_read, signal }
  --signals[0] = self.last_read
  --signals[1] = signal
  --self.last_read = gaswrap.mergeSignals(2, signals)
end
function WorkerField:MergeWriteSignal(signal)
  self.last_write = gaswrap.luaMergeSignals{ self.last_write, signal }
  --local signals = C.safemalloc(gaswrap.Signal, 2)
  --signals[0] = self.last_write
  --signals[1] = signal
  --self.last_write = gaswrap.mergeSignals(2, signals)
end
function WorkerField:SetReadWriteSignal(signal)
  local signals = gaswrap.luaForkSignals(signal, 2)
  --local signals = C.safemalloc(gaswrap.Signal, 2)
  --signal:fork(2, signals)
  self.last_read  = signals[1]
  self.last_write = signals[2]
  --print('['..THIS_NODE..'] SET RW '..self.id..': '..self.last_read.id..
  --                                              ' '..self.last_write.id)
end
function WorkerField:SetReadSignal(signal)
  self.last_read = signal
end
function WorkerField:SetWriteSignal(signal)
  self.last_write = signal
end
function WorkerField:GetAllSignals()
  return self:GetReadWriteSignal()
end
function WorkerField:SetAllSignals(signal)
  return self:SetReadWriteSignal(signal)
end



--[[
WorkerRelation : {
  name        = 'string',
  mode        = GRID,
  dims        = { #, #, ? },
  partition   = { blocking, block_id, bounds, map },
  fields      = list{ WorkerField },
}
--]]
local WorkerRelation = {}
WorkerRelation.__index = WorkerRelation

local function NewWorkerRelation(params)
  assert(params.mode == GRID, 'Relations must be of grid type on GASNet.')
  return setmetatable({
                        id        = params.id,
                        name      = params.name,
                        mode      = params.mode,
                        dims      = params.dims,
                        periodic  = params.periodic,
                        partition = nil,
                        fields    = newlist(),
                      }, WorkerRelation)
end

function WorkerRelation:Name()
  return self.name
end

function WorkerRelation:Dims()
  return self.dims
end

function WorkerRelation:Fields()
  return self.fields
end

function WorkerRelation:RecordPartition(blocking, block_id, bounds, map)
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
    blocking = blocking,  -- {#,#,?}
    block_id = block_id,  -- {#,#,?} (where in block grid this node is)
    bounds   = bounds,    -- 2 or 3 ranges specifying block coord bounds
    map      = map,       -- global map from block grid ids to node id #s
  }
end

function WorkerRelation:isPartitioned()
  return self.partition
end

function WorkerRelation:RecordField(params)
  local f_id = params.id
  assert(self.fields[f_id] == nil, "Recording already recorded field '"..
                                   params.name.."' with id # "..f_id)
  self.fields:insert(f_id)
  worker_field_metadata[f_id] = NewWorkerField(self, params)
end

function WorkerRelation:GetPartition()
  return self.partition
end

function WorkerRelation:GetPartitionBounds()
  return self.partition.bounds
end
function WorkerRelation:GetPartitionBlockId()
  return self.partition.block_id
end
function WorkerRelation:GetPartitionBlockDims()
  return self.partition.blocking
end
function WorkerRelation:GetNodeIdFromMap(nid)
  local id = self.partition.map
  for _,i in ipairs(nid) do id = id[i] end
  return id
end
function WorkerRelation:GetPartitionMap()
  return self.partition.map
end

local function RecordNewRelation(params)
  local unq_id = params.id
  assert(worker_relation_metadata[unq_id] == nil,
         "Recordling already recorded relation '"..params.name..
         "' with id # "..unq_id)
  worker_relation_metadata[unq_id] = NewWorkerRelation(params)
end

local function GetWorkerRelation(unq_id)
  return worker_relation_metadata[unq_id]
end

function GetWorkerField(f_id)
  return worker_field_metadata[f_id]
end

local function GetAllWorkerFields()
  local fs = newlist()
  for _,f in pairs(worker_field_metadata) do fs:insert(f) end
  return fs
end


-------------------------------------------------------------------------------
-- Copying ghost data, sending, receiving
-------------------------------------------------------------------------------

local struct GhostCopyArgs {
  num_ghosts   : uint32;
  ghosts       : &GhostInstance;
  field_bounds : BoundsStruct[3];
  field        : FieldInstance;
}

-- send ghosts
-- TODO: incomplete
-- one or multiple actions per ghost?
local CopyGridGhosts = {}
for _,sendrecv in ipairs({'Send','Recv'}) do
  CopyGridGhosts[sendrecv] = terra (copy_args : &opaque)
    --C.printf('*** CopyGridGhosts executing for %s.\n', sendrecv)
    var args = [&GhostCopyArgs](copy_args)
    var field        = args.field
    var fx_lo, fx_hi = args.field_bounds[0].lo, args.field_bounds[0].hi 
    var fy_lo, fy_hi = args.field_bounds[1].lo, args.field_bounds[1].hi 
    var fz_lo, fz_hi = args.field_bounds[2].lo, args.field_bounds[2].hi 
    var fstride      = field.strides
    for i = 0, args.num_ghosts do
      var ghost : GhostInstance = args.ghosts[i]
      var gx_lo, gx_hi = ghost.bounds[0].lo, ghost.bounds[0].hi 
      var gy_lo, gy_hi = ghost.bounds[1].lo, ghost.bounds[1].hi 
      var gz_lo, gz_hi = ghost.bounds[2].lo, ghost.bounds[2].hi 
      var gstride      = ghost.strides
      if ghost.ptr ~= nil then
        for z = gz_lo, gz_hi + 1 do
          for y = gy_lo, gy_hi + 1 do
            for x = gx_lo, gx_hi + 1 do
              var foff = ((x-fx_lo)*fstride[0] + (y-fy_lo)*fstride[1] +
                          (z-fz_lo)*fstride[2]) * field.elem_size
              var goff = ((x-gx_lo)*gstride[0] + (y-gy_lo)*gstride[1] +
                          (z-gz_lo)*gstride[2]) * ghost.elem_size
              escape
                if sendrecv == 'Send' then
                  emit quote
                    C.memcpy(ghost.ptr + goff, field.ptr + foff, ghost.elem_size)
                  end
                else
                  emit quote
                    C.memcpy(field.ptr + foff, ghost.ptr + goff, ghost.elem_size)
                  end
                end
              end
            end
          end
        end
      end
    end
    C.free(copy_args)
    --C.printf('*** CopyGridGhosts done for %s.\n', sendrecv)
  end
end

-- start_sig = task's write
-- merge output signal with read
-- caller must acquire scheduler
local terra CopyAndSendGridGhosts(start_sig : gaswrap.Signal,
                                  copy_args : &GhostCopyArgs) : gaswrap.Signal
  var n_ghosts            = copy_args.num_ghosts
  var ghosts              = copy_args.ghosts
  var worker_id : uint32  = 0

  -- Schedule a big copy and then individual sends
  var sigs  = [&gaswrap.Signal](C.malloc(n_ghosts * sizeof(gaswrap.Signal)))
  start_sig:exec(worker_id, CopyGridGhosts.Send, [&opaque](copy_args))
           :fork(n_ghosts, sigs)
  for i = 0, n_ghosts do if ghosts[i].ptr ~= nil then
    sigs[i] = ghosts[i].src:send(sigs[i])
  end end
  var done = gaswrap.mergeSignals(n_ghosts, sigs)
  C.free(sigs)

  return done
end
-- start_sig = previous read, before the task preceeding this exchange
-- merge output signal with task's write
-- caller must acquire scheduler
local terra RecvAndCopyGridGhosts(start_sig : gaswrap.Signal,
                                  copy_args : &GhostCopyArgs) : gaswrap.Signal
  var n_ghosts            = copy_args.num_ghosts
  var ghosts              = copy_args.ghosts
  var worker_id : uint32  = 0

  -- schedule receives and merge with the start signal
  -- then have the bulk copy execute after all of those
  var sigs = [&gaswrap.Signal](C.malloc(sizeof(gaswrap.Signal)*(n_ghosts+1)))
  sigs[n_ghosts] = start_sig
  for i = 0, n_ghosts do
    if ghosts[i].ptr ~= nil then sigs[i] = ghosts[i].dst:recv()
    else                    sigs[i] = gaswrap.newSignalSource():trigger() end
  end
  var done  = gaswrap.mergeSignals(n_ghosts+1, sigs)
                :exec(worker_id, CopyGridGhosts.Recv, [&opaque](copy_args))
  C.free(sigs)
  --var done  = start_sig:exec(worker_id, CopyGridGhosts.Recv,
  --                                      [&opaque](copy_args))

  return done
end

-- worker field as argument
-- NOTE: SCHEDULER MUST BE ACQUIRED OUTSIDE THIS FUNCTION
local function scheduleSendAndRecv(wfield)
  assert(getmetatable(wfield) == WorkerField)
  print('['..THIS_NODE..'] sched Send/Recv f#'..wfield.id)
  -- fill out argument structures
  local local_ptr   = wfield:GetInstance():DataPtrGhostAdjusted()
  local elem_size   = wfield:GetTypeStride()
  local f_strides   = wfield:GetStrides()
  local range       = wfield.relation:GetPartitionBounds():getranges()
  local function allocCopyArgs(n_ghosts, ghosts)
    local a = terralib.cast(&GhostCopyArgs,
                            C.malloc(terralib.sizeof(GhostCopyArgs)))
    --print(THIS_NODE, 'alloc...', a)
    a.num_ghosts      = n_ghosts
    a.ghosts          = ghosts
    a.field.ptr       = local_ptr
    a.field.elem_size = elem_size
    for d=1,#range do
      a.field.strides[d-1]    = f_strides[d]
      a.field_bounds[d-1].lo  = range[d][1]
      a.field_bounds[d-1].hi  = range[d][2]
    end
    if #range == 2 then
      a.field.strides[2]    = 0
      a.field_bounds[2].lo  = 0
      a.field_bounds[2].hi  = 0
    end
    return a
  end

  --print(THIS_NODE, 'snd/recv', wfield.send_ghosts, wfield.recv_ghosts)
  local sarg  = allocCopyArgs(wfield.num_send_ghosts, wfield.send_ghosts)
  local rarg  = allocCopyArgs(wfield.num_recv_ghosts, wfield.recv_ghosts)
  for i=1,wfield.num_send_ghosts do
    --printGI(wfield.send_ghosts[i-1],'sched send f#'..wfield.id..' g '..i)
  end
  for i=1,wfield.num_recv_ghosts do
    --printGI(wfield.recv_ghosts[i-1],'sched recv f#'..wfield.id..' g '..i)
  end

  -- schedule the send
  wfield:SetReadSignal(CopyAndSendGridGhosts(wfield:GetReadSignal(), sarg))
  wfield:SetWriteSignal(RecvAndCopyGridGhosts(wfield:GetWriteSignal(), rarg))
end


-------------------------------------------------------------------------------
-- Relations, fields, partitions
-------------------------------------------------------------------------------

------------------------------------
-- EVENT BROADCASTS/ EVENT HANDLERS
------------------------------------

-- send relation metadata
local function BroadcastNewRelation(unq_id, name, mode, dims, periodic)
  assert(mode == GRID, 'Unsupported relation mode ' .. mode)
  assert(type(dims) == 'table')
  local dims_ser     = gaswrap.lson_stringify(dims)
  local periodic_ser = gaswrap.lson_stringify(periodic)
  BroadcastLuaEventToComputeNodes( 'newRelation', unq_id, name,
                                                  mode, dims_ser, periodic_ser)
end
-- event handler for relation metadata
local function CreateNewRelation(unq_id, name, mode, dims, periodic)
  RecordNewRelation {
    id       = tonumber(unq_id),
    name     = name,
    mode     = mode,
    dims     = gaswrap.lson_eval(dims),
    periodic = gaswrap.lson_eval(periodic),
  }
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
      for yid, subm in ipairs(mt) do
        for zid, nid in ipairs(subm) do
          assert(nid >= 1 and nid <= numComputeNodes())
          SendGlobalGridPartition(nid, rel_id, blocking, {xid, yid, zid},
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
  local relation = GetWorkerRelation(tonumber(rel_id))
  assert(relation,
         'Relation #' .. rel_id .. ' to partition is not defined.')
  assert(relation.block_id == nil,
         'Already recorded a partition. ' ..
         'Are multiple partitions mapped to this node?')
  local blocking  = gaswrap.lson_eval(blocking_str)
  local block_id  = gaswrap.lson_eval(blocking_id_str)
  local range     = gaswrap.lson_eval(partition_str)
  local bounds    = (range[3]) and Util.NewRect3d(unpack(range))
                                or Util.NewRect2d(unpack(range))
  local map       = gaswrap.lson_eval(map_str)
  relation:RecordPartition(blocking, block_id, bounds, map)
end

-- record a new field over a relation
local function BroadcastNewField(f_id, rel_id, field_name, type_size)
  BroadcastLuaEventToComputeNodes('recordNewField', f_id, rel_id, field_name,
                                                    type_size)
end
local function RecordNewField(f_id, rel_id, field_name, type_size)
  local relation = GetWorkerRelation(tonumber(rel_id))
  relation:RecordField {
    id        = tonumber(f_id),
    name      = field_name,
    type_size = tonumber(type_size)
  }
end

local hid_base_used = 0

-- allocate field array over remote nodes and channels between neighbors
-- shared memory, one array across threads for now
local function RemotePrepareField(f_id, ghost_width, hid_base)
  hid_base_used = hid_base_used + 1
  BroadcastLuaEventToComputeNodes('prepareField', f_id,
                                  gaswrap.lson_stringify(ghost_width),
                                  hid_base_used)
end
-- event handler to allocate arrays, and channels for a field
local function PrepareField(f_id, ghost_width_str, hid_base)
  local field = GetWorkerField(tonumber(f_id))
  assert(field, 'Attempt to allocate unrecorded field #'..f_id)
  local ghost_width = gaswrap.lson_eval(ghost_width_str)
  field:AllocateInstance(ghost_width)
  ProcessAllGhostChannels(field, tonumber(hid_base), ghost_width)
end


-- dst length should be 2*n+1, twice src
local terra binary_stringify_terra( dst : rawstring, n : uint32, src : &uint8 )
  for k=0,n do
    var byte    = src[k]
    var lo      = [int8](byte and [uint8](0x0F))
    var hi      = [int8](byte and [uint8](0xF0)) / [uint8](0x10)
    dst[2*k+0]  = [int8](0x40) + lo
    dst[2*k+1]  = [int8](0x40) + hi
  end
  dst[2*n] = [int8](0)
end
-- src length should be 2*n, twice dst
local terra binary_destring_terra( dst : &uint8, n : uint32, src : rawstring )
  for k=0,n do
    var lo      = [uint8](src[2*k+0]) and [uint8](0x0F)
    var hi      = [uint8](src[2*k+1]) and [uint8](0x0F)
    var byte    = [uint8](0x10) * hi + lo
    dst[k]      = byte
  end
end

local function stringify_binary_data( n_bytes, valptr )
  local tmpstr      = terralib.cast(rawstring, C.malloc(n_bytes*2+1))
  binary_stringify_terra(tmpstr, n_bytes, terralib.cast(&uint8, valptr))
  local str         = ffi.string(tmpstr, n_bytes*2)
  C.free(tmpstr)
  return str
end

--[[
local function stringify_binary_data( n_bytes, valptr )
  local n_words   = math.ceil(n_bytes / 4)
  local tmp_buf   = terralib.cast(&uint32, C.malloc(n_words*4))
  assert(n_words > 0)
  tmp_buf[n_words-1] = 0 -- zero out where memcpy may not initialize
  C.memcpy(tmp_buf, valptr, n_bytes)

  local num_list  = newlist()
  for k=1,n_words do num_list[k] = tmp_buf[k-1] end
  C.free(tmp_buf)
  return gaswrap.lson_stringify(num_list)
end
local function destringify_binary_data( valptr, n_bytes, valstr )
  local num_list  = gaswrap.lson_eval(valstr)
  local n_words   = math.ceil(n_bytes / 4)
  local tmp_buf   = terralib.cast(&uint32, C.malloc(n_words*4))
  for k=1,n_words do tmp_buf[k-1] = num_list[k] end
  C.memcpy(valstr, tmp_buf, n_bytes)
  C.free(tmp_buf)
end
--]]

--
-- just constant for now
local function RemoteLoadFieldConstant(f_id, n_bytes, valptr)
  local value_ser = stringify_binary_data(n_bytes, valptr)
  BroadcastLuaEventToComputeNodes('loadFieldConstant', f_id, value_ser)
end
local struct LoadConstArgs{
  data_ptr  : &uint8
  valptr    : &uint8
  val_size  : uint32
  n_elems   : uint32
}
local terra load_field_constant_action(raw_args : &opaque)
  var args        = [&LoadConstArgs](raw_args)
  var data_ptr    = args.data_ptr
  var valptr      = args.valptr
  var val_size    = args.val_size
  var n_elems     = args.n_elems
  -- fill out array
  for i=0, n_elems do
    for k=0, val_size do
      data_ptr[k] = valptr[k]
    end
    data_ptr = data_ptr + val_size
  end
  -- free
  C.free(valptr)
  C.free(args)
end
local terra alloc_temp_val( n_bytes : uint32 ) : &uint8
  var p = [&uint8](C.malloc(n_bytes))
  for k=0,n_bytes do p[k] = 0 end
  return p
end
local terra alloc_LoadConstArgs()
  return [&LoadConstArgs](C.malloc(sizeof(LoadConstArgs)))
end
local function LoadFieldConstant(f_id, value_ser)
  local field    = GetWorkerField(tonumber(f_id))
  assert(field, 'Attempt to load into unrecorded field ' .. f_id .. '.')
  assert(field:isAllocated(), 'Attempt to load into unallocated field ' ..
                              field:Name())

  local args        = alloc_LoadConstArgs()
  args.data_ptr     = terralib.cast(&uint8, field:GetDataPtr())
  local elem_stride = field:GetTypeStride()
  -- fill out the value with up-to-stride 0-padded
  args.valptr       = alloc_temp_val(elem_stride)
  binary_destring_terra(args.valptr, field:GetTypeSize(), value_ser)
  args.val_size     = elem_stride
  args.n_elems      = field:GetElemCount()

  -- schedule action to load the field.
  local worker_id = 0
  -- single threaded right now
  gaswrap.acquireScheduler()
  local a_in  = field:GetReadWriteSignal()
  local a_out = a_in:exec(worker_id,
                          load_field_constant_action:getpointer(),
                          args)
  field:SetReadWriteSignal(a_out)
  gaswrap.releaseScheduler()
end

local function LoadBoolFieldFromRects(f_id, rects_ser)
  local rects = gaswrap.lson_eval(rects_ser)
  local field = GetWorkerField(assert(tonumber(f_id)))
  assert(field:isAllocated(), 'Attempt to load into unallocated field ' ..
                              field:Name())

  -- first, route the field signals back into the control plane
  gaswrap.acquireScheduler()
  field:GetReadWriteSignal():sink()
  local done_loading = gaswrap.newSignalSource()
  field:SetReadWriteSignal(done_loading)
  gaswrap.releaseScheduler()

  -- do the load
  local dataptr     = terralib.cast(&bool, field:GetDataPtr())
  local n_elems     = field:GetElemCount()
  local bound_rect  = field.relation:GetPartitionBounds()
  local n_dims      = #field.relation.dims
  -- first, zero out the data
  for i = 0, n_elems-1 do dataptr[i] = false end
  -- adjust the location of the data ptr
  dataptr           = terralib.cast(&bool,
                                field:GetInstance():DataPtrGhostAdjusted())
  local strides     = field:GetStrides()
  local offset      = 0
  for d=1,n_dims do 
    offset = offset + strides[d] * select(d, bound_rect:mins())
  end
  dataptr           = dataptr - offset
  -- then go through and raster each rectangle
  for _,r in ipairs(rects) do
    if n_dims == 2 then
      local clipped = Util.NewRect2d(unpack(r)):clip(bound_rect):getranges()
      for y = clipped[2][1],clipped[2][2] do
        for x = clipped[1][1],clipped[1][2] do
          dataptr[ x * strides[1] + y * strides[2] ] = true
      end end
    else assert(n_dims == 3)
      local clipped = Util.NewRect3d(unpack(r)):clip(bound_rect):getranges()
      for z = clipped[3][1],clipped[3][2] do
        for y = clipped[2][1],clipped[2][2] do
          for x = clipped[1][1],clipped[1][2] do
            dataptr[ x * strides[1] + y * strides[2] + z * strides[3] ] = true
      end end end
    end
  end

  -- trigger that the load is complete
  gaswrap.acquireScheduler()
  done_loading:trigger()
  gaswrap.releaseScheduler()
end
gaswrap.registerLuaEvent('LoadBoolFieldFromRects', LoadBoolFieldFromRects)


-----------------------------------
-- HELPER METHODS FOR CONTROL NODE
-----------------------------------

local controller_relations = {}
local controller_fields    = {}

--[[
  id,
  dims,
  _partition_map,
  _partition_bounds,
--]]
local ControllerGridRelation    = {}
ControllerGridRelation.__index  = ControllerGridRelation
local function is_cgrid_relation(obj)
  return getmetatable(obj) == ControllerGridRelation
end

--[[
  rel_id,
  id,
  type,
--]]
local ControllerField   = {}
ControllerField.__index = ControllerField
local function is_cfield(obj) return getmetatable(obj) == ControllerField end

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

  local periodic = args.periodic or (#args.dims == 2 and {false,false}
                                                      or {false,false,false})
  BroadcastNewRelation(rel_id, args.name, GRID, args.dims, periodic)

  local rel = setmetatable({
    id       = rel_id,
    dims     = args.dims,
    periodic = periodic,
  }, ControllerGridRelation)
  controller_relations[rel.id] = rel
  return rel
end

--[[
  name = 'string'
  rel  = ControllerGridRelation
  type = TerraType
--]]
local field_id_counter = 1
local function NewField(args)
  assert(type(args)=='table','expected table')
  assert(type(args.name) == 'string',"expected 'name' string arg")
  assert(is_cgrid_relation(args.rel),"expected 'rel' relation arg")
  assert(terralib.types.istype(args.type),"expected 'type' terra type arg")

  local f_id        = field_id_counter
  field_id_counter  = f_id + 1

  -- create the field
  BroadcastNewField(f_id, args.rel.id, args.name, terralib.sizeof(args.type))
  -- allocate memory to back the field
  local ghosts = {}
  for i,_ in ipairs(args.rel.dims) do ghosts[i] = BASIC_SAFE_GHOST_WIDTH end
  RemotePrepareField(f_id, ghosts, f_id)

  local f = setmetatable({
    id      = f_id,
    rel_id  = args.rel.id,
    type    = args.type,
  }, ControllerField)
  InitFieldGhostSemaphore(f_id)

  return f
end

--[[
  blocking = {#,#,?} -- dimensions of grid of blocks
--]]
function ControllerGridRelation:partition_across_nodes(args)
  assert(type(args)=='table','expected table')
  assert(is_2_nums(args.blocking) or is_3_nums(args.blocking),
         "expected 'blocking' arg: list of 2 or 3 numbers")
  local dims    = self.dims
  local blocks  = args.blocking
  assert(#dims == #blocks, 'dimension of blocking did not '..
                           'match dimension of grid')
  local bounds  = {}
  local map     = {}
  for i=1,#dims do
    assert(dims[i] >= blocks[i],
           '# cells < # blocks ('..dims[i]..' <= '..blocks[i]..')')
  end

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
function ControllerGridRelation:partition_within_nodes()
end

function ControllerField:LoadConst( c_val )
  local typsize   = terralib.sizeof(self.type)
  local temp_mem  = terralib.cast(&self.type, C.malloc(typsize))
  temp_mem[0]     = c_val
  RemoteLoadFieldConstant(self.id, typsize, temp_mem)
  C.free(temp_mem)
end

function ControllerField:LoadBoolFromRects( rects )
  local rawrects = newlist()
  for i,r in ipairs(rects) do rawrects[i] = r:getranges() end
  local rects_ser = gaswrap.lson_stringify(rawrects)
  BroadcastLuaEventToComputeNodes('LoadBoolFieldFromRects', self.id, rects_ser)
end


-----------------------------------
-- Globals
-----------------------------------

--[[
  size
--]]
local WorkerGlobal    = {}
WorkerGlobal.__index  = WorkerGlobal
local function is_wglobal(obj) return getmetatable(obj)==WorkerGlobal end

--[[
  size
  reductions
--]]
local ControllerGlobal    = {}
ControllerGlobal.__index  = ControllerGlobal
local function is_cglobal(obj) return getmetatable(obj)==ControllerGlobal end

local global_id_counter = 0
local worker_global_objects = {}
local controller_global_objects = {}

local function NewWorkerGlobal(g_id, size)
  g_id = assert(tonumber(g_id))
  size = assert(tonumber(size))
  local g = setmetatable({
    id    = g_id,
    size  = size,
    data  = C.malloc(size),
  }, WorkerGlobal)
  worker_global_objects[g_id] = g
end
local function GetWorkerGlobal(g_id)
  return assert(worker_global_objects[g_id],
                'expected to find global #'..g_id)
end
gaswrap.registerLuaEvent('NewWorkerGlobal', NewWorkerGlobal)

local function BroadcastNewGlobal(id, size)
  BroadcastLuaEventToComputeNodes( 'NewWorkerGlobal', id, size)
end
local function NewControllerGlobal(args)
  assert(args.size, 'expected size arg')
  local id = global_id_counter + 1
  global_id_counter = id

  BroadcastNewGlobal(id, args.size)

  local g = setmetatable({
    _id   = id,
    _size = args.size,
    _data = C.malloc(args.size),
  }, ControllerGlobal)
  controller_global_objects[id] = g

  return g
end
local function GetControllerGlobal(g_id)
  return assert(controller_global_objects[g_id],
                'expected to find global #'..g_id)
end

local function SetWorkerGlobal(g_id, val_data)
  local wg = GetWorkerGlobal(tonumber(g_id))
  binary_destring_terra(wg.data, wg.size, val_data)
end
gaswrap.registerLuaEvent('SetWorkerGlobal', SetWorkerGlobal)

function ControllerGlobal:set(val_buf)
  if val_buf ~= self._data then -- edge case
    C.memcpy(self._data, val_buf, self._size)
  end
  local value_ser = stringify_binary_data(self._size, self._data)
  BroadcastLuaEventToComputeNodes('SetWorkerGlobal', self._id, value_ser)
end

local global_reduction_semaphores = {}
local function InitGlobalReduceSemaphore(glob)
  global_reduction_semaphores[glob] = numComputeNodes()
end
local function DecrementGlobalReduceSemaphore(glob)
  local sval = global_reduction_semaphores[glob] - 1
  global_reduction_semaphores[glob] = sval
  return sval
end
local function WaitGlobalReduceSemaphore(glob)
  local sval = global_reduction_semaphores[glob]
  while sval and sval > 0 do
    C.usleep(1)
    gaswrap.pollLuaEvents(0,0)
    sval = global_reduction_semaphores[glob]
  end
end

function ControllerGlobal:get()
  -- make sure we're not waiting on any reductions
  WaitGlobalReduceSemaphore(self)
  return self._data
end

function WorkerGlobal:getptr()
  return self.data
end
function WorkerGlobal:getsize()
  return self.size
end
function WorkerGlobal:getid()
  return self.id
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
--]]


-----------------------------------
-- Reduction Operations
-----------------------------------

local ReduceOp    = {}
ReduceOp.__index  = ReduceOp
local function is_reduceop(obj) return getmetatable(obj) == ReduceOp end

local reduceop_counter = 0
local reduceop_controller_objects = {}

local struct ReduceOpArgs {
  accum   : &uint8
  val     : &uint8
}

--[[
{
  func        = terra function with above arguments,
  name        = 'string'?,
}
--]]
local function NewReduceOp(args)
  local f   = assert(args.func)
  reduceop_counter = reduceop_counter + 1
  local rop = setmetatable({
    id    = reduceop_counter,
    func  = f,
    name  = args.name,
  }, ReduceOp)
  reduceop_controller_objects[rop.id] = rop
  return rop
end
local function GetControllerReduceOp(red_id)
  return assert(reduceop_controller_objects[red_id],
                'expected to find reduction #'..red_id)
end

local function BroadcastFullyReducedValue(glob)
  glob:set(glob._data)
end
local function ReportGlobalReduction( g_id, red_id, data_str )
  -- do the reduction operation
  local g     = GetControllerGlobal(tonumber(g_id))
  local rop   = GetControllerReduceOp(tonumber(red_id))
  local data  = terralib.cast(&uint8, C.malloc(g._size))
  binary_destring_terra(data, g._size, data_str)
  rop.func(g._data, data)

  -- keep track of a semaphore
  if DecrementGlobalReduceSemaphore(g) == 0 then
    BroadcastFullyReducedValue(g)
  end

  C.free(data)
end
gaswrap.registerLuaEvent('ReportGlobalReduction', ReportGlobalReduction)

-----------------------------------
-- Field Accesses
-----------------------------------


-- enum constants
local READ_ONLY_PRIVILEGE     = 1
local READ_WRITE_PRIVILEGE    = 2
local REDUCE_PRIVILEGE        = 3
Exports.READ_ONLY_PRIVILEGE   = READ_ONLY_PRIVILEGE
Exports.READ_WRITE_PRIVILEGE  = READ_WRITE_PRIVILEGE
Exports.REDUCE_PRIVILEGE      = REDUCE_PRIVILEGE

--[[
{
  privilege   = <enum>,
  field       = ControllerField,
  centered    = bool,
}
--]]
local FAccess           = {}
FAccess.__index         = FAccess
local function NewFAccess(args)
  assert(is_cfield(args.field), 'expecting ewrap field arg')
  return setmetatable({
    privilege   = assert(args.privilege),
    field       = args.field,
    centered    = args.centered or false,
    -- TODO: Add stencil support 
  }, FAccess)
end
local function is_faccess(obj) return getmetatable(obj) == FAccess end

local WorkerFAccess     = {}
WorkerFAccess.__index   = WorkerFAccess
local function NewWorkerFAccess(rawobj)
  local field = GetWorkerField(rawobj.f_id)
  return setmetatable({
    privilege   = assert(rawobj.privilege),
    field       = field,
    centered    = rawobj.centered or false,
  }, WorkerFAccess)
end


--[[
{
  privilege   = <enum>,
  reduceop    = 'op_string'?,
  global      = ControllerGlobal,
}
--]]
local GAccess           = {}
GAccess.__index         = GAccess
local function NewGAccess(args)
  assert(args.privilege)
  if args.privilege == REDUCE_PRIVILEGE then
    assert(is_reduceop(args.reduceop), 'expected reduction operation')
  else assert(args.reduceop == nil, 'expected no reduction operation') end
  return setmetatable({
    privilege   = args.privilege,
    reduceop    = args.reduceop,
    global      = args.global,
  }, GAccess)
end
local function is_gaccess(obj) return getmetatable(obj) == GAccess end

local WorkerGAccess     = {}
WorkerGAccess.__index   = WorkerGAccess
local function NewWorkerGAccess(rawobj)
  local g = GetWorkerGlobal(rawobj.g_id)
  return setmetatable({
    privilege   = assert(rawobj.privilege),
    reduce_id   = rawobj.reduce_id,
    global      = g,
  }, WorkerGAccess)
end





-----------------------------------
-- Task Creation
-----------------------------------


local ControllerTask    = {}
ControllerTask.__index  = ControllerTask
local WorkerTask        = {}
WorkerTask.__index      = WorkerTask
local task_id_counter   = 0
local worker_task_store = {}

local struct TaskArgs {
  bounds  : BoundsStruct[3];
  fields  : &FieldInstance;
  globals : &GlobalInstance;
}

-- Event/handler for registering a new task. Returns a task id.
--[[
params : {
  func              = <terra func>,
  name              = 'string', (optional)
  rel_id            = #,
  processor         = <enum>,
  field_accesses    = list{ FAccess },
  globals           = list{ CGlobals },
  reductions        = map{ g}
}
--]]
local function RegisterNewTask(params)
  assert(on_control_node())
  assert(type(params) == 'table', 'expect named args')
  assert(terralib.isfunction(params.func),
                                "expect arg: terra function 'func'")
  assert(is_cgrid_relation(params.relation),
                                "expect arg: ewrap relation 'relation'")
  assert(params.processor,      "expect arg: enum value 'processor'")
  assert(params.field_accesses,
         "expect arg: list of FAccess 'field_accesses'")
  assert(params.global_accesses,
         "expect arg: list of GAccess 'global_accesses'")

  assert(params.processor == Pre.CPU, 'Only CPU tasks supported right now.')

  -- generate id and other arguments
  task_id_counter = task_id_counter + 1
  local task_id   = task_id_counter
  local task_name = params.name or params.func:getname()
  local fas       = newlist()
  local gas       = newlist()
  for i,fa in ipairs(params.field_accesses) do
    assert(is_faccess(fa), "entry #"..i.." is not a field access")
    fas:insert {
      privilege   = fa.privilege,
      f_id        = fa.field.id,
      centered    = fa.centered,
    }
  end
  for i,ga in ipairs(params.global_accesses) do
    assert(is_gaccess(ga), "entry #"..i.." is not a global access")
    gas:insert {
      privilege   = ga.privilege,
      reduce_id   = ga.reduceop and ga.reduceop.id,
      g_id        = ga.global._id,
    }
  end
  local bitcode   = terralib.saveobj(nil,'bitcode',{[task_name]=params.func})

  -- broadcast to workers
  BroadcastLuaEventToComputeNodes('newTask', bitcode,
    gaswrap.lson_stringify {
      id              = task_id,
      name            = task_name,
      rel_id          = params.relation.id,
      processor       = tostring(params.processor),
      field_accesses  = fas,
      global_accesses = gas,
  })

  local controller_task = setmetatable({
    id              = task_id,
    rel_id          = params.relation.id,
    field_accesses  = params.field_accesses,
    global_accesses = params.global_accesses,
  }, ControllerTask)
  return controller_task
end

local keep_hooks_live = newlist()
local function NewWorkerTask(bitcode, metadata)
  local md          = gaswrap.lson_eval(metadata)
  assert(md.processor == 'CPU')

  -- convert faccesses
  local wfaccesses = newlist()
  for i,fa in ipairs(md.field_accesses) do
    wfaccesses[i] = NewWorkerFAccess(fa)
  end
  -- convert global ids
  local wgaccesses = newlist()
  for i,raw_ga in ipairs(md.global_accesses) do
    wgaccesses[i] = NewWorkerGAccess(raw_ga)
  end

  -- convert bitcode
  local blob        = terralib.linkllvmstring(bitcode)
  local task_func   = blob:extern(md.name, {TaskArgs} -> {})
  keep_hooks_live:insert(task_func)
  keep_hooks_live:insert(blob)
  local tmp_ptr_mem = newlist()
  local task_wrapper = terra(args : &opaque)
    var task_args   = [&TaskArgs](args)

    -- if we have global reductions, allocate temporary stack space
    escape for i,ga in ipairs(wgaccesses) do
      if ga.privilege == REDUCE_PRIVILEGE then
        local ptrsym = symbol(&uint8)
        tmp_ptr_mem:insert(ptrsym)
        emit quote
          var [ptrsym] = [&uint8](C.malloc([ ga.global:getsize() ]))
          task_args.globals[i-1].ptr = ptrsym
        end
      end
    end end

    -- run task
    task_func(@task_args)

    -- if we have global reductions, then handle shipping those back
    escape for i,ga in ipairs(wgaccesses) do
      if ga.privilege == REDUCE_PRIVILEGE then
        local size    = ga.global:getsize()
        local g_id    = ga.global:getid()
        local red_id  = ga.reduce_id
        emit quote
          var str_buf = [rawstring](C.malloc(2*size+1))
          binary_stringify_terra(str_buf, size, task_args.globals[i-1].ptr)
          -- TODO: Move this to the work-plane
          gaswrap.sendLuaEvent(CONTROL_NODE, 'ReportGlobalReduction',
                                             [double](g_id),
                                             [double](red_id), str_buf)
          C.free(str_buf)
        end
      end
    end end
    -- cleanup allocations
    escape for _,ptr in ipairs(tmp_ptr_mem) do
      emit quote C.free(ptr) end
    end end

    -- clean up argument allocation
    if task_args.fields ~= nil  then C.free(task_args.fields) end
    if task_args.globals ~= nil then C.free(task_args.globals) end
    C.free(task_args)
  end
  task_wrapper:compile()

  local wtask = setmetatable({
    id              = md.id,
    name            = md.name,
    func            = task_wrapper,
    relation        = GetWorkerRelation(md.rel_id),
    processor       = Pre.CPU,
    field_accesses  = wfaccesses,
    global_accesses = wgaccesses,
  }, WorkerTask)
  return wtask
end
local function ReceiveNewTask(bitcode, metadata)
  local task = NewWorkerTask(bitcode, metadata)
  worker_task_store[task.id] = task
end


-----------------------------------
-- Task Execution
-----------------------------------

local terra allocTaskArgs( n_fields : uint32, n_globals : uint32 ) : &TaskArgs
  var args    = [&TaskArgs]( C.malloc(sizeof(TaskArgs)) )
  var fsize   = n_fields*sizeof(FieldInstance)
  var gsize   = n_globals*sizeof(GlobalInstance)
  if n_fields == 0 then args.fields = nil
                   else args.fields = [&FieldInstance](C.malloc(fsize)) end
  if n_globals == 0 then args.globals = nil
                    else args.globals = [&GlobalInstance](C.malloc(gsize)) end
  return args
end


local task_all_ready_for_ghosts = {}
function ControllerTask:exec()
  -- make sure no pending field network setup
  if not task_all_ready_for_ghosts[self] then
    for _, fa in ipairs(self.field_accesses) do
      if fa.privilege == READ_WRITE_PRIVILEGE then
        print('*** READ WRITE! Checking if ghost channels are set up for ' ..
              fa.field.id)
        WaitOnFieldGhostReady(fa.field.id)
        print('*** Proceeding. Ghost channels seem to be set up for ' ..
              fa.field.id)
      end
    end
    task_all_ready_for_ghosts[self] = true
  end
  -- make sure no pending global reductions
  for _, ga in ipairs(self.global_accesses) do
    WaitGlobalReduceSemaphore(ga.global)
    -- and setup semaphore barriers for anything we're going to reduce to
    if ga.privilege == REDUCE_PRIVILEGE then
      InitGlobalReduceSemaphore(ga.global)
    end
  end
  print('Launching! '..self.id)
  BroadcastLuaEventToComputeNodes('launchTask', self.id)
end

local function LaunchTask(task_id)
  local task  = worker_task_store[tonumber(task_id)]
  assert(task, 'Task #' .. task_id ..  ' is not registered.')
  print('['..THIS_NODE..'] launch task '..task_id..' started')

  -- unpack things
  local n_dims      = #task.relation.dims

  -- allocate signal arrays
  local n_fields    = #task.field_accesses
  local n_globals   = #task.global_accesses
  local sigs_f_in   = newlist()
  local sigs_f_out  = nil

  -- Assemble Arguments
  -- because a task may be scheduled more than once before being
  -- executed, we must either re-allocate the arguments each time
  -- or ensure that the arguments are the same on all executions
  local args        = allocTaskArgs(n_fields, n_globals)
  local range       = task.relation:GetPartitionBounds():getranges()
  for d = 1,n_dims do
    args.bounds[d-1].lo   = range[d][1]
    args.bounds[d-1].hi   = range[d][2]
  end
  if #range == 2 then args.bounds[2].lo = 0
                      args.bounds[2].hi = 0 end

  for i, fa in ipairs(task.field_accesses) do
    local local_ptr   = fa.field:GetInstance():DataPtrGhostAdjusted()
    local strides     = fa.field:GetStrides()
    local elem_size   = fa.field.instance:ElemSize()
    local af          = args.fields[i-1]
    local offset      = 0
    for d=1,n_dims do
      offset          = offset + strides[d] * range[d][1]
      af.strides[d-1] = strides[d]
    end
    af.ptr            = local_ptr - offset * fa.field:GetTypeStride()
    af.elem_size      = elem_size
  end

  for i,ga in ipairs(task.global_accesses) do
    args.globals[i-1].ptr   = ga.global:getptr()
  end

  -- Do Scheduling
  gaswrap.acquireScheduler()
  print('['..THIS_NODE..'] start sched')
  -- collect and merge all the input signals
  for i, fa in ipairs(task.field_accesses) do
    if fa.privilege == READ_ONLY_PRIVILEGE then
      sigs_f_in[i]  = fa.field:ForkWriteSignal()
    elseif fa.privilege == READ_WRITE_PRIVILEGE then
      sigs_f_in[i]  = fa.field:GetReadWriteSignal()
    elseif fa.privilege == REDUCE_PRIVILEGE then
      error('TODO: REDUCE PRIV UNIMPLEMENTED')
    else assert(false, 'unrecognized field privilege') end
  end
  local a_in = nil
  --print('['..THIS_NODE..'] GOT SIGS')
  if n_fields == 0 then
    a_in = gaswrap.newSignalSource():trigger()
  elseif n_fields == 1 then
    a_in = sigs_f_in[1]
  else
    a_in = gaswrap.luaMergeSignals(sigs_f_in)
    --a_in = gaswrap.mergeSignals(n_fields, sigs_f_in)
  end
  -- Schedule the task action itself
  local worker_id = 0 -- TODO: use partition in the future...?
  local a_out     = a_in:exec(worker_id, task.func:getpointer(), args)
  -- fork and record the output signal
  if n_fields == 0 then
    a_out:sink()
  elseif n_fields == 1 then
    sigs_f_out      = newlist{ a_out }
    --sigs_f_out[0]   = a_out
  else
    sigs_f_out      = gaswrap.luaForkSignals(a_out, n_fields)
    --a_out:fork(n_fields, sigs_f_out)
  end
  for i, fa in ipairs(task.field_accesses) do
    if fa.privilege == READ_ONLY_PRIVILEGE then
      fa.field:MergeReadSignal(sigs_f_out[i])
    elseif fa.privilege == READ_WRITE_PRIVILEGE then
      fa.field:SetReadWriteSignal(sigs_f_out[i])
      scheduleSendAndRecv(fa.field)
    elseif fa.privilege == REDUCE_PRIVILEGE then
      error('TODO: REDUCE PRIV UNIMPLEMENTED')
    else assert(false, 'unrecognized field privilege') end
  end
  -- Release
  gaswrap.releaseScheduler()
  print('['..THIS_NODE..'] launch task '..task_id..' exited')
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


local current_control_barrier = 0
local function OpenBarrierOnController()
  -- initialize barrier semaphore
  current_control_barrier = numComputeNodes()

  BroadcastLuaEventToComputeNodes('openBarrier')

  -- wait on the semaphore count
  while current_control_barrier > 0 do
    C.usleep(1)
    gaswrap.pollLuaEvents(0,0)
  end
end

local function CloseBarrierOnController()
  current_control_barrier = current_control_barrier - 1
end
gaswrap.registerLuaEvent('closeBarrier',        CloseBarrierOnController)

local terra barrier_action( args : &opaque )
  gaswrap.sendLuaEvent(CONTROL_NODE, 'closeBarrier')
end

local function OpenBarrierOnWorker()
  assert(not on_control_node(), 'worker only')

  -- collect necessary data
  local fields    = GetAllWorkerFields()
  local n_fields  = #fields
  local sigs_in   = terralib.new(gaswrap.Signal[n_fields])
  local sigs_out  = terralib.new(gaswrap.Signal[n_fields])

  -- schedule worker-level barrier
  gaswrap.acquireScheduler()

  -- merge all field signals
  for i,f in ipairs(fields) do
    sigs_in[i-1] = f:GetAllSignals()
  end
  local s_in = nil
  if      n_fields == 0 then  s_in = gaswrap.newSignalSource():trigger()
  elseif  n_fields == 1 then  s_in = sigs_in[0]
                        else gaswrap.mergeSignals(n_fields, sigs_in) end
  -- barrier action
  local s_out = s_in:exec(0,barrier_action:getpointer(),nil)
  -- fork all field signals
  if      n_fields == 0 then  s_out:sink()
  elseif  n_fields == 1 then  sigs_out[0] = s_out
                        else  s_out:fork(n_fields, sigs_out) end
  for i,f in ipairs(fields) do
    f:SetAllSignals(sigs_out[i-1])
  end

  gaswrap.releaseScheduler()
end
gaswrap.registerLuaEvent('openBarrier',         OpenBarrierOnWorker)





-------------------------------------------------------------------------------
-- Register event handlers 
-------------------------------------------------------------------------------

-- relational data
gaswrap.registerLuaEvent('newRelation',         CreateNewRelation)
gaswrap.registerLuaEvent('globalGridPartition', CreateGlobalGridPartition)
gaswrap.registerLuaEvent('recordNewField',      RecordNewField)
gaswrap.registerLuaEvent('prepareField',        PrepareField)
gaswrap.registerLuaEvent('loadFieldConstant',   LoadFieldConstant)

-- task code and data
gaswrap.registerLuaEvent('newTask',             ReceiveNewTask)
gaswrap.registerLuaEvent('launchTask',          LaunchTask)


-------------------------------------------------------------------------------
-- Exports for testing
-------------------------------------------------------------------------------

Exports._TESTING_BroadcastNewRelation         = BroadcastNewRelation
Exports._TESTING_BroadcastGlobalGridPartition = BroadcastGlobalGridPartition
Exports._TESTING_BroadcastNewField            = BroadcastNewField
Exports._TESTING_RemotePrepareField           = RemotePrepareField
Exports._TESTING_RemoteLoadFieldConstant      = RemoteLoadFieldConstant
Exports._TESTING_BroadcastLuaEventToComputeNodes =
  BroadcastLuaEventToComputeNodes

-------------------------------------------------------------------------------
-- Exports
-------------------------------------------------------------------------------

Exports.N_NODES                       = N_NODES
Exports.THIS_NODE                     = THIS_NODE

Exports.NewGridRelation               = NewGridRelation
Exports.NewField                      = NewField
Exports.NewGlobal                     = NewControllerGlobal

Exports.TaskArgs                      = TaskArgs
Exports.FieldInstance                 = FieldInstance
Exports.GhostInstance                 = GhostInstance
Exports.NewReduceOp                   = NewReduceOp
Exports.NewFAccess                    = NewFAccess
Exports.NewGAccess                    = NewGAccess
Exports.RegisterNewTask               = RegisterNewTask

Exports.SyncBarrier                   = OpenBarrierOnController

