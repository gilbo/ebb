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

local USE_CONSERVATIVE_GHOSTS   = true
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

local struct BoundsStruct { lo : uint64, hi : uint64 }

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
  local ghost_width = {}
  local n_elems     = 1
  local dim_size    = {}
  local strides     = {}
  for d = 1,3 do
    strides[d]      = n_elems   -- column major?
    ghost_width[d]  = params.ghost_width[d] or 0
    dim_size[d]     = widths[d] + 1 -- adjust for hi/lo inclusive convention
    n_elems         = n_elems * (dim_size[d] + 2*ghost_width[d])
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
  for d = 1,3 do
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


-- use an hid_base as there could be multiple channels for the same field
-- programmatically generate the rest, keep last 3 digits for ghost num, and
-- NODE_DIGITS digits for node ids (source and destination).
local function gen_hs_id(hid_base, src, dst, ghost_num)
  return ghost_num +
         dst      * math.pow(10, GHOST_DIGITS) +
         src      * math.pow(10, GHOST_DIGITS + NODE_DIGITS) +
         hid_base * math.pow(10, GHOST_DIGITS + NODE_DIGITS * 2)
end

local worker_ghost_num_channels = {}
local worker_ghost_channels_freeze = {}

local function ghost_id(offset, ndims)
  local g = 0
  for d = 1, ndims do
    g = g * 3 + (offset[d] + 1)
  end
  return g
end


--[[
-- what do we actually need
-- handshake_id       - channel initialization handshake id
-- f_id               - field id
--]]

--[[
{
  hid_base  = params.hid_base,             -- generate channel id
  ghost_id  = ghost_id(owner_off, ndims),  -- generate channel id
  fid       = self.id,      -- whhich field, useful to signal to controller
  nb_id     = nb_lin_id,    -- which node to set up channel with
  buffer    = nil,          -- data
  buf_size  = 0,            -- size of buffer
  elem_size = self:GetTypeSize(),  -- element size (for copying elements)
  bounds    = bounds_rect,  -- inclusive bounds
}
--]]
local GhostMetadata = {}
GhostMetadata.__index = GhostMetadata

--[[
  params {
    hid_base,    -- id info
    offset,      -- which ghost (offset wrt center)
    inner,       -- true if inner ghost, false if outer ghost
    ghost_width  -- ghost width
  }
--]]
function WorkerField:CreateGridGhostMetadata(params)
  assert(not worker_ghost_channels_freeze[params.hid_base],
         'Already started setting up ghost channels. All calls to '..
         'create ghost channels must precede channel set up calls.')
  local ndims     = #self.relation:Dims()
  local off       = params.offset
  local owner_off = params.inner and {  off[1],  off[2],  off[3] }
                                  or { -off[1], -off[2], off[3] and -off[3] }
  local partition = self.relation:GetPartition()
  local blocking  = partition.blocking
  local bid       = partition.block_id
  local allocate  = true
  local periodic  = self.relation.periodic
  -- compute neigbor's id
  local nb_bid    = {}
  local nb_lin_id = partition.map
  for d = 1, ndims do
    nb_bid[d] = bid[d] + off[d]
    if not periodic[d] then
      allocate = allocate and (nb_bid[d] >= 1 and nb_bid[d] <= blocking[d])
    end
    nb_bid[d] = (nb_bid[d] - 1) % blocking[d] + 1
    nb_lin_id = nb_lin_id[nb_bid[d]]
  end
  -- compute bounds for this ghost
  local pbounds      = partition.bounds:getranges()
  local ghost_width  = params.ghost_width
  local bounds       = {}
  for d = 1, ndims do
    bounds[d] = {}
    if off[d] == -1 then
      bounds[d][1] = params.inner and pbounds[d][1] or
                     pbounds[d][1] - ghost_width[d]
      bounds[d][2] = params.inner and pbounds[d][1] + ghost_width[d] - 1 or
                     pbounds[d][1] - 1
    elseif off[d] == 0 then
      bounds[d][1] = pbounds[d][1]
      bounds[d][2] = pbounds[d][2]
    elseif off[d] == 1 then
      bounds[d][1] = params.inner and pbounds[d][2] - ghost_width[d] + 1 or
                     pbounds[d][2] + 1
      bounds[d][2] = params.inner and pbounds[d][2] or
                     pbounds[d][2] + ghost_width[d]
    end
  end
  if ndims == 2 then bounds[3] = {1,1} end
  local bounds_rect = Util.NewRect3d(unpack(bounds))
  local elem_size   = self:GetTypeSize()
  -- ghost metadata
  local g        = {
    hid_base  = params.hid_base,             -- generate channel id
    ghost_id  = ghost_id(owner_off, ndims),  -- generate channel id
    fid       = self.id,      -- whhich field, useful to signal to controller
    nb_id     = nb_lin_id,    -- which node to set up channel with
    buffer    = nil,          -- data
    buf_size  = 0,            -- size of buffer
    elem_size = self:GetTypeSize(),  -- element size (for copying elements)
    bounds    = bounds_rect,  -- inclusive bounds
  }
  -- allocate buffer
  if allocate then
    local ghost_width = bounds_rect:getwidths()
    local buf_size = elem_size
    for d = 1, ndims do
      buf_size = buf_size * (ghost_width[d] + 1)
    end
    g.buf_size = buf_size
    g.buffer = terralib.cast(&uint8, C.malloc(buf_size))
    assert(g.buffer, 'Failed to allocate ghost buffer')
    if not worker_ghost_num_channels[params.hid_base] then
      worker_ghost_num_channels[params.hid_base] = 1
    else
      worker_ghost_num_channels[params.hid_base] =
        worker_ghost_num_channels[params.hid_base] + 1
    end
  end
  return setmetatable(g, GhostMetadata)
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
  print(THIS_NODE, 'DECREMENT f#'..f_id..': '..(semaphore_val-1))
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
  print(THIS_NODE, 'DONE waiting on f#'..f_id)
end





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

function GhostMetadata:CreateOutgoingGridGhostChannel(g)
  worker_ghost_channels_freeze[self.hid_base] = true
  -- set buffer
  g.ptr = self.buffer
  if not self.buffer then return end
  -- set bounds
  local range = self.bounds:getranges()
  for d = 1,#range do
    g.bounds[d-1].lo = range[d][1]
    g.bounds[d-1].hi = range[d][2]
  end
  local function DoneCallback(chan)
    print('*** Node ' .. gas.mynode() .. ' decrementing channel counter for ' ..
          self.fid .. '. Old counter value is ' ..
          worker_ghost_num_channels[self.hid_base])
    worker_ghost_num_channels[self.hid_base] =
      worker_ghost_num_channels[self.hid_base] - 1
    if worker_ghost_num_channels[self.hid_base] == 0 then
      -- inform controller that ghost channel setup is done
      print('*** Node ' .. gas.mynode() .. ' done with channels for ' ..
            self.fid)
      gaswrap.sendLuaEvent(CONTROL_NODE, 'MarkFieldGhostsReady', self.fid)
    end
    -- set channel
    g.src = chan
  end
  local hid = gen_hs_id(self.hid_base, gas.mynode(), self.nb_id, self.ghost_id)
  print('*** src', hid, self.buf_size)
  gaswrap.CreateAsyncBufSrcChannel(self.nb_id, hid,
                                   self.buffer, self.buf_size, DoneCallback)
end

function GhostMetadata:CreateIncomingGridGhostChannel(g)
  worker_ghost_channels_freeze[self.hid_base] = true
  -- set buffer
  g.ptr       = self.buffer
  g.elem_size = self.elem_size
  if not self.buffer then return end
  -- set bounds
  local range = self.bounds:getranges()
  local width = self.bounds:getwidths()
  local n_elems = 1
  for d = 1,3 do
    g.bounds[d-1].lo = range[d][1]
    g.bounds[d-1].hi = range[d][2]
    g.strides[d-1]   = n_elems
    n_elems          = n_elems * (width[d] + 1)
  end
  local function DoneCallback(chan)
    print('*** Node ' .. gas.mynode() .. ' decrementing channel counter for ' ..
          self.fid .. '. Old counter value is ' ..
          worker_ghost_num_channels[self.hid_base])
    worker_ghost_num_channels[self.hid_base] =
      worker_ghost_num_channels[self.hid_base] - 1
    if worker_ghost_num_channels[self.hid_base] == 0 then
      -- inform controller that ghost channel setup is done
      print('*** Node ' .. gas.mynode() .. ' done with channels for ' ..
            self.fid)
      gaswrap.sendLuaEvent(CONTROL_NODE, 'MarkFieldGhostsReady',
                           self.fid, self.hid_base)
    end
    -- set channel
    g.dst = chan
  end
  local hid = gen_hs_id(self.hid_base, self.nb_id, gas.mynode(), self.ghost_id)
  print('*** dst', hid, self.buf_size)
  gaswrap.CreateAsyncBufDstChannel(self.nb_id, hid,
                                   self.buffer, self.buf_size, DoneCallback)
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
end
function WorkerField:SetUpGhostChannels(hid_base, ghost_width)
  local ndims = #self.relation:Dims()
  local ghost_listing = ndims == 2 and ghost_listing_2d or ghost_listing_3d
  local partition     = self.relation:GetPartition()
  local inner_ghosts  = newlist()
  local outer_ghosts  = newlist()
  for i,off in ipairs(ghost_listing) do
    inner_ghosts[i] = self:CreateGridGhostMetadata(
      {
         hid_base     = hid_base,
         offset       = off,
         inner        = true,
         ghost_width  = ghost_width
      })
    outer_ghosts[i] = self:CreateGridGhostMetadata(
      {
         hid_base     = hid_base,
         offset       = off,
         inner        = false,
         ghost_width  = ghost_width
      })
  end
  self.inner_ghosts_size  = #ghost_listing
  self.outer_ghosts_size  = #ghost_listing
  self.inner_ghosts = terralib.new(GhostInstance[#ghost_listing])
  self.outer_ghosts = terralib.new(GhostInstance[#ghost_listing])
  for i = 1,#ghost_listing do
    inner_ghosts[i]:CreateOutgoingGridGhostChannel(self.inner_ghosts[i-1])
    outer_ghosts[i]:CreateIncomingGridGhostChannel(self.outer_ghosts[i-1])
  end
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
                        id           = params.id,
                        type_size    = params.type_size,
                        name         = params.name,
                        relation     = relation,
                        instance     = nil,
                        last_read    = nil,
                        last_write   = nil,
                        inner_ghosts = nil,
                        outer_ghosts = nil,
                        inner_ghosts_size = 0,
                        outer_ghosts_size = 0,
                      },
                      WorkerField)
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

function WorkerField:GetReadWriteSignal()
  local signals = terralib.new(gaswrap.Signal[2])
  signals[0] = self.last_read
  signals[1] = self.last_write
  return gaswrap.mergeSignals(2, signals)
end
function WorkerField:ForkReadSignal()
  local signals = terralib.new(gaswrap.Signal[2])
  self.last_read:fork(2, signals)
  self.last_read = signals[1]
  return signals[0]
end
function WorkerField:ForkWriteSignal()
  local signals = terralib.new(gaswrap.Signal[2])
  self.last_write:fork(2, signals)
  self.last_write = signals[1]
  return signals[0]
end
function WorkerField:MergeReadSignal(signal)
  local signals = terralib.new(gaswrap.Signal[2])
  signals[0] = self.last_read
  signals[1] = signal
  self.last_read = gaswrap.mergeSignals(2, signals)
end
function WorkerField:MergeWriteSignal(signal)
  local signals = terralib.new(gaswrap.Signal[2])
  signals[0] = self.last_write
  signals[1] = signal
  self.last_write = gaswrap.mergeSignals(2, signals)
end
function WorkerField:SetReadWriteSignal(signal)
  local signals = terralib.new(gaswrap.Signal[2])
  signal:fork(2, signals)
  self.last_read  = signals[0]
  self.last_write = signals[1]
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
function WorkerField:SetAllSignals()
  return self:SetReadWriteSignal()
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
    blocking = blocking,  -- how relation is partitioned
    block_id = block_id,  -- multi-dimensional partition block id
    bounds   = bounds,    -- bounds for this partition
    map      = map,       -- from block id to node id
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
  for _,f in pairs(field_metadata) do fs:insert(f) end
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
local CopyGridGhosts = {
  Send = 0,
  Recv = 0,
}
for k,_ in pairs(CopyGridGhosts) do
  CopyGridGhosts[k] = terra (copy_args : &opaque)
    C.printf('*** CopyGridGhosts executing for %s.\n', k)
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
                if k == 'Send' then
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
    C.printf('*** CopyGridGhosts done for %s.\n', k)
  end
end

-- start_sig = task's write
-- merge output signal with read
-- caller must acquire scheduler
local terra CopyAndSendGridGhosts(start_sig : gaswrap.Signal,
                                  copy_args : &GhostCopyArgs)
  var num_ghosts = copy_args.num_ghosts
  var ghosts     = copy_args.ghosts
  var copy_sig_forked : &gaswrap.Signal =
    [&gaswrap.Signal](C.malloc(sizeof(gaswrap.Signal) * num_ghosts))
  var send_sig : &gaswrap.Signal =
    [&gaswrap.Signal](C.malloc(sizeof(gaswrap.Signal) * num_ghosts))
  -- just schedule on one worker thread for now
  var worker_id : uint32 = 0
  -- copy into buffers once start_sig triggers
  var copy_sig = start_sig:exec(
    worker_id, CopyGridGhosts.Send, [&opaque](copy_args))
  copy_sig:fork(num_ghosts, copy_sig_forked)
  -- send buffers after copying
  for i = 0, num_ghosts do
    if ghosts[i].ptr == nil then
      copy_sig_forked[i]:sink()
      send_sig[i] = gaswrap.newSignalSource():trigger()
    else
      send_sig[i] = ghosts[i].src:send(copy_sig_forked[i])
      --copy_sig_forked[i]:sink()
      --send_sig[i] = gaswrap.newSignalSource():trigger()
    end
  end
  -- merge send done signals
  C.printf('Merge begin in send\n')
  var done_sig = gaswrap.mergeSignals(num_ghosts, send_sig)
  C.printf('Merge end in send\n')
  C.free(copy_sig_forked)
  C.free(send_sig)
  return done_sig
end
-- start_sig = previous read, before the task preceeding this exchange
-- merge output signal with task's write
-- caller must acquire scheduler
local terra RecvAndCopyGridGhosts(start_sig : gaswrap.Signal,
                                  copy_args : &GhostCopyArgs)
  var num_ghosts = copy_args.num_ghosts
  var ghosts     = copy_args.ghosts
  var recv_and_start_sig : &gaswrap.Signal =
    [&gaswrap.Signal](C.malloc(sizeof(gaswrap.Signal) * (num_ghosts + 1)))
  -- receive all the buffers
  for i = 0, num_ghosts do
    if ghosts[i].ptr == nil then
      recv_and_start_sig[i] = gaswrap.newSignalSource():trigger()
    else
      recv_and_start_sig[i] = ghosts[i].dst:recv()
      --recv_and_start_sig[i] = gaswrap.newSignalSource():trigger()
    end
  end
  recv_and_start_sig[num_ghosts] = start_sig
  -- just schedule on one worker thread for now
  var worker_id : uint32 = 0
  -- copy out from buffers once all data is received and start_sig triggers
  C.printf('Merge begin in receive\n')
  var copy_sig = gaswrap.mergeSignals(num_ghosts+1, recv_and_start_sig)
  C.printf('Merge end in receive\n')
  -- copy into buffers
  var done_sig = copy_sig:exec(
    worker_id, CopyGridGhosts.Recv, [&opaque](copy_args))
  -- merge send done signals
  C.free(recv_and_start_sig)
  return done_sig
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
  local relation = GetWorkerRelation(tonumber(rel_id))
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
  field:SetUpGhostChannels(tonumber(hid_base), ghost_width)
end


local function stringify_binary_data( n_bytes, valptr )
  assert(n_bytes % 4 == 0, 'unexpected size, bin_stringify NEEDS FIXING')
  local n_words   = n_bytes / 4
  local w_ptr     = terralib.cast(&uint32, valptr)
  local num_list  = newlist()
  for k=1,n_words do num_list[k] = w_ptr[k-1] end
  return gaswrap.lson_stringify(num_list)
end
local function destringify_binary_data( valstr, n_bytes, valptr )
  assert(n_bytes % 4 == 0, 'unexpected size, bin_destringify NEEDS FIXING')
  local num_list  = gaswrap.lson_eval(valstr)
  local n_words   = n_bytes / 4
  local w_ptr     = terralib.cast(&uint32, valptr)
  for k=1,n_words do w_ptr[k-1] = num_list[k] end
end

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
  destringify_binary_data(value_ser, field:GetTypeSize(), args.valptr)
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
function ControllerGridRelation:partition_within_nodes()
end

function ControllerField:LoadConst( c_val )
  local typsize   = terralib.sizeof(self.type)
  local temp_mem  = terralib.cast(&self.type, C.malloc(typsize))
  temp_mem[0]     = c_val
  RemoteLoadFieldConstant(self.id, typsize, temp_mem)
  C.free(temp_mem)
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

--[[
{
  privilege   = <enum>,
  field       = WorkerField,
}
--]]
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
  bounds : BoundsStruct[3];
  fields : &FieldInstance;
}

-- Event/handler for registering a new task. Returns a task id.
--[[
params : {
  func              = <terra func>,
  name              = 'string', (optional)
  rel_id            = #,
  processor         = <enum>,
  field_accesses    = list{ FAccess },
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
  assert(params.field_accesses, "expect arg: list of FAccess 'field_ids'")

  assert(params.processor == Pre.CPU, 'Only CPU tasks supported right now.')

  -- generate id and other arguments
  task_id_counter = task_id_counter + 1
  local task_id   = task_id_counter
  local task_name = params.name or params.func:getname()
  local fas       = newlist()
  for i,fa in ipairs(params.field_accesses) do
    assert(is_faccess(fa), "entry #"..i.." is not a field access")
    fas:insert {
      privilege  = fa.privilege,
      f_id       = fa.field.id,
      centered   = fa.centered,
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
  })

  local controller_task = setmetatable({
    id             = task_id,
    rel_id         = params.rel_id,
    field_accesses = params.field_accesses,
  }, ControllerTask)
  return controller_task
end

local keep_hooks_live = newlist()
local function NewWorkerTask(bitcode, metadata)
  local md          = gaswrap.lson_eval(metadata)
  assert(md.processor == 'CPU')

  -- convert bitcode
  local blob        = terralib.linkllvmstring(bitcode)
  local task_func   = blob:extern(md.name, {TaskArgs} -> {})
  keep_hooks_live:insert(task_func)
  keep_hooks_live:insert(blob)
  local task_wrapper = terra(args : &opaque)
    var task_args   = [&TaskArgs](args)
    --C.printf('[%d] start exec action in wrapper\n', THIS_NODE)
    task_func(@task_args)
    -- free memory allocated for args
    --C.printf('[%d] done exec action in wrapper\n', THIS_NODE)
    if task_args.fields ~= nil then C.free(task_args.fields) end
    C.free(task_args)
  end
  task_wrapper:compile()

  -- convert faccesses
  local wfaccesses = newlist()
  for i,fa in ipairs(md.field_accesses) do
    wfaccesses[i] = NewWorkerFAccess(fa)
  end

  local wtask = setmetatable({
    id              = md.id,
    name            = md.name,
    func            = task_wrapper,
    rel_id          = md.rel_id,
    processor       = Pre.CPU,
    field_accesses  = wfaccesses,
  }, WorkerTask)
  return wtask
end
local function ReceiveNewTask(bitcode, metadata)
  local task = NewWorkerTask(bitcode, metadata)
  worker_task_store[task.id] = task
end

function WorkerTask:GetRelation()
  return GetWorkerRelation(self.rel_id)
end


-----------------------------------
-- Task Execution
-----------------------------------

local terra allocTaskArgs( n_fields : uint32 ) : &TaskArgs
  var args    = [&TaskArgs]( C.malloc(sizeof(TaskArgs)) )
  if n_fields == 0 then args.fields = nil
  else
    args.fields = [&FieldInstance]( C.malloc(n_fields*sizeof(FieldInstance)) )
  end
  return args
end


local task_all_ready_for_ghosts = {}
function ControllerTask:exec()
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
  BroadcastLuaEventToComputeNodes('launchTask', self.id)
end

local function LaunchTask(task_id)
  local task  = worker_task_store[tonumber(task_id)]
  assert(task, 'Task #' .. task_id ..  ' is not registered.')
  --print(THIS_NODE..' launch task started')

  -- unpack things
  local relation    = task:GetRelation()
  local n_dims      = #relation.dims

  -- allocate signal arrays
  local n_fields    = #task.field_accesses
  local sigs_f_in   = terralib.new(gaswrap.Signal[n_fields])
  local recv_done   = terralib.new(gaswrap.Signal[n_fields])
  local send_done   = terralib.new(gaswrap.Signal[n_fields])
  local sigs_f_out  = terralib.new(gaswrap.Signal[n_fields])

  -- Assemble Arguments
  -- because a task may be scheduled more than once before being
  -- executed, we must either re-allocate the arguments each time
  -- or ensure that the arguments are the same on all executions
  local args        = allocTaskArgs(n_fields)
  local range       = relation:GetPartitionBounds():getranges()
  for d = 1,3 do
    args.bounds[d-1].lo   = range[d][1]
    args.bounds[d-1].hi   = range[d][2]
  end

  local send_args = terralib.new((&GhostCopyArgs)[n_fields])
  local recv_args = terralib.new((&GhostCopyArgs)[n_fields])

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
    if fa.privilege == READ_WRITE_PRIVILEGE then
      local sa = terralib.cast(&GhostCopyArgs,
                               C.malloc(terralib.sizeof(GhostCopyArgs)))
      sa.num_ghosts      = fa.field.inner_ghosts_size
      sa.ghosts          = fa.field.inner_ghosts
      sa.field.ptr       = local_ptr
      sa.field.elem_size = elem_size
      local ra = terralib.cast(&GhostCopyArgs,
                               C.malloc(terralib.sizeof(GhostCopyArgs)))
      ra.num_ghosts      = fa.field.outer_ghosts_size
      ra.ghosts          = fa.field.outer_ghosts
      ra.field.ptr       = local_ptr
      ra.field.elem_size = elem_size
      for d = 1,3 do
        sa.field_bounds[d-1].lo = range[d][1] 
        sa.field_bounds[d-1].hi = range[d][2] 
        sa.field.strides[d-1]   = strides[d]
        ra.field_bounds[d-1].lo = range[d][1] 
        ra.field_bounds[d-1].hi = range[d][2] 
        ra.field.strides[d-1]   = strides[d]
      end
      send_args[i-1] = sa
      recv_args[i-1] = ra
    else
      send_args[i-1] = nil
      recv_args[i-1] = nil
    end
  end

  -- Do Scheduling
  gaswrap.acquireScheduler()
  -- collect and merge all the input signals
  for i, fa in ipairs(task.field_accesses) do
    if fa.privilege == READ_ONLY_PRIVILEGE then
      sigs_f_in[i-1]  = fa.field:ForkWriteSignal()
    elseif fa.privilege == READ_WRITE_PRIVILEGE then
      local recv_in   = fa.field:ForkReadSignal()
      recv_done[i-1]  = RecvAndCopyGridGhosts(recv_in, recv_args[i-1])
      sigs_f_in[i-1]  = fa.field:GetReadWriteSignal()
    elseif fa.privilege == REDUCE_PRIVILEGE then
      error('TODO: REDUCE PRIV UNIMPLEMENTED')
    else assert('unrecognized field privilege') end
  end
  local a_in = nil
  if n_fields == 0 then
    a_in = gaswrap.newSignalSource():trigger()
  elseif n_fields == 1 then
    a_in = sigs_f_in[0]
  else
    print('Merge begin in a_in')
    a_in = gaswrap.mergeSignals(n_fields, sigs_f_in)
    print('Merge end in a_in')
  end
  -- Schedule the task action itself
  local worker_id = 0 -- TODO: use partition in the future...?
  local a_out     = a_in:exec(worker_id, task.func:getpointer(), args)
  -- fork and record the output signal
  if n_fields == 0 then
    a_out:sink()
  elseif n_fields == 1 then
    sigs_f_out[0]   = a_out
  else
    a_out:fork(n_fields, sigs_f_out)
  end
  for i, fa in ipairs(task.field_accesses) do
    if fa.privilege == READ_ONLY_PRIVILEGE then
      fa.field:MergeReadSignal(sigs_f_out[i-1])
    elseif fa.privilege == READ_WRITE_PRIVILEGE then
      local f_out = terralib.new(gaswrap.Signal[2])
      sigs_f_out[i-1]:fork(2, f_out)
      local send_done = CopyAndSendGridGhosts(f_out[0], send_args[i-1])
      fa.field:SetReadSignal(f_out[0])
      local writes = terralib.new(gaswrap.Signal[2])
      writes[0] = recv_done[i-1]
      writes[1] = f_out[1]
    print('Merge begin in setwrite on node ' .. gas.mynode())
      fa.field:SetWriteSignal(gaswrap.mergeSignals(2, writes))
    print('Merge end in setwrite on node ' .. gas.mynode())
    elseif fa.privilege == REDUCE_PRIVILEGE then
      error('TODO: REDUCE PRIV UNIMPLEMENTED')
    else assert('unrecognized field privilege') end
  end
  -- Release
  gaswrap.releaseScheduler()
  --print(THIS_NODE..' launch task exited')
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
                        else  gaswrap.mergeSignals(n_fields, sigs_in) end
  -- barrier action
  local s_out = s_in:exec(0,barrier_action:getpointer(),nil)
  -- fork all field signals
  if      n_fields == 0 then  s_out = gaswrap.newSignalSource():trigger()
  elseif  n_fields == 1 then  s_out = sigs_in[0]
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

Exports.TaskArgs                      = TaskArgs
Exports.FieldInstance                 = FieldInstance
Exports.NewFAccess                    = NewFAccess
Exports.RegisterNewTask               = RegisterNewTask

Exports.SyncBarrier                   = OpenBarrierOnController

