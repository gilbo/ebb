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


-- This file only depends on the C headers and legion,
-- so it's fairly separate from the rest of the compiler,
-- so it won't cause inadvertent dependency loops.

local LW = {}
package.loaded["ebb.src.legionwrap"] = LW

local C     = require "ebb.src.c"
local DLD   = require "ebb.lib.dld"
local Util  = require 'ebb.src.util'

-- have this module expose the full C-API.  Then, we'll augment it below.
local APIblob = terralib.includecstring([[
#include "legion_c.h"
#include "reductions_cpu.h"
#include "ebb_mapper.h"
]])
for k,v in pairs(APIblob) do LW[k] = v end

local newlist = terralib.newlist


-------------------------------------------------------------------------------
--[[  Legion environment                                                   ]]--
-------------------------------------------------------------------------------

local LE = rawget(_G, '_legion_env')
local run_config = rawget(_G, '_run_config')
local use_partitioning = run_config.use_partitioning

local struct LegionEnv {
  runtime : LW.legion_runtime_t,
  ctx     : LW.legion_context_t
}
LE.legion_env = C.safemalloc( LegionEnv )
local legion_env = LE.legion_env[0]

local terra null()
  return [&opaque](0)
end


-------------------------------------------------------------------------------
--[[            Data Accessors - Fields, Futures Declarations              ]]--
-------------------------------------------------------------------------------

-- Logical Regions
local LogicalRegion     = {}
LogicalRegion.__index   = LogicalRegion
LW.LogicalRegion        = LogicalRegion

-- Inline Physical Regions (if we need these, we must create inline launchers,
-- request regions, and free launchers and regions)
local InlinePhysicalRegion    = {}
InlinePhysicalRegion.__index  = InlinePhysicalRegion
LW.InlinePhysicalRegion       = InlinePhysicalRegion

-- Futures
local FutureBlob        = {}
FutureBlob.__index      = FutureBlob

-- 1D, 2D and 3D field accessors to access fields within Legion tasks
LW.FieldAccessor = {}
for d = 1, 3 do
  LW.FieldAccessor[d] = struct {
    ptr     : &uint8,
    strides : LW.legion_byte_offset_t[d],
    handle  : LW.legion_accessor_generic_t
  }
  LW.FieldAccessor[d].__typename =
    function() return 'FieldAccessor_'..tostring(d) end
end

-- legion methods for different dimensions (templates)
LW.LegionPoint = {}
LW.LegionRect = {}
LW.LegionRectFromDom = {}
LW.LegionDomFromRect = {}
LW.LegionRawPtrFromAcc = {}
for d = 1,3 do
  local str = '_' .. tostring(d) .. 'd'
  LW.LegionPoint[d] = LW['legion_point' .. str .. '_t']
  LW.LegionRect[d] = LW['legion_rect' .. str .. '_t']
  LW.LegionRectFromDom[d] = LW['legion_domain_get_rect' .. str]
  LW.LegionDomFromRect[d] = LW['legion_domain_from_rect' .. str]
  LW.LegionRawPtrFromAcc[d] = LW['legion_accessor_generic_raw_rect_ptr' .. str]
end

LW.DomainRect = {}
for d = 1,3 do
  local args, lo_args, hi_args = {}, {}, {}
  for k=1,d do
    lo_args[k]  = symbol(int, 'lo_'..k)
    hi_args[k]  = symbol(int, 'hi_'..k)
    args[k]     = lo_args[k]
    args[d+k]   = hi_args[k]
  end
  LW.DomainRect[d] = terra( [args] )
    var lo = [LW.LegionPoint[d]]({ arrayof(LW.coord_t, [lo_args]) })
    var hi = [LW.LegionPoint[d]]({ arrayof(LW.coord_t, [hi_args]) })
    var bounds = [LW.LegionRect[d]]({ lo, hi })
    return [LW.LegionDomFromRect[d]](bounds)
  end
end

local LegionCreatePoint = {}
LegionCreatePoint[1] = terra(pt1 : int)
  return [LW.LegionPoint[1]]{arrayof(LW.coord_t, pt1)}
end
LegionCreatePoint[2] = terra(pt1 : int, pt2 : int)
  return [LW.LegionPoint[2]]{arrayof(LW.coord_t, pt1, pt2)}
end
LegionCreatePoint[3] = terra(pt1 : int, pt2 : int, pt3 : int)
  return [LW.LegionPoint[3]]{arrayof(LW.coord_t, pt1, pt2, pt3)}
end

local LegionDomainPoint = macro(function(pt1,pt2,pt3)
  if      pt3 then return `LW.legion_domain_point_from_point_3d(
                                LegionCreatePoint[3](pt1,pt2,pt3))
  elseif  pt2 then return `LW.legion_domain_point_from_point_2d(
                                LegionCreatePoint[2](pt1,pt2))
              else return `LW.legion_domain_point_from_point_1d(
                                LegionCreatePoint[1](pt1))          end
end,function(pt1,pt2,pt3)
  if      pt3 then return LW.legion_domain_point_from_point_3d(
                                LegionCreatePoint[3](pt1,pt2,pt3))
  elseif  pt2 then return LW.legion_domain_point_from_point_2d(
                                LegionCreatePoint[2](pt1,pt2))
              else return LW.legion_domain_point_from_point_1d(
                                LegionCreatePoint[1](pt1))          end
end)


-------------------------------------------------------------------------------
--[[  Region Requirements                                                  ]]--
-------------------------------------------------------------------------------

local RegionReqs         = {}
RegionReqs.__index = RegionReqs

--[[
{
  log_region_handle,
  num_group,
  num_total,
  offset,
  privilege,
  coherence,
  reduce_func_id,
  centered,
  ids
}
--]]
function LW.NewRegionReqs(params)
  local reduce_func_id = (params.privilege == LW.REDUCE)
            and LW.GetFieldReductionId(params.reduce_op, params.reduce_typ)
            or nil
  local relation = params.relation
  local offset   = params.offset
  local ids = terralib.newlist()
  for i = 1, params.num_total do
    ids[i] = offset + i - 1
  end
  local reg_req = setmetatable({
    log_reg_handle  = relation._logical_region_wrapper:get_handle(),
    num_group       = params.num_group,   -- can be zero
    num_total       = params.num_total,
    offset          = params.offset,      -- can be zero 
    privilege       = params.privilege,
    coherence       = params.coherence,
    reduce_func_id  = reduce_func_id,       
    centered        = params.centered,
    ids             = ids,
  }, RegionReqs)
  return reg_req
end

function RegionReqs:isreduction() return self.privilege == LW.REDUCE end

function RegionReqs:GetIds()
  return self.ids
end

function RegionReqs:GroupNum()
  return self.num_group
end

function RegionReqs:TotalNum()
  return self.num_total
end


-------------------------------------------------------------------------------
--[[  Legion Tasks                                                         ]]--
-------------------------------------------------------------------------------

struct LW.TaskArgs {
  task        : LW.legion_task_t,
  regions     : &LW.legion_physical_region_t,
  num_regions : uint32,
  lg_ctx      : LW.legion_context_t,
  lg_runtime  : LW.legion_runtime_t
}

-- Get a new unique task id.
local new_task_id   = 100  -- start registering tasks at 101 ...
function LW.get_new_task_id()
  new_task_id = new_task_id + 1
  return new_task_id
end

-- Get task if for a given task. Register the task and create a new id if the
-- task is not memoized. Tasks are memoized by the function version name, which
-- corresponds to a { function, type, proc, blocksize, relation, subset }. Note
-- that we have different tasks for gpus and cpus. This is intentional, it
-- allows Ebb planner to make decisions about which tasks to launch on what
-- partitions.
--[[
function signature:
  task_func : task function
  on_gpu    : is this task function on gpu?
  ufv_name  : task function version name
  returns a task id
--]]
local RegisterAndGetTaskId = Util.memoize_from(3,
function(ufv_name, on_gpu, task_func)
  -- new task id
  local TID = LW.get_new_task_id()
  -- register given task function with the task id
  local terra register()
    escape if run_config.use_llvm then
      local ir = terralib.saveobj(nil, "llvmir", { entry = task_func })
      emit quote 
        LW.legion_runtime_register_task_variant_llvmir(
          legion_env.runtime, TID, [(on_gpu and LW.TOC_PROC) or LW.LOC_PROC],
          true,
          LW.legion_task_config_options_t { leaf = true, inner = false, idempotent = false },
          ufv_name, [&opaque](0), 0,
          ir, "entry")
      end  -- emit quote
    else
      emit quote
        LW.legion_runtime_register_task_variant_fnptr(
          legion_env.runtime, TID, [(on_gpu and LW.TOC_PROC) or LW.LOC_PROC],
          LW.legion_task_config_options_t { leaf = true, inner = false, idempotent = false },
          ufv_name, [&opaque](0), 0,
          task_func)
      end  -- emit quote
    end end  -- escape if else
  end  -- terra function
  register()
  return TID
end)


local TaskLauncher         = {}
TaskLauncher.__index = TaskLauncher

--[[
{
  taskname
  taskfunc
  gpu               -- boolean
  use_index_launch  -- boolean
  n_copies          -- how many copies to launch if doing index launch
}
--]]

function LW.NewTaskLauncher(params)
  if not terralib.isfunction(params.task_func) then
    error('must provide Terra function as "task_func" argument', 2)
  end
  if type(params.ufv_name) ~= 'string' then
    error('must provide a task name as "ufv_name" argument', 2)
  end

  -- register task and get task id
  local TID = RegisterAndGetTaskId(params.ufv_name, params.gpu,
                                   params.task_func)
  -- common task arguments across all partitions
  local argstruct   = C.safemalloc( LW.legion_task_argument_t )
  argstruct.args    = null()
  argstruct.arglen  = 0

  -- create launcher
  local launcher_create =
    (params.use_index_launch and LW.legion_index_launcher_create) or
    LW.legion_task_launcher_create

  -- task launcher arguments
  local launcher_args = newlist({TID})
  if params.use_index_launch then
    -- partition specific arguments
    local argmap = LW.legion_argument_map_create()
    local domain = LW.DomainRect[1](0,params.n_copies-1)
    launcher_args:insertall({domain, argstruct[0], argmap,
                             LW.legion_predicate_true(), false})
  else
    launcher_args:insertall({argstruct[0], LW.legion_predicate_true()})
  end
  local tag = 0
  if params.node_id then
    tag = params.node_id + 1
  end
  launcher_args:insertall({0, tag})
  local launcher = launcher_create(unpack(launcher_args))

  return setmetatable({
    _TID          = TID,
    _launcher     = launcher,
    _index_launch = params.use_index_launch,
    _on_gpu       = params.gpu,
    _reduce_func_id = nil,
  }, TaskLauncher)
end

function TaskLauncher:Destroy()
  self._taskfunc = nil
  self._taskfuncptr = nil
  local DestroyVariant = self._index_launch and
    LW.legion_index_launcher_destroy or
    LW.legion_task_launcher_destroy
  DestroyVariant(self._launcher)
  self._launcher = nil
end

function TaskLauncher:IsOnGPU()
  return self._on_gpu
end

function TaskLauncher:AddRegionReqs(reqs, regs)
  if use_partitioning then
    -- using partitioning
    assert(regs ~= nil)
    assert(reqs:TotalNum() == #regs,
      "Recorded " .. tostring(reqs:TotalNum()) .. " requirements vs got " ..
      tostring(#regs) .. " requirements.")
  end
  local req_ids = reqs:GetIds()
  for i, req_id in ipairs(req_ids)  do
    local partn_or_reg = regs and regs[i]
    local handle       = partn_or_reg and partn_or_reg:get_handle() or
                         reqs.log_reg_handle
    -- Assemble the call
    local args = newlist { self._launcher, handle }
    local str = 'legion'
    local is_logical_partition = (getmetatable(partn_or_reg) ==
                                  LW.LogicalPartition)
    if self._index_launch   then    args:insert(0)
                                    str = str .. '_index'
                            else    str = str .. '_task' end
    str = str .. '_launcher_add_region_requirement_logical'
    if is_logical_partition then     str = str .. '_partition'
                            else     str = str .. '_region' end
    if reqs:isreduction()   then     args:insert(reqs.reduce_func_id)
                                     str = str .. '_reduction'
                            else     args:insert(reqs.privilege)
                            end
    args:insertall { reqs.coherence, reqs.log_reg_handle, 0, false }
    -- Do the call
    local id = LW[str](unpack(args))
    -- assert that region requirements are actually added as recorded
    -- when building legion signature
    assert(id == req_id)
  end
end

function TaskLauncher:AddField(reg_reqs, fid)
  local req_ids = reg_reqs:GetIds()
  for _, rid in ipairs(req_ids) do
    local AddFieldVariant =
      (self._index_launch and LW.legion_index_launcher_add_field) or
      LW.legion_task_launcher_add_field
    AddFieldVariant(self._launcher, rid, fid, true)
  end
end

function TaskLauncher:AddFuture(future)
  local AddFutureVariant =
    (self._index_launch and LW.legion_index_launcher_add_future) or
    LW.legion_task_launcher_add_future
  AddFutureVariant(self._launcher, future)
end

function TaskLauncher:AddFutureReduction(op, ebb_typ)
  self._reduce_func_id = LW.GetGlobalReductionId(op, ebb_typ)
end

-- If there's a future it will be returned
function TaskLauncher:Execute(runtime, ctx, redop_id)
  local exec_str = 'legion'..
                   (self._index_launch and '_index' or '_task')..
                   '_launcher_execute'
  local exec_args = newlist({runtime, ctx, self._launcher})
  -- possibly add reduction to future
  if self._reduce_func_id and self._index_launch then
    exec_str = exec_str .. '_reduction'
    exec_args:insert(self._reduce_func_id)
  end
  return LW[exec_str](unpack(exec_args))
end


-------------------------------------------------------------------------------
--[[  Future methods                                                       ]]--
-------------------------------------------------------------------------------


function FutureBlob:LegionFuture()
  return self.legion_future
end

function FutureBlob:AssignTo(global, offset)
  global:SetData(self)
  global:SetOffset(offset)
  self.ref_count = self.ref_count + 1
end

function FutureBlob:Release()
  self.ref_count = self.ref_count - 1
  if self.ref_count == 0 then
    LW.legion_future_destroy(self.legion_future)
    self.legion_future = nil
  end
end

function LW.AssignFutureBlobFromFuture(global, legion_future)
  local f = { legion_future = legion_future, ref_count = 0 }
  setmetatable(f, FutureBlob)
  f:AssignTo(global, 0)
  return f
end

function LW.AssignFutureBlobFromValue(global, cdata)
  local ttype = global:Type():terratype()
  local tsize = terralib.sizeof(ttype)
  local data_blob = terralib.cast(&ttype, C.malloc(tsize))
  data_blob[0] = cdata
  local legion_future = LW.legion_future_from_buffer(legion_env.runtime,
                                                     data_blob,
                                                     terralib.sizeof(ttype))
  local f = { legion_future = legion_future, ref_count = 0 }
  setmetatable(f, FutureBlob)
  local old_future = global:Data()
  if old_future then
    old_future:Release()
  end
  f:AssignTo(global, 0)
  return f
end

function LW.AssignFutureBlobFromCollection(globals)
  local blob_size = 0
  for _, g in pairs(globals) do
    blob_size = blob_size + terralib.sizeof(g:Type():terratype())
  end
  local data_blob = terralib.cast(&uint8, C.malloc(blob_size))
  local f = { ref_count = 0 }
  setmetatable(f, FutureBlob)
  local offset = 0
  for _, g in pairs(globals) do
    local old_future = g:Data()
    local tsize = terralib.sizeof(g:Type():terratype())
    C.memcpy(data_blob[offset], old_future:GetResult(g), tsize)
    local old_future = g:Data()
    if old_future then
      old_future:Release()
    end
    f:AssignTo(g, offset)
    offset = offset + tsize
  end
  f.legion_future = LW.legion_future_from_buffer(legion_env.runtime, data_blob,
                                                 terralib.sizeof(ttype))
  C.free(data_blob)
  return f
end

function FutureBlob:GetResult(global)
  local ttype = global:Type():terratype()
  local leg_result = LW.legion_future_get_result(self.legion_future)
  local offset = global:Offset()
  local d = terralib.cast(&uint8, leg_result.value)
  local data = terralib.new(ttype, terralib.cast(&ttype, d + offset)[0])
  LW.legion_task_result_destroy(leg_result)
  return data
end


-------------------------------------------------------------------------------
--[[  Logical Region Methods                                               ]]--
-------------------------------------------------------------------------------


-- NOTE: Call from top level task only.
function LogicalRegion:AllocateRows(num)
  if self.type ~= 'unstructured' then
    error("Cannot allocate rows for grid relation ", self.relation:Name(), 3)
  else
    if self.live_rows + num > self.max_rows then
      error("Cannot allocate more rows for relation ", self.relation:Name())
    end
  end
  LW.legion_index_allocator_alloc(self.isa, num)
  self.live_rows = self.live_rows + num
end


local allocate_field_fid_counter = 0
-- NOTE: Assuming here that the compile time limit is never hit.
-- NOTE: Call from top level task only.
function LogicalRegion:_HIDDEN_ReserveFields()
  if self._field_reserve then
    error('Function ReserveFields() should only be called once')
  end

  -- indexed by number of bytes
  -- (this particular set of sizes chosen by inspecting applications)
  self._field_reserve = {
    [1] = {},
    [2] = {},
    [3] = {},
    [4] = {},
    [8] = {},
    [12] = {},
    [16] = {},
    [24] = {},
    [32] = {},
    [36] = {},
    [48] = {},
    [64] = {},
    [72] = {}
  }
  self._field_reserve_count = {
    [1] = 20,
    [2] = 20,
    [3] = 20,
    [4] = 40,
    [8] = 40,
    [12] = 40,
    [16] = 40,
    [24] = 40,
    [32] = 40,
    [36] = 40,
    [48] = 40,
    [64] = 20,
    [72] = 20
  }
  for nbytes, list in pairs(self._field_reserve) do
    for i=1,self._field_reserve_count[nbytes] do

      local fid = LW.legion_field_allocator_allocate_field(
                    self.fsa,
                    nbytes,
                    allocate_field_fid_counter
                  )
      assert(fid == allocate_field_fid_counter)
      allocate_field_fid_counter = allocate_field_fid_counter + 1

      table.insert(list, fid)
    end
  end
end

function LogicalRegion:AllocateField(typ)
  local typsize = typ
  if type(typ) ~= 'number' then typsize = terralib.sizeof(typ) end
  local field_reserve = self._field_reserve[typsize]
  if not field_reserve then
    error('No field reserve for type size ' .. typsize)
  end
  local fid = table.remove(field_reserve)
  if not fid then
    error('Ran out of fields of size '..typsize..' to allocate;\n'..
          'This error is the result of a hack to investigate performance '..
          'issues in field allocation.  Fixing it 100% will probably '..
          'require Mike making changes in the Legion runtime.\n')
  end
  LW.ebb_mapper_add_field(legion_env.runtime, legion_env.ctx,
                          self.handle, fid)
  return fid
end

function LogicalRegion:FreeField(fid)
  LW.legion_field_allocator_free_field(self.fsa, fid)
end

function LogicalRegion:AttachNameToField(fid, name)
  LW.legion_field_id_attach_name(legion_env.runtime, self.fs, fid,
                                 name, true)
end

function LogicalRegion:InitConstField(fid, cdata_ptr, cdata_size)
  LW.legion_runtime_fill_field(legion_env.runtime, legion_env.ctx,
    self.handle,
    self.handle, --parent
    fid,
    -- void * and size_t respectively for next two args
    cdata_ptr, cdata_size,
    LW.legion_predicate_true()
  )
end

local CreateGridIndexSpace = {}

-- Internal method: Ask Legion to create 1 dimensional index space
CreateGridIndexSpace[1] = terra(x : int)
  var pt_lo = LW.legion_point_1d_t { arrayof(LW.coord_t, 0) }
  var pt_hi = LW.legion_point_1d_t { arrayof(LW.coord_t, x-1) }
  var rect  = LW.legion_rect_1d_t { pt_lo, pt_hi }
  var dom   = LW.legion_domain_from_rect_1d(rect)
  return LW.legion_index_space_create_domain(
            legion_env.runtime, legion_env.ctx, dom)
end

-- Internal method: Ask Legion to create 2 dimensional index space
CreateGridIndexSpace[2] = terra(x : int, y : int)
  var pt_lo = LW.legion_point_2d_t { arrayof(LW.coord_t, 0, 0) }
  var pt_hi = LW.legion_point_2d_t { arrayof(LW.coord_t, x-1, y-1) }
  var rect  = LW.legion_rect_2d_t { pt_lo, pt_hi }
  var dom   = LW.legion_domain_from_rect_2d(rect)
  return LW.legion_index_space_create_domain(
            legion_env.runtime, legion_env.ctx, dom)
end

-- Internal method: Ask Legion to create 3 dimensional index space
CreateGridIndexSpace[3] = terra(x : int, y : int, z : int)
  var pt_lo = LW.legion_point_3d_t { arrayof(LW.coord_t, 0, 0, 0) }
  var pt_hi = LW.legion_point_3d_t { arrayof(LW.coord_t, x-1, y-1, z-1) }
  var rect  = LW.legion_rect_3d_t { pt_lo, pt_hi }
  var dom   = LW.legion_domain_from_rect_3d(rect)
  return LW.legion_index_space_create_domain(
            legion_env.runtime, legion_env.ctx, dom)
end

-- Allocate an unstructured logical region
-- NOTE: Call from top level task only.
function LW.NewLogicalRegion(params)
  -- Max rows for index space = n_rows right now ==> no inserts
  -- Should eventually figure out an upper bound on number of rows and use that
  -- when creating index space.
  local l = setmetatable({
    type      = 'unstructured',
    relation  = params.relation,
    field_ids = 0,
    n_rows    = params.n_rows,
    live_rows = 0,
    max_rows  = params.n_rows
  }, LogicalRegion)

  -- legion throws an error with 0 max rows
  if l.max_rows == 0 then l.max_rows = 1 end
  l.is  = LW.legion_index_space_create(legion_env.runtime,
                                       legion_env.ctx, l.max_rows)
  l.isa = LW.legion_index_allocator_create(legion_env.runtime,
                                           legion_env.ctx, l.is)
  -- field space
  l.fs  = LW.legion_field_space_create(legion_env.runtime,
                                       legion_env.ctx)
  l.fsa = LW.legion_field_allocator_create(legion_env.runtime,
                                           legion_env.ctx, l.fs)
  l:_HIDDEN_ReserveFields()

  -- logical region
  l.handle = LW.legion_logical_region_create(legion_env.runtime,
                                             legion_env.ctx, l.is, l.fs)
  LW.legion_logical_region_attach_name(legion_env.runtime, l.handle,
                                       l.relation:Name(), false)

  -- actually allocate rows
  l:AllocateRows(l.n_rows)
  return l
end

-- Allocate a structured logical region
-- NOTE: Call from top level task only.
function LW.NewGridLogicalRegion(params)
  local l = setmetatable({
    type        = 'grid',
    relation    = params.relation,
    field_ids   = 0,
    offsets     = (#params.dims == 2) and {0,0} or {0,0,0},
    dims        = params.dims,
  }, LogicalRegion)

  -- index space
  local dims = l.dims
  l.is = CreateGridIndexSpace[#dims](unpack(dims))
  -- field space
  l.fs = LW.legion_field_space_create(legion_env.runtime,
                                      legion_env.ctx)
  l.fsa = LW.legion_field_allocator_create(legion_env.runtime,
                                           legion_env.ctx, l.fs)
  l:_HIDDEN_ReserveFields()

  -- logical region
  l.handle = LW.legion_logical_region_create(legion_env.runtime,
                                             legion_env.ctx,
                                             l.is, l.fs)
  LW.legion_logical_region_attach_name(legion_env.runtime, l.handle,
                                       l.relation:Name(), false)
  setmetatable(l, LogicalRegion)
  return l
end

function LogicalRegion:get_handle()
  return self.handle
end


-------------------------------------------------------------------------------
--[[  Physical Region Methods                                              ]]--
-------------------------------------------------------------------------------

-- Inline physical region for entire relation, not just a part of it
function LW.NewInlinePhysicalRegion(params)
  if not params.relation then
    error('Expects relation argument', 2)
  elseif not params.fields then
    error('Expects fields list argument', 2)
  elseif not params.privilege then
    error('Expects privilege argument', 2)
  end

  -- legion data (inline launchers, physical regions, accessors) to free later
  local ils  = {}
  local prs  = {}
  local accs = {}

  -- structured/ grid and dimensions
  local relation = params.relation
  local is_grid  = relation:isGrid()
  local dims     = relation:Dims()

  -- data pointer, stride, offset
  local ptrs  = {}
  local strides = {}
  local offsets = {}

  for i, field in ipairs(params.fields) do
    -- create inline launcher
    ils[i]  = LW.legion_inline_launcher_create_logical_region(
      params.relation._logical_region_wrapper:get_handle(),  -- legion_logical_region_t handle
      params.privilege,         -- legion_privilege_mode_t
      LW.EXCLUSIVE,             -- legion_coherence_property_t
      params.relation._logical_region_wrapper:get_handle(),  -- legion_logical_region_t parent
      0,                        -- legion_mapping_tag_id_t region_tag /* = 0 */
      false,                    -- bool verified /* = false*/
      0,                        -- legion_mapper_id_t id /* = 0 */
      0                         -- legion_mapping_tag_id_t launcher_tag /* = 0 */
    )
    -- add field to launcher
    LW.legion_inline_launcher_add_field(ils[i], params.fields[i]._fid, true)
    -- execute launcher to get physical region
    prs[i]  = LW.legion_inline_launcher_execute(legion_env.runtime,
                                                legion_env.ctx, ils[i])
  end

  -- get field data pointers, strides
  local index_space = LW.legion_physical_region_get_logical_region(prs[1]).index_space
  local domain = LW.legion_index_space_get_domain(legion_env.runtime,
                                                  legion_env.ctx,
                                                  index_space)
  if is_grid then
    local ndims = #dims
    local rect = LW.LegionRectFromDom[ndims](domain)
    for d = 1, ndims do
      assert(dims[d] == (rect.hi.x[d-1] - rect.lo.x[d-1] + 1))
    end
    local subrect = terralib.new((LW.LegionRect[ndims])[1])
    local stride = terralib.new(LW.legion_byte_offset_t[ndims])
    for i, field in ipairs(params.fields) do
      accs[i] = LW.legion_physical_region_get_field_accessor_generic(prs[i], field._fid)
      ptrs[i] = terralib.cast(&uint8,
                              LW.LegionRawPtrFromAcc[ndims](accs[i], rect, subrect, stride))
      local s = {}
      strides[i] = s
      for d = 1, ndims do
        s[d] = tonumber(stride[d-1].offset)
      end
      offsets[i] = 0
      LW.legion_accessor_generic_destroy(accs[i])
    end
  else
    local base = terralib.new((&opaque)[1])
    local stride = terralib.new(uint64[1])
    for i, field in ipairs(params.fields) do
      accs[i] = LW.legion_physical_region_get_field_accessor_generic(prs[i], field._fid)
      base[0] = nil
      stride[0] = 0
      LW.legion_accessor_generic_get_soa_parameters(accs[i], base, stride)
      ptrs[i]    = terralib.cast(&uint8, base[0])
      strides[i] = { tonumber(stride[0]) }
      offsets[i] = 0
      LW.legion_accessor_generic_destroy(accs[i])
    end
  end

  local iprs = setmetatable({
    inline_launchers  = ils,
    physical_regions  = prs,
    is_grid           = is_grid,
    dims              = dims,
    data_ptrs         = ptrs,
    strides           = strides,
    offsets           = offsets,
    fields            = params.fields,
  }, InlinePhysicalRegion)

  return iprs
end

function InlinePhysicalRegion:GetDataPointers()
  return self.data_ptrs
end

function InlinePhysicalRegion:GetDimensions()
  return self.dims
end

function InlinePhysicalRegion:GetStrides()
  return self.strides
end

function InlinePhysicalRegion:GetOffsets()
  return self.offsets
end

function InlinePhysicalRegion:GetLuaDLDs()
  local dld_list = {}
  for i,f in ipairs(self.fields) do
    local typ       = f:Type()
    local typdim    = { typ.N or typ.Nrow or 1, typ.Ncol or 1 }
    local dims      = { self.dims[1] or 1, self.dims[2] or 1,
                                           self.dims[3] or 1 }
    local typstride = sizeof(typ:terrabasetype())
    local strides   = {}
    local elemsize  = typstride*typdim[1]*typdim[2]
    for k,s_bytes in ipairs(self.strides[i]) do
      assert(s_bytes % elemsize == 0, 'unexpected type stride')
      strides[k] = s_bytes / elemsize
    end
    if not strides[2] then strides[2] = strides[1] end
    if not strides[3] then strides[3] = strides[2] end
    assert(self.offsets[i] == 0, 'expecting 0 offsets from legion')
    dld_list[i] = DLD.NewDLD {
      base_type       = typ:basetype():DLDEnum(),
      location        = DLD.CPU,
      type_stride     = sizeof(typ:terratype()),
      type_dims       = typdim,

      address         = self.data_ptrs[i],
      dim_size        = dims,
      dim_stride      = strides,
    }
  end
  return dld_list
end

function InlinePhysicalRegion:GetTerraDLDs()
  local dld_list  = self:GetLuaDLDs()
  local dld_array = C.safemalloc(DLD.C_DLD, #dld_list)
  for i,dld in ipairs(dld_list) do dld_array[i-1] = dld:toTerra() end
  return dld_array
end

function InlinePhysicalRegion:Destroy()
  for i = 1, #self.inline_launchers do
      local il = self.inline_launchers[i]
      local pr = self.physical_regions[i]

      LW.legion_runtime_unmap_region(legion_env.runtime, legion_env.ctx, pr)
      LW.legion_physical_region_destroy(pr)
      LW.legion_inline_launcher_destroy(il)
  end
end


-------------------------------------------------------------------------------
--[[  Partitioning logical regions                                         ]]--
-------------------------------------------------------------------------------

local LogicalSubRegion    = {}
LogicalSubRegion.__index  = LogicalSubRegion
LW.LogicalSubRegion       = LogicalSubRegion

local function NewLogicalSubRegion(rect, color_pt, ipart, lpart)
  assert(Util.isrect2d(rect) or Util.isrect3d(rect))
  local is3d    = Util.isrect3d(rect)

  local lis = LW.legion_index_partition_get_index_subspace_domain_point(
    legion_env.runtime,
    legion_env.ctx,
    ipart,
    color_pt
  )
  local lreg = LW.legion_logical_partition_get_logical_subregion(
    legion_env.runtime,
    legion_env.ctx,
    lpart,
    lis
  )

  local offsets = { rect:mins() }
  local dims    = { rect:maxes() }
  for k=1,(is3d and 3 or 2) do dims[k] = dims[k] - offsets[k] + 1 end
  return setmetatable({
    -- always a grid
    type        = 'grid',
    offsets     = offsets,
    dims        = dims,
    rect        = rect,
    is          = lis,
    handle      = lreg,
  }, LogicalSubRegion)
end

function LogicalSubRegion:get_handle()
  return self.handle
end

function LogicalSubRegion:get_rect()
  return self.rect
end

local LogicalPartition   = {}
LogicalPartition.__index = LogicalPartition
LW.LogicalPartition      = LogicalPartition

local function CreateGridRegionPartition(lreg_obj, subrects, part_kind)
  assert(lreg_obj.type == 'grid')
  assert(terralib.israwlist(subrects) and #subrects>0,'expect rectangle list')
  local is3d = #lreg_obj.dims == 3

  -- unpack
  --local dims      = lreg_obj.dims       -- dimensions of grid
  --local offsets   = lreg_obj.offsets    -- offset in global grid coordinates
  local lreg      = lreg_obj.handle     -- parent logical region/subregion
  local lis       = lreg_obj.is         -- index-space

  -- indexing scheme
  local n_rect        = #subrects
  local color_space   = LW.DomainRect[1](0,n_rect-1)
  local coloring      = LW.legion_domain_point_coloring_create()

  -- loop to fill out the partition coloring
  for i,rect in ipairs(subrects) do
    assert(is3d and Util.isrect3d(rect) or Util.isrect2d(rect),
           "dimensions of subrect and grid don't match")
    LW.legion_domain_point_coloring_color_domain(
      coloring,
      LegionDomainPoint(i-1),
      LW.DomainRect[is3d and 3 or 2]( rect:mins_maxes() )
    )
  end

  local idx_part = LW.legion_index_partition_create_domain_point_coloring(
    legion_env.runtime,
    legion_env.ctx,
    lis, -- parent idx space
    color_space,
    coloring,
    part_kind,
    -1 -- AUTO-GENERATE
  )
  -- can free the coloring now
  LW.legion_domain_point_coloring_destroy(coloring)

  local l_part = LW.legion_logical_partition_create(
    legion_env.runtime,
    legion_env.ctx,
    lreg, -- parent logial region
    idx_part
  )

  -- iterate over and cache all of the subregion objects
  local subregions = newlist()
  for i,rect in ipairs(subrects) do
    subregions:insert( NewLogicalSubRegion(
      rect, LegionDomainPoint(i-1), idx_part, l_part
    ))
  end

  local mode = (part_kind == LW.DISJOINT_KIND) and 'disjoint' or 'overlap'
  local new_obj = setmetatable({
    _mode         = mode,
    _lreg         = lreg_obj,
    _lpart        = l_part,
    _ipart        = idx_part,
    _subregions   = subregions,
    _subrects     = subrects,    -- useful during debugging
  }, LogicalPartition)
  return new_obj
end

function LogicalPartition:destroy()
  LW.legion_logical_partition_destroy(
    legion_env.runtime,
    legion_env.ctx,
    self._lpart
  )
  LW.legion_index_partition_destroy(
    legion_env.runtime,
    legion_env.ctx,
    self._ipart
  )
end

function LogicalPartition:get_handle()
  return self._lpart
end

function LogicalPartition:attach_name(name)
  self.legion_logical_partition_attach_name(
    legion_env.runtime,
    self.lpart,
    name..'_logical_partition',
    false
  )
  self.legion_index_partition_attach_name(
    legion_env.runtime,
    self.ipart,
    name..'_idx_partition',
    false
  )
end

function LogicalPartition:subregions()
  return self._subregions
end

---------------------------------------------

function LogicalRegion:CreateDisjointPartition(subrects)
  return CreateGridRegionPartition(self, subrects, LW.DISJOINT_KIND)
end

function LogicalRegion:CreateOverlappingPartition(subrects)
  return CreateGridRegionPartition(self, subrects, LW.ALIASED_KIND)
end

function LogicalSubRegion:CreateDisjointPartition(subrects)
  return CreateGridRegionPartition(self, subrects, LW.DISJOINT_KIND)
end

function LogicalSubRegion:CreateOverlappingPartition(subrects)
  return CreateGridRegionPartition(self, subrects, LW.ALIASED_KIND)
end


-------------------------------------------------------------------------------
--[[  Methods for copying fields and scanning through fields/ regions      ]]--
-------------------------------------------------------------------------------

function LW.CopyField (params)
  if not params.region  then error('Needs region argument', 2) end
  if not params.src_fid then error('Needs src_fid argument', 2) end
  if not params.dst_fid then error('Needs dst_fid argument', 2) end

  local src_region  = params.region
  local dst_region  = params.region
  local src_fid     = params.src_fid
  local dst_fid     = params.dst_fid

  local cplauncher =
    LW.legion_copy_launcher_create(LW.legion_predicate_true(), 0, 0)
  
  -- SETUP ARGUMENTS
  local src_idx =
  LW.legion_copy_launcher_add_src_region_requirement_logical_region(
    cplauncher,
    src_region,
    LW.READ_ONLY,
    LW.EXCLUSIVE,
    src_region,
    0,
    false
  )
  local dst_idx =
  LW.legion_copy_launcher_add_dst_region_requirement_logical_region(
    cplauncher,
    dst_region,
    LW.WRITE_ONLY,
    LW.EXCLUSIVE,
    dst_region,
    0,
    false
  )
  LW.legion_copy_launcher_add_src_field(
    cplauncher,
    src_idx,
    src_fid,
    true
  )
  LW.legion_copy_launcher_add_dst_field(
    cplauncher,
    dst_idx,
    dst_fid,
    true
  )

  -- EXEC
  LW.legion_copy_launcher_execute(legion_env.runtime,
                                  legion_env.ctx, cplauncher)

  -- CLEANUP
  LW.legion_copy_launcher_destroy(cplauncher)
end


-- The ControlScanner lets the top-level/control task
-- scan any logical region in order to load or extract data from fields
LW.ControlScanner         = {}
LW.ControlScanner.__index = LW.ControlScanner

function LW.NewControlScanner(params)
  -- create inline launcher
  local ilps  = LW.NewInlinePhysicalRegion(
    { relation   = params.relation,
      fields     = params.fields,
      privilege  = params.privilege
    } )
  local launchobj = setmetatable(
    {
      inline_physical_regions = ilps,
      relation                = params.relation
    }, LW.ControlScanner)

  return launchobj
end

function LW.ControlScanner:ScanThenClose()

  local dims = self.relation:Dims()
  local ptrs = self.inline_physical_regions:GetDataPointers()
  local strides = self.inline_physical_regions:GetStrides()

  -- define what to do when the iteration terminates
  local function close_up()
    self:close()
    return nil
  end

  -- define an iterator/generator
  if #dims == 1 then
    assert(not self.relation:isGrid())
    local nx = dims[1]
    local xi = -1
    return function()
      xi = xi+1
      if xi>= nx then return close_up() end
      local callptrs = {}
      for i = 1, #ptrs do
        callptrs[i] = ptrs[i] + 
                      xi*strides[i][1]
      end
      return {xi}, callptrs
    end
  elseif #dims == 2 then
    local nx = dims[1]
    local ny = dims[2]
    local xi = -1
    local yi = 0
    return function()
      xi = xi+1
      if xi >= nx then xi = 0; yi = yi + 1 end
      if yi >= ny then return close_up() end
      local callptrs = {}
      for i = 1, #ptrs do
        callptrs[i] = ptrs[i] + 
                      yi*strides[i][2] + xi*strides[i][1]
      end
      return {xi,yi}, callptrs
    end
  elseif #dims == 3 then
    local xi = -1
    local yi = 0
    local zi = 0
    local nx = dims[1]
    local ny = dims[2]
    local nz = dims[3]
    return function()
      xi = xi+1
      if xi >= nx then xi = 0; yi = yi + 1 end
      if yi >= ny then yi = 0; zi = zi + 1 end
      if zi >= nz then return close_up() end
      local callptrs = {}
      for i = 1, #ptrs do
        callptrs[i] = ptrs[i] + 
                      zi*strides[i][3] + yi*strides[i][2] + xi*strides[i][1]
      end
      return {xi,yi,zi}, callptrs
    end
  end
end


function LW.ControlScanner:close()
  self.inline_physical_regions:Destroy()
end

-------------------------------------------------------------------------------
--[[  Reductions                                                           ]]--
-------------------------------------------------------------------------------


local reduction_function_counter = 0
local no_more_reduction_ids = false
local function unsupported_reduce_err(is_field, op, typ)
error([[
invalid reduction operation / data type combination:
    ]]..(is_field and 'FieldReduction ' or 'GlobalReduction ')..
        op..' '..tostring(typ)..'\n'..[[
  IF YOU ARE SEEING THIS, then please tell the developers.
  This error is due to the inability to dynamically register Legion
  reduction functions.
]], 3)
end
LW.GetFieldReductionId = Util.memoize_from(1, function(op, ebb_typ)
  if no_more_reduction_ids then unsupported_reduce_err(true, op, ebb_typ) end
  reduction_function_counter = reduction_function_counter + 1
  return reduction_function_counter
end)
LW.GetGlobalReductionId = Util.memoize_from(1, function(op, ebb_typ)
  if no_more_reduction_ids then unsupported_reduce_err(false, op, ebb_typ) end
  reduction_function_counter = reduction_function_counter + 1
  return reduction_function_counter
end)


function LW.RegisterReductions()
  local T = require 'ebb.src.types'
  local reduction_ops = {
    {'+', 'plus'},
    {'*', 'times'},
    {'max', 'max'},
    {'min', 'min'},
  }
  local reduction_op_translate = {}
  for i, redop in ipairs(reduction_ops) do
    local ebb_op, lg_op = unpack(redop)
    reduction_op_translate[ebb_op] = lg_op
  end

  local basetyps = {
    {T.int, 'int32'},
    {T.float, 'float'},
    {T.double, 'double'},
  }

  -- construct list of all types with their mappings to string names
  local alltyps = terralib.newlist()
  local typ_map = {}
  for _,typ in pairs(basetyps) do
    local ebbt, lgt = unpack(typ)
    alltyps:insert(typ)
    typ_map[ebbt] = lgt
  end
  for i=2,4 do
    for _,typ in pairs(basetyps) do
      local ebbt, lgt = unpack(typ)
      local vebbt, vlgt = T.vector(ebbt,i), lgt..'_vec'..i
      alltyps:insert({vebbt, vlgt})
      typ_map[vebbt] = vlgt
    end
    for j=2,4 do
      for _,typ in pairs(basetyps) do
        local ebbt, lgt = unpack(typ)
        local mebbt, mlgt = T.matrix(ebbt,i,j), lgt..'_mat'..i..'x'..j
        alltyps:insert({mebbt, mlgt})
        typ_map[mebbt] = mlgt
      end
    end
  end

  -- now register all the corresponding functions
  for _, redop in ipairs(reduction_ops) do
    local ebb_op, lg_op = unpack(redop)
    for _, typ in ipairs(alltyps) do
      local ebbt, lgt = unpack(typ)

      local f_reg_func = LW['register_reduction_field_'..lg_op..'_'..lgt]
      if f_reg_func then
        f_reg_func( LW.GetFieldReductionId(ebb_op, ebbt) )
      end
      local g_reg_func = LW['register_reduction_global_'..lg_op..'_'..lgt]
      if g_reg_func then
        g_reg_func( LW.GetGlobalReductionId(ebb_op, ebbt) )
      end
    end
  end
  no_more_reduction_ids = true -- seal the memoization caches

  -- define accessor function here
  -- MESSY FUNCTION
  --    Pros: Hides a lot of the complexity of Legion reductions in one spot
  --    Cons: interrogates Ebb value and key types to work correctly
  --          would be nice to not have those dependencies here ???
  --          (unsure of that claim as broader policy)
  LW.GetSafeReductionFunc = Util.memoize_from(1,
  function(op, ebb_typ, key_typ)
    local valstruct = LW[tostring(ebb_typ:basetype())..'_'..
                         (ebb_typ.valsize or 1)]
    local valarray  = ebb_typ:terrabasetype()[key_typ.valsize or 1]

    local opstr     = reduction_op_translate[op]
    local typstr    = typ_map[ebb_typ]
    if not opstr or not typstr or not valstruct then
      error('INTERNAL: unrecognized reduction combo: '..
            op..' '..tostring(ebb_typ)..'\n'..
            '  PLEASE REPORT to the developers')
    end

    local is_grid   = key_typ.ndims > 1
    local str = 'safe_reduce_'..
                (is_grid and 'domain_point_' or '')..
                opstr..'_'..typstr
    local reduction_function = LW[str]

    return macro(function(accessor, key, coords)
      local legion_ptr_pt = is_grid and (`key:domainPoint())
                                     or (`LW.legion_ptr_t({key.a0}))
      return `reduction_function( accessor,
                                  [legion_ptr_pt],
                                  valstruct({coords}) )
    end)
  end)
end
LW.RegisterReductions = terralib.cast({}->{},LW.RegisterReductions)


-------------------------------------------------------------------------------
--[[  Miscellaneous methods                                                ]]--
-------------------------------------------------------------------------------

function LW.heavyweightBarrier()
  LW.legion_runtime_issue_execution_fence(legion_env.runtime, legion_env.ctx)
end


-------------------------------------------------------------------------------
--[[  Temporary hacks                                                      ]]--
-------------------------------------------------------------------------------

-- empty task function
-- to make it work with legion without blowing up memory
local terra _TEMPORARY_EmptyTaskFunction(data : & opaque, datalen : C.size_t,
                                         userdata : &opaque, userlen : C.size_t,
                                         proc_id : LW.legion_lowlevel_id_t)

  var task_args : LW.TaskArgs
  LW.legion_task_preamble(data, datalen, proc_id, &task_args.task,
                          &task_args.regions, &task_args.num_regions,
                          &task_args.lg_ctx, &task_args.lg_runtime)

  C.printf("** WARNING: Executing empty Legion task. ")
  C.printf("This is a hack for Ebb/Legion. ")
  C.printf("If you are seeing this message and do not know what this is, ")
  C.printf("please contact the developers.\n")

  -- legion postamble
  LW.legion_task_postamble(task_args.lg_runtime, task_args.lg_ctx,
                           [&opaque](0), 0)
end

--[[
args {
legion_context,
legion_runtime,
relation,
region
}
--]]
local function _TEMPORARY_LaunchEmptyLegionTask(args)
  -- create task launcher
  local task_launcher = LW.NewTaskLauncher {
    ufv_name         = 'TEMPORARY_PrepareSimulation_' .. args.relation:Name(),
    task_func        = _TEMPORARY_EmptyTaskFunction,
    gpu              = false,
    use_index_launch = false,
  }
  -- create one centered region requirement
  local reg_reqs = LW.NewRegionReqs {
    num_group       = 0,
    num_total       = 1,
    offset          = 0,
    relation        = args.relation,
    privilege       = LW.READ_WRITE,
    coherence       = LW.EXCLUSIVE,
    centered        = true
  }
  -- add region requirement
  task_launcher:AddRegionReqs(reg_reqs, args.region)
  -- add all fields to region requirement
  for _, field in pairs(args.relation._fields) do
    task_launcher:AddField(reg_reqs, field._fid)
  end
  -- launch task
  task_launcher:Execute(args.legion_runtime, args.legion_context)
  -- destroy task launcher
  task_launcher:Destroy()
end

-- empty task function launch
-- to make it work with legion without blowing up memory
-- THIS CODE IS EXTREMELY UGLY, AND HOPEFULLY ONLY TEMPORARY
function LW._TEMPORARY_LaunchEmptySingleTaskOnRelation(relation)
  if not use_partitioning then
    _TEMPORARY_LaunchEmptyLegionTask({
      legion_context = legion_env.ctx,
      legion_runtime = legion_env.runtime,
      relation       = relation,
    })
  else
    -- get partition data from planner
    local planner
    if use_partitioning then
      planner = require "ebb.src.planner"
    end
    local typeversion = {
      field_accesses = {},
      relation       = function() return relation end,
      all_accesses   = function() return {} end
    }
    planner.note_launch { typedfunc = typeversion }
    local legion_partition_data = planner.query_for_partitions(typeversion)
    local access, access_data = next(legion_partition_data)
    assert(next(legion_partition_data, access) == nil)
    -- launch emptu task per node/partition
    for node, region in ipairs(access_data.partition) do
      -- assert that there is only one region returned by planner and
      -- that the region is disjoint
      _TEMPORARY_LaunchEmptyLegionTask({
        legion_context = legion_env.ctx,
        legion_runtime = legion_env.runtime,
        relation       = relation,
        region         = region
      })
    end
  end
end
