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
--[[  Data Accessors - Logical / Physical Regions, Fields, Futures         ]]--
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
LW.FutureBlob           = FutureBlob

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

local LegionCreatePoint = {}
LegionCreatePoint[1] = terra(pt1 : int)
  return [LW.LegionPoint[1]]{array(pt1)}
end
LegionCreatePoint[2] = terra(pt1 : int, pt2 : int)
  return [LW.LegionPoint[2]]{array(pt1, pt2)}
end
LegionCreatePoint[3] = terra(pt1 : int, pt2 : int, pt3 : int)
  return [LW.LegionPoint[3]]{array(pt1, pt2, pt3)}
end


-------------------------------------------------------------------------------
--[[  Region Requirements                                                  ]]--
-------------------------------------------------------------------------------

LW.RegionReq         = {}
LW.RegionReq.__index = LW.RegionReq

--[[
{
  relation,
  privilege
  coherence,
  reduce_op,
  reduce_typ,
  centered
  num
}
--]]
function LW.NewRegionReq(params)
  local reduce_func_id = (params.privilege == LW.REDUCE)
            and LW.GetFieldReductionId(params.reduce_op, params.reduce_typ)
             or nil
  local relation = params.relation
  local reg_req = setmetatable({
    log_reg_handle  = relation._logical_region_wrapper.handle,
    num             = params.num,
    -- GOAL: Remove the next data item as a dependency
    --        through more refactoring
    relation_for_partitioning = relation,
    privilege       = params.privilege,
    coherence       = params.coherence,
    reduce_func_id  = reduce_func_id,       
    partition       = nil,
    centered        = params.centered,
  }, LW.RegionReq)
  return reg_req
end

function LW.RegionReq:isreduction() return self.privilege == LW.REDUCE end

function LW.RegionReq:PartitionData()  -- TODO: make default case single partition
  local relation = self.relation_for_partitioning
  if not self.partition and use_partitioning then
    if self.centered then
      if not relation:IsPartitioningSet() then
        local ndims = #relation:Dims()
        local num_partitions = {}
        for i = 1,ndims do
            num_partitions[i] = run_config.num_partitions_default
        end
        relation:SetPartitions(num_partitions)
      end
      self.partition = relation:GetOrCreateDisjointPartitioning()
    end
  end
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
  local ir = terralib.saveobj(nil, "llvmir", { entry = task_func })
  local terra register()
    escape
      if use_llvm then
        emit quote
          LW.legion_runtime_register_task_variant_llvmir(
            legion_env.runtime, TID, [(on_gpu and LW.TOC_PROC) or LW.LOC_PROC],
            true,
            LW.legion_task_config_options_t { leaf = true, inner = false,
                                              idempotent = false },
            ufv_name, [&opaque](0), 0,
            ir, "entry")
        end  -- emit quote
      else
        emit quote
          LW.legion_runtime_register_task_variant_fnptr(
            legion_env.runtime, TID, [(on_gpu and LW.TOC_PROC) or LW.LOC_PROC],
            LW.legion_task_config_options_t { leaf = true, inner = false,
                                              idempotent = false },
            ufv_name, [&opaque](0), 0,
            task_func)
        end  -- emit quote
      end  -- if else
    end  -- escape
  end
  register()
  return TID
end)

LW.TaskLauncher         = {}
LW.TaskLauncher.__index = LW.TaskLauncher

--[[
{
  taskname
  taskfunc
  gpu               -- boolean
  use_index_launch  -- boolean
  domain            -- ??
}
--]]
LW.TaskLauncher         = {}
LW.TaskLauncher.__index = LW.TaskLauncher

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
  local launcher_args = terralib.newlist({TID})
  if params.use_index_launch then
    local argmap = LW.legion_argument_map_create()  -- partition specific arguments
    launcher_args:insertall({params.domain, argstruct[0], argmap,
    LW.legion_predicate_true(), false})
  else
    launcher_args:insertall({argstruct[0], LW.legion_predicate_true()})
  end
  launcher_args:insertall({0, 0})
  local launcher = launcher_create(unpack(launcher_args))

  return setmetatable({
    _TID          = TID,
    _launcher     = launcher,
    _index_launch = params.use_index_launch,
    _on_gpu       = params.gpu,
    _reduce_func_id = nil,
  }, LW.TaskLauncher)
end

function LW.TaskLauncher:Destroy()
  self._taskfunc = nil
  self._taskfuncptr = nil
  local DestroyVariant = self._index_launch and
    LW.legion_index_launcher_destroy or
    LW.legion_task_launcher_destroy
  DestroyVariant(self._launcher)
  self._launcher = nil
end

function LW.TaskLauncher:IsOnGPU()
  return self._on_gpu
end

function LW.TaskLauncher:AddRegionReq(req)
  local use_part = req.partition ~= nil
  local partition_handle = use_part and req.partition.handle
                                     or req.log_reg_handle

  -- Assemble the call
  local args = terralib.newlist { self._launcher, partition_handle }
  local str  = 'legion'
  if self._index_launch then    args:insert(0)
                                str = str .. '_index'
                        else    str = str .. '_task' end
  str = str .. '_launcher_add_region_requirement_logical'
  if use_part then              str = str .. '_partition'
              else              str = str .. '_region' end
  if req:isreduction() then     args:insert(req.reduce_func_id)
                                str = str .. '_reduction'
                       else     args:insert(req.privilege)
                       end
  args:insertall { req.coherence, req.log_reg_handle, 0, false }
  -- Do the call
  return LW[str](unpack(args))
end

function LW.TaskLauncher:AddField(reg_req, fid)
  local AddFieldVariant =
    (self._index_launch and LW.legion_index_launcher_add_field) or
    LW.legion_task_launcher_add_field
  AddFieldVariant(self._launcher, reg_req, fid, true)
end

function LW.TaskLauncher:AddFuture(future)
  local AddFutureVariant =
    (self._index_launch and LW.legion_index_launcher_add_future) or
    LW.legion_task_launcher_add_future
  AddFutureVariant(self._launcher, future)
end

function LW.TaskLauncher:AddFutureReduction(op, ebb_typ)
  self._reduce_func_id = LW.GetGlobalReductionId(op, ebb_typ)
end

-- If there's a future it will be returned
function LW.TaskLauncher:Execute(runtime, ctx, redop_id)
  local exec_str = 'legion'..
                   (self._index_launch and '_index' or '_task')..
                   '_launcher_execute'
  local exec_args = terralib.newlist({runtime, ctx, self._launcher})
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
  return fid
end

function LogicalRegion:FreeField(fid)
  LW.legion_field_allocator_free_field(self.fsa, fid)
end

function LogicalRegion:AttachNameToField(fid, name)
  LW.legion_field_id_attach_name(legion_env.runtime, self.fs, fid,
                                 name, true)
end

local CreateGridIndexSpace = {}

-- Internal method: Ask Legion to create 1 dimensional index space
CreateGridIndexSpace[1] = terra(x : int)
  var pt_lo = LW.legion_point_1d_t { arrayof(int, 0) }
  var pt_hi = LW.legion_point_1d_t { arrayof(int, x-1) }
  var rect  = LW.legion_rect_1d_t { pt_lo, pt_hi }
  var dom   = LW.legion_domain_from_rect_1d(rect)
  return LW.legion_index_space_create_domain(
            legion_env.runtime, legion_env.ctx, dom)
end

-- Internal method: Ask Legion to create 2 dimensional index space
CreateGridIndexSpace[2] = terra(x : int, y : int)
  var pt_lo = LW.legion_point_2d_t { arrayof(int, 0, 0) }
  var pt_hi = LW.legion_point_2d_t { arrayof(int, x-1, y-1) }
  var rect  = LW.legion_rect_2d_t { pt_lo, pt_hi }
  var dom   = LW.legion_domain_from_rect_2d(rect)
  return LW.legion_index_space_create_domain(
            legion_env.runtime, legion_env.ctx, dom)
end

-- Internal method: Ask Legion to create 3 dimensional index space
CreateGridIndexSpace[3] = terra(x : int, y : int, z : int)
  var pt_lo = LW.legion_point_3d_t { arrayof(int, 0, 0, 0) }
  var pt_hi = LW.legion_point_3d_t { arrayof(int, x-1, y-1, z-1) }
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
    bounds      = params.dims,
    dimensions  = #params.dims,
  }, LogicalRegion)

  -- index space
  local bounds = l.bounds
  l.is = CreateGridIndexSpace[l.dimensions](unpack(bounds))
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
      params.relation._logical_region_wrapper.handle,  -- legion_logical_region_t handle
      params.privilege,         -- legion_privilege_mode_t
      LW.EXCLUSIVE,             -- legion_coherence_property_t
      params.relation._logical_region_wrapper.handle,  -- legion_logical_region_t parent
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
  }, LW.InlinePhysicalRegion)

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

local LogicalPartition = {}
LogicalPartition.__index = LogicalPartition

-- method used to create logical partition
function LogicalPartition:IsPartitionedByField()
  self.ptype = 'FIELD'
end
function LogicalPartition:IsPartitionedBlock()
  self.ptype = 'BLOCK'
end

-- coloring field used to create this logical partition
function LogicalPartition:ColoringField()
  return self.color_field
end

-- color space (partition domain)
function LogicalPartition:Domain()
  return self.domain
end

-- create a color space with num_colors number of colors
-- this corresponds to the number of partitions
local terra CreateColorSpace(num_colors : LW.legion_color_t)
  var lo = LW.legion_point_1d_t({arrayof(int, 0)})
  var hi = LW.legion_point_1d_t({arrayof(int, num_colors - 1)})
  var bounds = LW.legion_rect_1d_t({lo, hi})
  return LW.legion_domain_from_rect_1d(bounds)
end

-- create partition by coloring field
function LogicalRegion:CreatePartitionsByField(rfield)
  local color_space = CreateColorSpace(self.relation:TotalPartitions())
  local partn = LW.legion_index_partition_create_by_field(
    legion_env.runtime, legion_env.ctx,
    self.handle, self.handle, rfield._fid,
    color_space,
    100, false)
  local lp = LW.legion_logical_partition_create(
    legion_env.runtime, legion_env.ctx, self.handle, partn)
  local lp = {
               ptype       = 'FIELD',  -- partition type (by field or block)
               color_field = rfield,   -- field used to generate the partition
               domain      = color_space,  -- partition domain (color space)
               index_partn = partn,    -- legion index partition handle
               handle      = lp,       -- legion logical partition handle
             }
  setmetatable(lp, LogicalPartition)
  return lp
end

-- block partition helpers
local AddDomainColor = {}
AddDomainColor[1] = terra(coloring : LW.legion_domain_coloring_t,
                          color : LW.legion_color_t,
                          lo1 : int,
                          hi1 : int
                          )
  var lo_pt = [LW.LegionPoint[1]]{ array(lo1) }
  var hi_pt = [LW.LegionPoint[1]]{ array(hi1) }
  var domain = [LW.LegionDomFromRect[1]]([LW.LegionRect[1]]({ lo_pt, hi_pt }))
  LW.legion_domain_coloring_color_domain(coloring, color, domain)
end  -- terra function
AddDomainColor[2] = terra(coloring : LW.legion_domain_coloring_t,
                          color : LW.legion_color_t,
                          lo1 : int, lo2 : int,
                          hi1 : int, hi2 : int 
                          )
  var lo_pt = [LW.LegionPoint[2]]{ array(lo1, lo2) }
  var hi_pt = [LW.LegionPoint[2]]{ array(hi1, hi2) }
  var domain = [LW.LegionDomFromRect[2]]([LW.LegionRect[2]]({ lo_pt, hi_pt }))
  LW.legion_domain_coloring_color_domain(coloring, color, domain)
end  -- terra function
AddDomainColor[3] = terra(coloring : LW.legion_domain_coloring_t,
                          color : LW.legion_color_t,
                          lo1 : int, lo2 : int, lo3 : int,
                          hi1 : int, hi2 : int, hi3 : int
                          )
  var lo_pt = [LW.LegionPoint[3]]{ array(lo1, lo2, lo3) }
  var hi_pt = [LW.LegionPoint[3]]{ array(hi1, hi2, hi3) }
  var domain = [LW.LegionDomFromRect[3]]([LW.LegionRect[3]]({ lo_pt, hi_pt }))
  LW.legion_domain_coloring_color_domain(coloring, color, domain)
end  -- terra function

local function min(x, y)
  if x <= y then return x else return y end
end
local function max(x, y)
  if x >= y then return x else return y end
end

-- create block partitions
function LogicalRegion:CreateBlockPartitions(ghost_width) 
  -- NOTE: partitions include boundary regions. Partitioning is not subset
  -- specific right now, but it is a partitioning over the entire logical
  -- region.
  local num_partitions = self.relation:NumPartitions()
  local dims = self.relation:Dims()
  local ndims = #dims
  -- check if number of elements along each dimension is a multiple of number
  -- of partitions. compute number of elements along each dimension in each
  -- partition for the disjoint case.
  local divisible = {}
  local elems_lo = {}
  local elems_hi = {}
  local num_lo   = {}
  for d = 1, ndims do
    local num_elems = dims[d]
    local num_partns = num_partitions[d]
    divisible[d] = (num_elems % num_partns == 0)
    elems_lo[d] = math.floor(num_elems/num_partns)
    elems_hi[d] = math.ceil(num_elems/num_partns)
    num_lo[d]   = num_partns - (num_elems % num_partns)
  end
  -- check if partitioning will be disjoint
  local disjoint = true
  local ghost_width = ghost_width
  if ghost_width then
    for d = 1, 2*ndims do
      disjoint = disjoint and (ghost_width[d] == 0)
    end
  else
    ghost_width = {}
    for d = 1, 2*ndims do
      ghost_width[d] = 0
    end
  end
  -- color space
  local total_partitions = self.relation:TotalPartitions()
  local color_space = CreateColorSpace(total_partitions)
  local coloring = LW.legion_domain_coloring_create();
  -- determine number of elements in each partition/ color and create logical partition
  -- TODO: what should we do for periodic boundary conditions?
  local color = 0
  -- number of partitions in each dimension, initialization of loop variables
  local nps = {1, 1, 1}
  local lo = {}
  local hi = {}
  for d = 1,ndims do
    nps[d] = num_partitions[d]
    lo[d]  = 0
    hi[d]  = -1
  end
  for p3 = 1, nps[3] do
    if ndims > 2 then
      local d = 3
      local elems = (p3 > num_lo[d] and elems_hi[d]) or elems_lo[d]
      lo[d] = max(0, hi[d] + 1 - ghost_width[2*d - 1])
      hi[d] = min(dims[d] - 1, lo[d] + elems - 1 + ghost_width[2*d])
      lo[d-1] = 0
      hi[d-1] = -1
    end
    for p2 = 1, nps[2] do
      if ndims > 1 then
        local d = 2
        local elems = (p2 > num_lo[d] and elems_hi[d]) or elems_lo[d]
        lo[d] = max(0, hi[d] + 1 - ghost_width[2*d - 1])
        hi[d] = min(dims[d] - 1, lo[d] + elems - 1 + ghost_width[2*d])
        lo[d-1] = 0
        hi[d-1] = -1
      end
      for p1 = 1, nps[1] do
        local d = 1
        local elems = (p1 > num_lo[d] and elems_hi[d]) or elems_lo[d]
        lo[d] = max(0, hi[d] + 1 - ghost_width[2*d - 1])
        hi[d] = min(dims[d] - 1, lo[d] + elems - 1 + ghost_width[2*d])
        local color_args = terralib.newlist({coloring, color})
        color_args:insertall(lo)
        color_args:insertall(hi)
        AddDomainColor[ndims](unpack(color_args))
        color = color + 1
      end
    end
  end
  -- create logical partition with the coloring
  local partn = LW.legion_index_partition_create_domain_coloring(
    legion_env.runtime, legion_env.ctx, self.is, color_space, coloring, disjoint, -1)
  local lp = LW.legion_logical_partition_create(
    legion_env.runtime, legion_env.ctx, self.handle, partn)
  local lp = {
    ptype       = 'BLOCK',  -- partition type (by field or block)
    domain      = color_space,  -- partition domain (color_space)
    index_partn = partn,
    handle      = lp,
  }
  setmetatable(lp, LogicalPartition)
  return lp
end


-------------------------------------------------------------------------------
--[[  Methods for copying fields and scanning through fields/ regions       ]]--
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
  local reduction_op_translate = {
    ['+']   = 'plus',
    ['*']   = 'times',
    ['max'] = 'max',
    ['min'] = 'min'
  }
  local basetyps = {
    [T.int]     = 'int32',
    [T.float]   = 'float',
    [T.double]  = 'double',
  }

  -- construct list of all types with their mappings to string names
  local typ_map = {}
  for ebbt,lgt in pairs(basetyps) do typ_map[ebbt] = lgt end
  for i=2,4 do
    for ebbt,lgt in pairs(basetyps) do
      typ_map[T.vector(ebbt,i)] = lgt..'_vec'..i
    end
    for j=2,4 do
      for ebbt,lgt in pairs(basetyps) do
        typ_map[T.matrix(ebbt,i,j)] = lgt..'_mat'..i..'x'..j
      end
    end
  end

  -- now register all the corresponding functions
  for ebb_op, lg_op in pairs(reduction_op_translate) do
    for ebbt, lgt in pairs(typ_map) do
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

  C.printf("** WARNING: Executing empty tproc_id : LW.legion_lowlevel_id_t) ask. ")
  C.printf("This is a hack for Ebb/Legion. ")
  C.printf("If you are seeing this message and do not know what this is, ")
  C.printf("please contact the developers.\n")

  -- legion postamble
  LW.legion_task_postamble(task_args.lg_runtime, task_args.lg_ctx,
                           [&opaque](0), 0)
end

-- empty task function launcher
-- to make it work with legion without blowing up memory
local _TEMPORARY_memoize_empty_task_launcher = Util.memoize_named({
  'relation' },
  function(args)
    -- one region requirement for the relation
    local reg_req = LW.NewRegionReq {
      num             = 0,
      relation        = args.relation,
      privilege       = LW.READ_WRITE,
      coherence       = LW.EXCLUSIVE,
      centered        = true
    }
    -- task launcher
    local task_launcher = LW.NewTaskLauncher {
      ufv_name         = "_TEMPORARY_PrepareSimulation",
      task_func        = _TEMPORARY_EmptyTaskFunction,
      gpu              = false,
      use_index_launch = false,
      domain           = nil
    }
    -- add region requirement to task launcher
    task_launcher:AddRegionReq(reg_req)
    -- iterate over user define fields and subset boolmasks
    -- assumption: these are the only fields that are needed to force on physical
    -- instance with valid data for all fields over the region
    for _, field in pairs(args.relation._fields) do
      task_launcher:AddField(0, field._fid)
    end
    for _, subset in pairs(args.relation._subsets) do
      task_launcher:AddField(0, subset._boolmask._fid)
    end
    return task_launcher
  end
)

-- empty task function launch
-- to make it work with legion without blowing up memory
function LW._TEMPORARY_LaunchEmptySingleTaskOnRelation(relation)
  local task_launcher = _TEMPORARY_memoize_empty_task_launcher(
  {
    relation = relation
  })
  task_launcher:Execute(legion_env.runtime, legion_env.ctx)
end
