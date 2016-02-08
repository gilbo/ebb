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

-- have this module expose the full C-API.  Then, we'll augment it below.
local APIblob = terralib.includecstring([[
#include "legion_c.h"
#include "reductions_cpu.h"
#include "ebb_mapper.h"
]])
for k,v in pairs(APIblob) do LW[k] = v end

local VERBOSE = rawget(_G, 'EBB_LOG_LEGION')


-------------------------------------------------------------------------------
--[[  Legion environment                                                   ]]--
-------------------------------------------------------------------------------


local LE = rawget(_G, '_legion_env')
local struct LegionEnv {
  runtime : LW.legion_runtime_t,
  ctx     : LW.legion_context_t
}
LE.legion_env = C.safemalloc( LegionEnv )
local legion_env = LE.legion_env[0]


-------------------------------------------------------------------------------
--[[  Data Accessors - Logical / Physical Regions, Fields, Futures         ]]--
-------------------------------------------------------------------------------

-- Field IDs
local fid_t = LW.legion_field_id_t

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
--[[  Legion Tasks                                                         ]]--
-------------------------------------------------------------------------------
--[[ A simple task is a task that does not have any return value. A future_task
--   is a task that returns a Legion future, or return value.
--]]--

local NUM_TASKS = 100

local TIDS = {
  simple_cpu = {},
  simple_gpu = {},
  future_cpu = {},
  future_gpu = {}
}
for i = 1, NUM_TASKS do
    TIDS.simple_cpu[i] = 99+i
    TIDS.simple_gpu[i] = 199+i
end
for i = 1, NUM_TASKS do
    TIDS.future_cpu[i] = 299+i
    TIDS.future_gpu[i] = 399+i
end

struct LW.TaskArgs {
  task        : LW.legion_task_t,
  regions     : &LW.legion_physical_region_t,
  num_regions : uint32,
  lg_ctx      : LW.legion_context_t,
  lg_runtime  : LW.legion_runtime_t
}

LW.TaskLauncher         = {}
LW.TaskLauncher.__index = LW.TaskLauncher

LW.SimpleTaskPtrType = { LW.TaskArgs } -> {}
LW.FutureTaskPtrType = { LW.TaskArgs } -> LW.legion_task_result_t

local USED_TIDS = {
  simple_cpu = 0,
  simple_gpu = 0,
  future_cpu = 0,
  future_gpu = 0
}

function LW.NewTaskLauncher(params)
  if not terralib.isfunction(params.taskfunc) then
    error('must provide Terra function as "taskfunc" argument', 2)
  end
  local taskfunc    = params.taskfunc
  local taskptrtype = &taskfunc:getdefinitions()[1]:gettype()
  local TID
  local taskfuncptr = C.safemalloc( taskptrtype )
  taskfuncptr[0]    = taskfunc:getdefinitions()[1]:getpointer()
  local task_ids    = params.task_ids

  -- get task id
  TID = params.gpu and params.task_ids.gpu or params.task_ids.cpu
  if not TID then
    local tid_str = ((taskptrtype == LW.SimpleTaskPtrType) and 'simple_') or
                    'future_'
    if params.gpu then
      tid_str = tid_str .. 'gpu'
    else
      tid_str = tid_str .. 'cpu'
    end
    local id = USED_TIDS[tid_str]
    TID = TIDS[tid_str][id + 1]
    if params.gpu then
      task_ids.gpu = TID
    else
      task_ids.cpu = TID
    end
    USED_TIDS[tid_str] = id + 1
    if VERBOSE then
      print("Ebb LOG: task id " .. tostring(taskfunc.name) ..
            " = " .. tostring(TID))
    end
  end

  -- create task launcher
  -- by looking carefully at the legion_c wrapper
  -- I was able to determine that we don't need to
  -- persist this structure
  local argstruct   = C.safemalloc( LW.legion_task_argument_t )
  argstruct.args    = taskfuncptr -- taskptrtype*
  argstruct.arglen  = terralib.sizeof(taskptrtype)
  local launcher_create =
    (params.use_index_launch and LW.legion_index_launcher_create) or
    LW.legion_task_launcher_create
  local launcher_args = terralib.newlist({TID})
  if params.use_index_launch then
    local argmap = LW.legion_argument_map_create()
    launcher_args:insertall({params.domain, argstruct[0], argmap,
    LW.legion_predicate_true(), false})
  else
    launcher_args:insertall({argstruct[0], LW.legion_predicate_true()})
  end
  launcher_args:insertall({0, 0})
  local launcher = launcher_create(unpack(launcher_args))

  return setmetatable({
    _taskfunc     = taskfunc,     -- important to prevent garbage collection
    _taskfuncptr  = taskfuncptr,  -- important to prevent garbage collection
    _TID          = TID,
    _launcher     = launcher,
    _index_launch = params.use_index_launch,
    _on_gpu       = params.gpu
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


function LW.TaskLauncher:AddRegionReq(reg_partn, parent, permission, coherence, redoptyp)
  local region_args = terralib.newlist({self._launcher, reg_partn.handle})
  local index_or_task_str = (self._index_launch and '_index') or '_task'
  local reg_or_partn_str = ((reg_partn == parent) and '_region') or '_partition'
  local p = permission
  local red_str = ((p == LW.REDUCE) and '_reduction') or ''
  local add_region_requirement =
    LW['legion' .. index_or_task_str ..
       '_launcher_add_region_requirement_logical' ..
       reg_or_partn_str ..red_str]
  if self._index_launch then
    region_args:insert(0)
  end
  local red_or_permission =
    ((p == LW.REDUCE) and LW.reduction_ids[redoptyp]) or p
  region_args:insertall({red_or_permission, coherence,
                         parent.handle, 0, false})
  return add_region_requirement(unpack(region_args))
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

-- If there's a future it will be returned
function LW.TaskLauncher:Execute(runtime, ctx, redoptype)
  local exec_str =
    'legion' .. ((self._index_launch and '_index') or '_task') .. '_launcher_execute'
  local exec_args = terralib.newlist({runtime, ctx, self._launcher})
  local reduce_id = nil
  if redoptype and self._index_launch then
    exec_args:insert(LW.reduction_ids[redoptype])
    exec_str = exec_str .. '_reduction'
  end
  return LW[exec_str](unpack(exec_args))
end


terra LW.simple_task(
  task        : LW.legion_task_t,
  regions     : &LW.legion_physical_region_t,
  num_regions : uint32,
  ctx         : LW.legion_context_t,
  runtime     : LW.legion_runtime_t
)
  var arglen = LW.legion_task_get_arglen(task)
  C.assert(arglen == sizeof(LW.SimpleTaskPtrType))
  var taskfunc = @[&LW.SimpleTaskPtrType](LW.legion_task_get_args(task))
  taskfunc( LW.TaskArgs { task, regions, num_regions, ctx, runtime } )
end

terra LW.future_task(
  task        : LW.legion_task_t,
  regions     : &LW.legion_physical_region_t,
  num_regions : uint32,
  ctx         : LW.legion_context_t,
  runtime     : LW.legion_runtime_t
) : LW.legion_task_result_t
  var arglen = LW.legion_task_get_arglen(task)
  C.assert(arglen == sizeof(LW.FutureTaskPtrType))
  var taskfunc = @[&LW.FutureTaskPtrType](LW.legion_task_get_args(task))
  var result = taskfunc( LW.TaskArgs {
                            task, regions, num_regions, ctx, runtime } )
  return result
end

terra LW.RegisterTasks()
  escape
    local procs = {'cpu', 'gpu'}
    local tasks = {'simple', 'future'}
    for p = 1, 2 do
      local proc = procs[p]
      local legion_proc = ((proc == 'cpu') and LW.LOC_PROC)  or LW.TOC_PROC
      for t = 1, 2 do
        local task = tasks[t]
        local task_ids = TIDS[task .. '_' .. proc]
        local task_function = LW[task .. '_task']
        local name_format = task .. '_' .. proc .. '_%d'
        local reg_function =
          ((task == 'simple') and LW.legion_runtime_register_task_void) or
          LW.legion_runtime_register_task
        emit quote
          var ids = arrayof(LW.legion_task_id_t, [task_ids])
          for i = 0, NUM_TASKS do
            var name : int8[25]
            C.sprintf(name, name_format, ids[i])
            reg_function(
              ids[i], [legion_proc], true, false, 1,
              LW.legion_task_config_options_t {
                leaf = true,
                inner = false,
                idempotent = false },
              name, task_function)
          end  -- inner for loop
        end  -- quote
      end  -- task loop
    end  -- proc loop
  end  -- escape
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
    [16] = {},
    [24] = {},
  }
  for nbytes, list in pairs(self._field_reserve) do
    for i=1,40 do

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

  local fid = table.remove( self._field_reserve[typsize] )
  if not fid then
    error('Ran out of fields of size '..typsize..' to allocate;\n'..
          'This error is the result of a hack to investigate performance '..
          'issues in field allocation.  Fixing it 100% will probably '..
          'require Mike making changes in the Legion runtime.\n')
  end
  --local fid = LW.legion_field_allocator_allocate_field(
  --              self.fsa,
  --              typsize,
  --              allocate_field_fid_counter
  --            )
  --assert(fid == allocate_field_fid_counter)
  --allocate_field_fid_counter = allocate_field_fid_counter + 1
  return fid
end
function LogicalRegion:FreeField(fid)
  LW.legion_field_allocator_free_field(self.fsa, fid)
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
      accs[i] = LW.legion_physical_region_get_accessor_generic(prs[i])
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
      accs[i] = LW.legion_physical_region_get_accessor_generic(prs[i])
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

-- reduction ids
LW.reduction_ids = {}
-- NOTE: supporting only cpu reductions for +, *, max, min
-- on int, float and double supported right now
LW.reduction_types = {
  ['int']    = 'int32',
  ['float']  = 'float',
  ['double'] = 'double'
}
for i = 2,4 do
  LW.reduction_types['vec' .. tostring(i) .. 'f'] = 'float_vec' .. tostring(i)
  LW.reduction_types['vec' .. tostring(i) .. 'd'] = 'double_vec' .. tostring(i)
  LW.reduction_types['vec' .. tostring(i) .. 'i'] = 'int32_vec' .. tostring(i)
  for j = 2,4 do
    LW.reduction_types['mat' .. tostring(i) .. 'x' .. tostring(j) .. 'f']
      = 'float_mat' .. tostring(i) .. 'x' .. tostring(j)
    LW.reduction_types['mat' .. tostring(i) .. 'x' .. tostring(j) .. 'd']
      = 'double_mat' .. tostring(i) .. 'x' .. tostring(j)
    LW.reduction_types['mat' .. tostring(i) .. 'x' .. tostring(j) .. 'i']
      = 'int32_mat' .. tostring(i) .. 'x' .. tostring(j)
  end
end
LW.reduction_ops = {
  ['+']   = 'plus',
  ['*']   = 'times',
  ['max'] = 'max',
  ['min'] = 'min'
}
local num_reduction_functions = 0
for _, o in pairs(LW.reduction_ops) do
  for t, tt in pairs(LW.reduction_types) do
    local field_register_reduction =
      LW['register_reduction_field_' .. o .. '_' .. tt]
    if field_register_reduction then
      num_reduction_functions = num_reduction_functions + 1
      LW.reduction_ids['field_' .. o .. '_' .. t] = num_reduction_functions
    end
    local global_register_reduction =
      LW['register_reduction_global_' .. o .. '_' .. tt]
    if global_register_reduction then
      num_reduction_functions = num_reduction_functions + 1
      LW.reduction_ids['global_' .. o .. '_' .. t] = num_reduction_functions
    end
  end
end
terra LW.RegisterReductions()
  escape
    for _, o in pairs(LW.reduction_ops) do
      for t, tt in pairs(LW.reduction_types) do
        local field_register_reduction =
          LW['register_reduction_field_' .. o .. '_' .. tt]
        if field_register_reduction then
          local reduction_id = LW.reduction_ids['field_' .. o .. '_' .. t]
          emit `field_register_reduction(reduction_id)
        end
        local global_register_reduction =
          LW['register_reduction_global_' .. o .. '_' .. tt]
        if global_register_reduction then
          local reduction_id = LW.reduction_ids['global_' .. o .. '_' .. t]
          emit `global_register_reduction(reduction_id)
        end
      end
    end
  end
end


-------------------------------------------------------------------------------
--[[  Miscellaneous methods                                                ]]--
-------------------------------------------------------------------------------

function LW.heavyweightBarrier()
  LW.legion_runtime_issue_execution_fence(legion_env.runtime, legion_env.ctx)
end
