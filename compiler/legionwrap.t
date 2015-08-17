
-- This file only depends on the C headers and legion,
-- so it's fairly separate from the rest of the compiler,
-- so it won't cause inadvertent dependency loops.

local LW = {}
package.loaded["compiler.legionwrap"] = LW

local C = require "compiler.c"

-- have this module expose the full C-API.  Then, we'll augment it below.
local APIblob = terralib.includecstring([[
#include "legion_c.h"
#include "liszt_gpu_mapper.h"
]])
for k,v in pairs(APIblob) do LW[k] = v end


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
    strides : LW.legion_byte_offset_t[d]
  }
  LW.FieldAccessor[d].__typename =
    function() return 'FieldAccessor_'..tostring(d) end
end

local LegionRect = {}
LW.LegionRect = LegionRect
local LegionGetRectFromDom = {}
LW.LegionGetRectFromDom = LegionGetRectFromDom
local LegionRawPtrFromAcc = {}
LW.LegionRawPtrFromAcc = LegionRawPtrFromAcc

LegionRect[1] = LW.legion_rect_1d_t
LegionRect[2] = LW.legion_rect_2d_t
LegionRect[3] = LW.legion_rect_3d_t

LegionGetRectFromDom[1] = LW.legion_domain_get_rect_1d
LegionGetRectFromDom[2] = LW.legion_domain_get_rect_2d
LegionGetRectFromDom[3] = LW.legion_domain_get_rect_3d

LegionRawPtrFromAcc[1] = LW.legion_accessor_generic_raw_rect_ptr_1d
LegionRawPtrFromAcc[2] = LW.legion_accessor_generic_raw_rect_ptr_2d
LegionRawPtrFromAcc[3] = LW.legion_accessor_generic_raw_rect_ptr_3d


-------------------------------------------------------------------------------
--[[  Task Launcher                                                        ]]--
-------------------------------------------------------------------------------


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

function LW.NewTaskLauncher(params)
  if not terralib.isfunction(params.taskfunc) then
    error('must provide Terra function as "taskfunc" argument', 2)
  end
  local taskfunc    = params.taskfunc
  local taskptrtype = &taskfunc:getdefinitions()[1]:gettype()
  local TID
  local taskfuncptr = C.safemalloc( taskptrtype )
  taskfuncptr[0]    = taskfunc:getdefinitions()[1]:getpointer()

  if     taskptrtype == LW.SimpleTaskPtrType then
    TID = LW.TID_SIMPLE_CPU
    if params.gpu then TID = LW.TID_SIMPLE_GPU end
  elseif taskptrtype == LW.FutureTaskPtrType then
    TID = LW.TID_FUTURE_CPU
    if params.gpu then TID = LW.TID_FUTURE_GPU end
  else
    error('The supplied function had ptr type\n'..
          '  '..tostring(taskptrtype)..'\n'..
          'Was expecting one of the following types\n'..
          '  '..tostring(LW.SimpleTaskPtrType)..'\n'..
          '  '..tostring(LW.FutureTaskPtrType)..'\n', 2)
  end

  -- By looking carefully at the legion_c wrapper
  -- I was able to determine that we don't need to
  -- persist this structure
  local argstruct   = C.safemalloc( LW.legion_task_argument_t )
  argstruct.args    = taskfuncptr -- taskptrtype*
  argstruct.arglen  = terralib.sizeof(taskptrtype)

  local launcher = LW.legion_task_launcher_create(
    TID,
    argstruct[0],
    LW.legion_predicate_true(),
    0,
    0
  )

  return setmetatable({
    _taskfunc     = taskfunc,     -- important to prevent garbage collection
    _taskfuncptr  = taskfuncptr,  -- important to prevent garbage collection
    _TID          = TID,
    _launcher     = launcher,
  }, LW.TaskLauncher)
end

function LW.TaskLauncher:Destroy()
  self._taskfunc = nil
  self._taskfuncptr = nil
  LW.legion_task_launcher_destroy(self._launcher)
  self._launcher = nil
end

function LW.TaskLauncher:AddRegionReq(lr, permission, coherence)
  local reg_req =
  LW.legion_task_launcher_add_region_requirement_logical_region(
    self._launcher,
    lr.handle,
    permission,
    coherence,
    lr.handle, -- superfluous parent ?
    0,
    false
  )
  return reg_req
end

function LW.TaskLauncher:AddField(reg_req, fid)
  LW.legion_task_launcher_add_field(
    self._launcher,
    reg_req,
    fid,
    true
  )
end

function LW.TaskLauncher:AddFuture(future)
  LW.legion_task_launcher_add_future(self._launcher, future)
end

-- If there's a future it will be returned
function LW.TaskLauncher:Execute(runtime, ctx)
  return LW.legion_task_launcher_execute(runtime, ctx, self._launcher)
end

-------------------------------------------------------------------------------
--[[  Legion Tasks                                                         ]]--
-------------------------------------------------------------------------------
--[[ A simple task is a task that does not have any return value. A future_task
--   is a task that returns a Legion future, or return value.
--]]--


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

LW.TID_SIMPLE_CPU = 200
LW.TID_SIMPLE_GPU = 250

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

LW.TID_FUTURE_CPU = 300
LW.TID_FUTURE_GPU = 350


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
  local ttype = global:Type():terraType()
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
    blob_size = blob_size + terralib.sizeof(g:Type():terraType())
  end
  local data_blob = terralib.cast(&uint8, C.malloc(blob_size))
  local f = { ref_count = 0 }
  setmetatable(f, FutureBlob)
  local offset = 0
  for _, g in pairs(globals) do
    local old_future = g:Data()
    local tsize = terralib.sizeof(g:Type():terraType())
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
  local ttype = global:Type():terraType()
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
function LogicalRegion:AllocateField(typ)
  local fid = LW.legion_field_allocator_allocate_field(
                self.fsa,
                terralib.sizeof(typ),
                allocate_field_fid_counter
              )
  assert(fid == allocate_field_fid_counter)
  allocate_field_fid_counter = allocate_field_fid_counter + 1
  return fid
end
function LogicalRegion:FreeField(fid)
  LW.legion_field_allocator_free_field(self.fsa, fid)
end

-- Internal method: Ask Legion to create 1 dimensional index space
local terra Create1DGridIndexSpace(x : int)
  var pt_lo = LW.legion_point_1d_t { arrayof(int, 0) }
  var pt_hi = LW.legion_point_1d_t { arrayof(int, x-1) }
  var rect  = LW.legion_rect_1d_t { pt_lo, pt_hi }
  var dom   = LW.legion_domain_from_rect_1d(rect)
  return LW.legion_index_space_create_domain(
            legion_env.runtime, legion_env.ctx, dom)
end

-- Internal method: Ask Legion to create 2 dimensional index space
local terra Create2DGridIndexSpace(x : int, y : int)
  var pt_lo = LW.legion_point_2d_t { arrayof(int, 0, 0) }
  var pt_hi = LW.legion_point_2d_t { arrayof(int, x-1, y-1) }
  var rect  = LW.legion_rect_2d_t { pt_lo, pt_hi }
  var dom   = LW.legion_domain_from_rect_2d(rect)
  return LW.legion_index_space_create_domain(
            legion_env.runtime, legion_env.ctx, dom)
end

-- Internal method: Ask Legion to create 3 dimensional index space
local terra Create3DGridIndexSpace(x : int, y : int, z : int)
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
  local l = {
              type      = 'unstructured',
              relation  = params.relation,
              field_ids = 0,
              n_rows    = params.n_rows,
              live_rows = 0,
              max_rows  = params.n_rows
            }
  if l.max_rows == 0 then l.max_rows = 1 end  -- legion throws an error with 0 max rows
  l.is  = LW.legion_index_space_create(legion_env.runtime,
                                       legion_env.ctx, l.max_rows)
  l.isa = LW.legion_index_allocator_create(legion_env.runtime,
                                           legion_env.ctx, l.is)
  -- field space
  l.fs  = LW.legion_field_space_create(legion_env.runtime,
                                       legion_env.ctx)
  l.fsa = LW.legion_field_allocator_create(legion_env.runtime,
                                           legion_env.ctx, l.fs)
  -- logical region
  l.handle = LW.legion_logical_region_create(legion_env.runtime,
                                             legion_env.ctx, l.is, l.fs)
  setmetatable(l, LogicalRegion)
  -- actually allocate rows
  l:AllocateRows(l.n_rows)
  return l
end

-- Allocate a structured logical region
-- NOTE: Call from top level task only.
function LW.NewGridLogicalRegion(params)
  local l = {
              type        = 'grid',
              relation    = params.relation,
              field_ids   = 0,
              bounds      = params.dims,
              dimensions  = #params.dims,
            }
  -- index space
  local bounds = l.bounds
  if l.dimensions == 1 then
    l.is = Create1DGridIndexSpace(bounds[1])
  end
  if l.dimensions == 2 then
    l.is = Create2DGridIndexSpace(bounds[1], bounds[2])
  end
  if l.dimensions == 3 then
    l.is = Create3DGridIndexSpace(bounds[1], bounds[2], bounds[3])
  end
  -- field space
  l.fs = LW.legion_field_space_create(legion_env.runtime,
                                      legion_env.ctx)
  l.fsa = LW.legion_field_allocator_create(legion_env.runtime,
                                           legion_env.ctx, l.fs)
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
    LW.legion_inline_launcher_add_field(ils[i], params.fields[i].fid, true)
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
    local rect = LW.LegionGetRectFromDom[ndims](domain)
    for d = 1, ndims do
      assert(dims[d] == (rect.hi.x[d-1] - rect.lo.x[d-1] + 1))
    end
    local subrect = terralib.new((LW.LegionRect[ndims])[1])
    local stride = terralib.new(LW.legion_byte_offset_t[ndims])
    for i, field in ipairs(params.fields) do
      accs[i] = LW.legion_physical_region_get_field_accessor_generic(prs[i], field.fid)
      ptrs[i] = terralib.cast(&uint8,
                              LW.LegionRawPtrFromAcc[ndims](accs[i], rect, subrect, stride))
      local s = {}
      strides[i] = s
      for d = 1, ndims do
        s[d] = stride[d-1].offset
      end
      offsets[i] = 0
      LW.legion_accessor_generic_destroy(accs[i])
    end
  else
    local base = terralib.new((&opaque)[1])
    local stride = terralib.new(uint64[1])
    for i, field in ipairs(params.fields) do
      accs[i] = LW.legion_physical_region_get_field_accessor_generic(prs[i], field.fid)
      LW.legion_accessor_generic_get_soa_parameters(accs[i], base, stride)
      ptrs[i]    = terralib.cast(&uint8, base[0])
      strides[i] = { stride[0] }
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
    offsets           = offsets
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
--[[  Methods for Introspecting on the Machine we're running on            ]]--
-------------------------------------------------------------------------------


function LW.GetMachineData ()
end
--[[

  void
  legion_machine_get_all_processors(
    legion_machine_t machine,
    legion_processor_t *processors,
    unsigned processors_size);

  /**
   * @see LegionRuntime::LowLevel::Machine::get_all_processors()
   */
  unsigned
  legion_machine_get_all_processors_size(legion_machine_t machine);


  // -----------------------------------------------------------------------
  // Processor Operations
  // -----------------------------------------------------------------------

  /**
   * @see LegionRuntime::LowLevel::Processor::kind()
   */
  legion_processor_kind_t
  legion_processor_kind(legion_processor_t proc_);

  // -----------------------------------------------------------------------
  // Memory Operations
  // -----------------------------------------------------------------------

  /**
   * @see LegionRuntime::LowLevel::Memory::kind()
   */
  legion_memory_kind_t
  legion_memory_kind(legion_memory_t proc_);


  // -----------------------------------------------------------------------
  // Machine Query Interface Operations
  // -----------------------------------------------------------------------

  /**
   * @return Caller takes ownership of return value.
   *
   * @see LegionRuntime::HighLevel::MappingUtilities::MachineQueryInterface()
   */
  legion_machine_query_interface_t
  legion_machine_query_interface_create(legion_machine_t machine);

  /**
   * @param handle Caller must have ownership of parameter `handle`.
   *
   * @see LegionRuntime::HighLevel::MappingUtilities::~MachineQueryInterface()
   */
  void
  legion_machine_query_interface_destroy(
    legion_machine_query_interface_t handle);

  /**
   * @see LegionRuntime::HighLevel::MappingUtilities
   *                   ::MachineQueryInterface::find_memory_kind()
   */
  legion_memory_t
  legion_machine_query_interface_find_memory_kind(
    legion_machine_query_interface_t handle,
    legion_processor_t proc,
    legion_memory_kind_t kind);
]]


---------------------)(Q#$&Y@)#*$(*&_)@----------------------------------------
--[[         GILBERT added this to make field loading work.                ]]--
--[[           Think of this as a workspace, not final organization        ]]--
-----------------------------------------)(Q#$&Y@)#*$(*&_)@--------------------


local function iterate1d(n)
  local i = -1
  return function()
    i = i+1
    if i>= n then return nil end
    return i
  end
end
local function iterate2d(nx,ny)
  local xi = -1
  local yi = 0
  return function()
    xi = xi+1
    if xi >= nx then xi = 0; yi = yi + 1 end
    if yi >= ny then return nil end
    return xi, yi
  end
end
local function iterate3d(nx,ny,nz)
  local xi = -1
  local yi = 0
  local zi = 0
  return function()
    xi = xi+1
    if xi >= nx then xi = 0; yi = yi + 1 end
    if yi >= ny then yi = 0; zi = zi + 1 end
    if zi >= nz then return nil end
    return xi, yi, zi
  end
end
local function linid(ids,dims)
      if #dims == 1 then return ids[1]
  elseif #dims == 2 then return ids[1] + dims[1] * ids[2]
  elseif #dims == 3 then return ids[1] + dims[1] * (ids[2] + dims[2]*ids[3])
  else error('INTERNAL > 3 dimensional address???') end
end


function LW.heavyweightBarrier()
  LW.legion_runtime_issue_execution_fence(legion_env.runtime, legion_env.ctx)
end

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
  if not params.logical_region then
    error('Expects logical_region argument', 2)
  elseif not params.privilege then
    error('Expects privilege argument', 2)
  elseif not params.fields then
    error('Expects fields list argument', 2)
  end
  if not params.dimensions then
    error('Expects dimensions argument', 2)
  end

  local ils   = {}
  local prs   = {}
  local fids  = {}

  for i,fid in ipairs(params.fields) do
    fids[i] = fid
    -- create inline launcher
    ils[i]  = LW.legion_inline_launcher_create_logical_region(
      params.logical_region,  -- legion_logical_region_t handle
      params.privilege,       -- legion_privilege_mode_t
      LW.EXCLUSIVE,           -- legion_coherence_property_t
      params.logical_region,  -- legion_logical_region_t parent
      0,                      -- legion_mapping_tag_id_t region_tag /* = 0 */
      false,                  -- bool verified /* = false*/
      0,                      -- legion_mapper_id_t id /* = 0 */
      0                       -- legion_mapping_tag_id_t launcher_tag /* = 0 */
    )
    -- add field to launcher
    LW.legion_inline_launcher_add_field(ils[i], fids[i], true)
    -- execute launcher to get physical region
    prs[i]  = LW.legion_inline_launcher_execute(legion_env.runtime,
                                                legion_env.ctx, ils[i])
  end

  local launchobj = setmetatable({
    inline_launchers  = ils,
    physical_regions  = prs,
    fids              = fids,
    dimensions        = params.dimensions,
  }, LW.ControlScanner)

  return launchobj
end

function LW.ControlScanner:ScanThenClose()
  local dims = self.dimensions
  local accs = {}
  local offs = {}
  local ptrs = {}

  -- initialize field-independent data for iteration
  local subrect
  local rect
  local get_raw_rect_ptr
  if #dims == 1 then
    subrect            = C.safemalloc( LW.legion_rect_1d_t )
    rect               = C.safemalloc( LW.legion_rect_1d_t )
    get_raw_rect_ptr   = LW.legion_accessor_generic_raw_rect_ptr_1d
  elseif #dims == 2 then
    subrect            = C.safemalloc( LW.legion_rect_2d_t )
    rect               = C.safemalloc( LW.legion_rect_2d_t )
    get_raw_rect_ptr   = LW.legion_accessor_generic_raw_rect_ptr_2d
  elseif #dims == 3 then
    subrect            = C.safemalloc( LW.legion_rect_3d_t )
    rect               = C.safemalloc( LW.legion_rect_3d_t )
    get_raw_rect_ptr   = LW.legion_accessor_generic_raw_rect_ptr_3d
  else
    error('INTERNAL n_dimensions > 3')
  end
  for d=1,#dims do
    rect.lo.x[d-1] = 0
    rect.hi.x[d-1] = dims[d]-1
  end

  -- initialize field-dependent data for iteration
  for k,fid in ipairs(self.fids) do
    local pr  = self.physical_regions[k]
    accs[k]   = LW.legion_physical_region_get_field_accessor_generic(pr, fid)
    local offtemp = C.safemalloc(LW.legion_byte_offset_t[#dims])
    ptrs[k] = terralib.cast(&int8, get_raw_rect_ptr(
                accs[k], rect[0], subrect,
                terralib.cast(&LW.legion_byte_offset_t, offtemp)
              ))
    offs[k] = {}
    for d=1,#dims do
      offs[k][d] = offtemp[0][d-1].offset
    end
  end

  -- define what to do when the iteration terminates
  local function close_up()
    for k=1,#self.fids do
      LW.legion_accessor_generic_destroy(accs[k])
    end
    self:close()
    return nil
  end

  -- define an iterator/generator
  if #dims == 1 then
    local iter = iterate1d(dims[1])
    return function()
      local i = iter()
      if i == nil then return close_up() end

      local callptrs = {}
      for fi=1,#self.fids do 
        callptrs[fi] = ptrs[fi] + i*offs[fi][1]
      end
      return {i}, callptrs
    end
  elseif #dims == 2 then
    local iter = iterate2d(dims[1], dims[2])
    return function()
      local xi,yi = iter()
      if xi == nil then return close_up() end

      local callptrs = {}
      for fi=1,#self.fids do 
        callptrs[fi] = ptrs[fi] + yi*offs[fi][2] + xi*offs[fi][1]
      end
      return {xi,yi}, callptrs
    end
  elseif #dims == 3 then
    local iter = iterate3d(dims[1], dims[2], dims[3])
    return function()
      local xi,yi,zi = iter()
      if xi == nil then return close_up() end

      local callptrs = {}
      for fi=1,#self.fids do 
        callptrs[fi] = ptrs[fi] +
                       zi*offs[fi][3] + yi*offs[fi][2] + xi*offs[fi][1]
      end
      return {xi,yi,zi}, callptrs
    end
  end
end


function LW.ControlScanner:close()
  for i=1,#self.fids do
    local il  = self.inline_launchers[i]
    local pr  = self.physical_regions[i]

    LW.legion_runtime_unmap_region(legion_env.runtime, legion_env.ctx, pr)
    LW.legion_physical_region_destroy(pr)
    LW.legion_inline_launcher_destroy(il)
  end
end





