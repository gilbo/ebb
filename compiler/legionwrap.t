
-- This file only depends on the C headers and legion,
-- so it's fairly separate from the rest of the compiler,
-- so it won't cause inadvertent dependency loops.

local LW = {}
package.loaded["compiler.legionwrap"] = LW

local C = require "compiler.c"

-- have this module expose the full C-API.  Then, we'll augment it below.
local APIblob = terralib.includecstring([[
#include "legion_c.h"
]])
for k,v in pairs(APIblob) do LW[k] = v end


-------------------------------------------------------------------------------
--[[                          Legion environment                           ]]--
-------------------------------------------------------------------------------

local LE = rawget(_G, '_legion_env')
local struct EnvArgsForTerra {
  runtime : &LW.legion_runtime_t,
  ctx     : &LW.legion_context_t
}
LE.terraargs = global(EnvArgsForTerra)

-------------------------------------------------------------------------------
--[[                       Kernel Launcher Template                        ]]--
-------------------------------------------------------------------------------
--[[ Kernel laucnher template is a wrapper for a callback function that is
--   passed to a Legion task, when a Liszt kernel is invoked. We pass a
--   function as an argument to a Legion task, because there isn't a way to
--   dynamically register and invoke tasks.
--]]--

struct LW.TaskArgs {
  task        : LW.legion_task_t,
  regions     : &LW.legion_physical_region_t,
  num_regions : uint32,
  lg_ctx      : LW.legion_context_t,
  lg_runtime  : LW.legion_runtime_t
}


---------------
-- NO RETURN --
---------------

struct LW.SimpleKernelLauncherTemplate {
  Launch : { LW.TaskArgs } -> {};
}

terra LW.NewSimpleKernelLauncher( kernel_code : { LW.TaskArgs } -> {} )
  var l : LW.SimpleKernelLauncherTemplate
  l.Launch = kernel_code
  return l
end

LW.SimpleKernelLauncherSize = terralib.sizeof(LW.SimpleKernelLauncherTemplate)

-- Pack kernel launcher into a task argument for Legion.
terra LW.SimpleKernelLauncherTemplate:PackToTaskArg()
  var sub_args = LW.legion_task_argument_t {
    args       = [&opaque](self),
    arglen     = LW.SimpleKernelLauncherSize
  }
  return sub_args
end


-------------------------------
-- RETURN LEGION TASK RESULT --
-------------------------------

struct LW.FutureKernelLauncherTemplate {
  Launch : { LW.TaskArgs } -> LW.legion_task_result_t;
}

terra LW.NewFutureKernelLauncher( kernel_code : { LW.TaskArgs } ->
                                                LW.legion_task_result_t )
  var l : LW.FutureKernelLauncherTemplate
  l.Launch = kernel_code
  return l
end

LW.FutureKernelLauncherSize = terralib.sizeof(LW.FutureKernelLauncherTemplate)

-- Pack kernel launcher into a task argument for Legion.
terra LW.FutureKernelLauncherTemplate:PackToTaskArg()
  var sub_args = LW.legion_task_argument_t {
    args       = [&opaque](self),
    arglen     = LW.FutureKernelLauncherSize
  }
  return sub_args
end


-------------------------------------------------------------------------------
--[[                             Legion Tasks                              ]]--
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
  C.printf("Executing simple task\n")
  var arglen = LW.legion_task_get_arglen(task)
  C.printf("Arglen in task is %i\n", arglen)
  assert(arglen == LW.SimpleKernelLauncherSize)
  var kernel_launcher =
    [&LW.SimpleKernelLauncherTemplate](LW.legion_task_get_args(task))
  kernel_launcher.Launch( LW.TaskArgs {
    task, regions, num_regions, ctx, runtime
  } )
  C.printf("Completed executing simple task\n")
end

LW.TID_SIMPLE = 200

terra LW.future_task(
  task        : LW.legion_task_t,
  regions     : &LW.legion_physical_region_t,
  num_regions : uint32,
  ctx         : LW.legion_context_t,
  runtime     : LW.legion_runtime_t
) : LW.legion_task_result_t
  C.printf("Executing future task\n")
  var arglen = LW.legion_task_get_arglen(task)
  assert(arglen == LW.FutureKernelLauncherSize)
  var kernel_launcher =
    [&LW.FutureKernelLauncherTemplate](LW.legion_task_get_args(task))
  var result = kernel_launcher.Launch( LW.TaskArgs {
    task, regions, num_regions, ctx, runtime
  } )
  -- TODO: dummy seems likely broken.  It should refer to this task?
  C.printf("Completed executing future task task\n")
  return result
end

LW.TID_FUTURE = 300

-- GLB: Why do we need this table?
LW.TaskTypes = { simple = 'simple', future = 'future' }




-------------------------------------------------------------------------------
--[[                                 Types                                 ]]--
-------------------------------------------------------------------------------

local fid_t = LW.legion_field_id_t

local LogicalRegion     = {}
LogicalRegion.__index   = LogicalRegion
LW.LogicalRegion        = LogicalRegion

local PhysicalRegion    = {}
PhysicalRegion.__index  = PhysicalRegion
LW.PhysicalRegion       = PhysicalRegion


-------------------------------------------------------------------------------
--[[                            Future methods                             ]]--
-------------------------------------------------------------------------------

function LW.CreateFuture(typ, cdata)
  local data_type = typ:terraType()
  local data = terralib.new(data_type[1])
  data[0] = cdata
  local future = LW.legion_future_from_buffer(LE.runtime, data,
                                              terralib.sizeof(data_type))
  return future
end

function LW.DestroyFuture(future)
  LW.legion_future_destroy(future)
end

function LW.GetResultFromFuture(typ, future)
  local leg_result = LW.legion_future_get_result(future)
  local data_type = typ:terraType()
  local data = terralib.new(data_type, terralib.cast(&data_type, leg_result.value)[0])
  LW.legion_task_result_destroy(leg_result)
  return data
end


-------------------------------------------------------------------------------
--[[                        Logical region methods                         ]]--
-------------------------------------------------------------------------------

-- NOTE: Call from top level task only.
function LogicalRegion:AllocateRows(num)
  if self.type ~= 'unstructured' then
    error("Cannot allocate rows for grid relation ", self.relation:Name(), 3)
  else
    if self.rows_live + num > self.rows_max then
      error("Cannot allocate more rows for relation ", self.relation:Name())
    end
  end
  LW.legion_index_allocator_alloc(self.isa, num)
  self.rows_live = self.rows_live + num
end

-- NOTE: Assuming here that the compile time limit is never be hit.
-- NOTE: Call from top level task only.
function LogicalRegion:AllocateField(typ)
  local fid = LW.legion_field_allocator_allocate_field(
                 self.fsa, terralib.sizeof(typ.terratype), self.field_ids)
  self.field_ids = self.field_ids + 1
  return fid
end

-- Internal method: Ask Legion to create 1 dimensional index space
local terra Create1DGridIndexSpace(x : int)
  var pt_lo = LW.legion_point_1d_t { arrayof(int, 0) }
  var pt_hi = LW.legion_point_1d_t { arrayof(int, x-1) }
  var rect  = LW.legion_rect_1d_t { pt_lo, pt_hi }
  var dom   = LW.legion_domain_from_rect_1d(rect)
  return LW.legion_index_space_create_domain(
            @(LE.terraargs.runtime), @(LE.terraargs.ctx), dom)
end

-- Internal method: Ask Legion to create 2 dimensional index space
local terra Create2DGridIndexSpace(x : int, y : int)
  var pt_lo = LW.legion_point_2d_t { arrayof(int, 0, 0) }
  var pt_hi = LW.legion_point_2d_t { arrayof(int, x-1, y-1) }
  var rect  = LW.legion_rect_2d_t { pt_lo, pt_hi }
  var dom   = LW.legion_domain_from_rect_2d(rect)
  return LW.legion_index_space_create_domain(
            @(LE.terraargs.runtime), @(LE.terraargs.ctx), dom)
end

-- Internal method: Ask Legion to create 3 dimensional index space
local terra Create3DGridIndexSpace(x : int, y : int, z : int)
  var pt_lo = LW.legion_point_3d_t { arrayof(int, 0, 0, 0) }
  var pt_hi = LW.legion_point_3d_t { arrayof(int, x-1, y-1, z-1) }
  var rect  = LW.legion_rect_3d_t { pt_lo, pt_hi }
  var dom   = LW.legion_domain_from_rect_3d(rect)
  return LW.legion_index_space_create_domain(
            @(LE.terraargs.runtime), @(LE.terraargs.ctx), dom)
end

-- Allocate an unstructured logical region
-- NOTE: Call from top level task only.
function LW.NewLogicalRegion(params)
  local l = {
              type      = 'unstructured',
              relation  = params.relation,
              field_ids = 0,
              n_rows    = params.n_rows,
            }
  -- index space
  l.is  = Create1DGridIndexSpace(l.n_rows)
  l.isa = LW.legion_index_allocator_create(LE.runtime, LE.ctx, l.is)
  -- field space
  l.fs  = LW.legion_field_space_create(LE.runtime, LE.ctx)
  l.fsa = LW.legion_field_allocator_create(LE.runtime, LE.ctx, l.fs)
  -- logical region
  l.handle = LW.legion_logical_region_create(LE.runtime, LE.ctx, l.is, l.fs)
  setmetatable(l, LogicalRegion)
  return l
end

-- Allocate a structured logical region
-- NOTE: Call from top level task only.
function LW.NewGridLogicalRegion(params)
  local l = {
              type        = 'grid',
              relation    = params.relation,
              field_ids   = 0,
              bounds      = params.bounds,
              dimensions  = params.dimensions,
            }
  -- index space
  local bounds = params.bounds
  if params.dimensions == 1 then
    l.is = Create1DGridIndexSpace(bounds[1])
  end
  if params.dimensions == 2 then
    l.is = Create2DGridIndexSpace(bounds[1], bounds[2])
  end
  if params.dimensions == 3 then
    l.is = Create3DGridIndexSpace(bounds[1], bounds[2], bounds[3])
  end
  -- field space
  l.fs = LW.legion_field_space_create(LE.runtime, LE.ctx)
  l.fsa = LW.legion_field_allocator_create(LE.runtime, LE.ctx, l.fs)
  -- logical region
  l.handle = LW.legion_logical_region_create(LE.runtime, LE.ctx, l.is, l.fs)
  setmetatable(l, LogicalRegion)
  return l
end


-------------------------------------------------------------------------------
--[[                    Privilege and coherence values                     ]]--
-------------------------------------------------------------------------------

-- There are more privileges in Legion, like write discard. We should separate
-- exclusive into read_write and write_discard for better performance.
LW.privilege = {
  EXCLUSIVE           = LW.READ_WRITE,
  READ                = LW.READ_ONLY,
  READ_OR_EXCLUISVE   = LW.READ_WRITE,
  REDUCE              = LW.REDUCE,
  REDUCE_OR_EXCLUSIVE = LW.REDUCE,
}

-- TODO: How should we use this? Right now, read/ exclusive use EXCLUSIVE,
-- reduction uses ATOMIC.
LW.coherence = {
  EXCLUSIVE           = LW.EXCLUSIVE,
  READ                = LW.EXCLUSIVE,
  READ_OR_EXCLUISVE   = LW.EXCLUSIVE,
  REDUCE              = LW.REDUCE,
  REDUCE_OR_EXCLUSIVE = LW.REDUCE,
}


-------------------------------------------------------------------------------
--[[                        Physical region methods                        ]]--
-------------------------------------------------------------------------------


-- Create inline physical region, useful when physical regions are needed in
-- the top level task.
-- NOTE: Call from top level task only.
-- TODO: This is broken
-- function LogicalRegion:CreatePhysicalRegion(params)
--   local lreg = self.handle
--   local privilege = params.privilege or LW.privilege.default
--   local coherence = params.coherence or LW.coherence.default
--   local input_launcher = LW.legion_inline_launcher_create_logical_region(
--                             lreg, privilege, coherence, lreg,
--                             0, false, 0, 0)
--   local fields = params.fields
--   for i = 1, #fields do
--     LW.legion_inline_launcher_add_field(input_launcher, fields[i].fid, true)
--   end
--   local p = {}
--   p.handle =
--     LW.legion_inline_launcher_execute(LE.runtime, LE.ctx, input_launcher)
--   setmetatable(p, PhysicalRegion)
--   return p
-- end

-- Wait till physical region is valid, to be called after creating an inline
-- physical region.
-- NOTE: Call from top level task only.
function PhysicalRegion:WaitUntilValid()
  LW.legion_physical_region_wait_until_valid(self.handle)
end




---------------------)(Q#$&Y@)#*$(*&_)@----------------------------------------
--[[         GILBERT added this to make field loading work.                ]]--
--[[           Think of this as a workspace, not final organization        ]]--
-----------------------------------------)(Q#$&Y@)#*$(*&_)@--------------------



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
  if not (params.n_rows or params.dimensions) then
    error('Expects fields list argument', 2)
  end

  -- create the launcher and bind in the fields
  local il = LW.legion_inline_launcher_create_logical_region(
    params.logical_region,  -- legion_logical_region_t handle
    params.privilege,       -- legion_privilege_mode_t
    LW.EXCLUSIVE,           -- legion_coherence_property_t
    params.logical_region,  -- legion_logical_region_t parent
    0,                      -- legion_mapping_tag_id_t region_tag /* = 0 */
    false,                  -- bool verified /* = false*/
    0,                      -- legion_mapper_id_t id /* = 0 */
    0                       -- legion_mapping_tag_id_t launcher_tag /* = 0 */
  )
  local fields = {}
  for i,fid in ipairs(params.fields) do
    LW.legion_inline_launcher_add_field(il, fid, true)
    fields[i] = fid
  end

  -- launch and create the physical region mapping
  local pr = LW.legion_inline_launcher_execute(LE.runtime, LE.ctx, il)

  -- create object to return
  local dimensions = nil
  if params.dimensions then
    dimensions =
      { params.dimensions[1], params.dimensions[2], params.dimensions[3] }
  end
  local launchobj = setmetatable({
    inline_launcher = il,
    physical_region = pr,
    fields          = fields,
    n_rows          = params.n_rows,
    dimensions      = dimensions,
  }, LW.ControlScanner)
  return launchobj
end

function LW.ControlScanner:scanLinear(per_row_callback)
  local accs = {}
  local offs = {}
  local ptrs = {}

  -- initialize access data for the field data
  local subrect = global(LW.legion_rect_1d_t)
  local rect    = global(LW.legion_rect_1d_t)
        rect:get().lo.x[0] = 0
        rect:get().hi.x[0] = self.n_rows-1
  for i,fid in ipairs(self.fields) do
    accs[i] = LW.legion_physical_region_get_field_accessor_generic(
                self.physical_region, fid)
    offs[i] = global(LW.legion_byte_offset_t)
    ptrs[i] = terralib.cast(&int8, LW.legion_accessor_generic_raw_rect_ptr_1d(
                accs[i], rect, subrect:getpointer(), offs[i]:getpointer() ))
  end

  -- Iterate through the rows
  for k=0,self.n_rows-1 do
    per_row_callback(k, unpack(ptrs))
    for i=1,#self.fields do
      ptrs[i] = ptrs[i] + offs[i]:get().offset
    end
  end

  -- clean-up
  for i=1,#self.fields do
    LW.legion_accessor_generic_destroy(accs[i])
  end
end


-- UGGGGH AWFUL MESS BUT IT WORKS AT LEAST
function LW.ControlScanner:scanGrid(per_row_callback)
  local dims = self.dimensions
  local accs = {}
  local offs = {}
  local ptrs = {}

  -- initialize access data for the field data
  local subrect
  local rect
  local get_raw_rect_ptr
  if #dims == 2 then
    subrect            = global(LW.legion_rect_2d_t)
    rect               = global(LW.legion_rect_2d_t)
    rect:get().lo.x[0] = 0
    rect:get().lo.x[1] = 0
    rect:get().hi.x[0] = dims[1]-1
    rect:get().hi.x[1] = dims[2]-1
    get_raw_rect_ptr   = LW.legion_accessor_generic_raw_rect_ptr_2d
  elseif #dims == 3 then
    subrect            = global(LW.legion_rect_3d_t)
    rect               = global(LW.legion_rect_3d_t)
    rect:get().lo.x[0] = 0
    rect:get().lo.x[1] = 0
    rect:get().lo.x[3] = 0
    rect:get().hi.x[0] = dims[1]-1
    rect:get().hi.x[1] = dims[2]-1
    rect:get().hi.x[2] = dims[3]-1
    get_raw_rect_ptr   = LW.legion_accessor_generic_raw_rect_ptr_3d
  else
    error('INTERNAL: impossible branch')
  end

  for i,fid in ipairs(self.fields) do
    accs[i] = LW.legion_physical_region_get_field_accessor_generic(
                self.physical_region, fid)
    local offtemp = global(LW.legion_byte_offset_t[#dims])
    ptrs[i] = terralib.cast(&int8, get_raw_rect_ptr(
                accs[i], rect, subrect:getpointer(),
                terralib.cast(&LW.legion_byte_offset_t, offtemp:getpointer())
              ))
    offs[i] = {
      offtemp:get()[0].offset,
      offtemp:get()[1].offset,
    }
    if #dims == 3 then offs[i][3] = offtemp:get()[2].offset end
  end

  -- kludge the 3d case
  if #dims == 2 then
    dims[3] = 1
    for i=1,#self.fields do offs[i][3] = 0 end -- doesn't matter
  end

  -- Iterate through the rows
  for zi = 0,dims[3]-1 do
    for yi = 0,dims[2]-1 do
      for xi = 0,dims[1]-1 do
        local i = (zi * dims[2] + yi) * dims[1] + xi
        local callptrs = {}
        for fi=1,#self.fields do
          callptrs[fi] = ptrs[fi] +
                         zi * offs[fi][3] + yi * offs[fi][2] + xi * offs[fi][1]
        end
        per_row_callback(i, unpack(callptrs))
  end end end

  -- clean-up
  for i=1,#self.fields do
    LW.legion_accessor_generic_destroy(accs[i])
  end
end

function LW.ControlScanner:close()
  LW.legion_runtime_unmap_region(LE.runtime, LE.ctx, self.physical_region)
  LW.legion_physical_region_destroy(self.physical_region)
  LW.legion_inline_launcher_destroy(self.inline_launcher)
end





