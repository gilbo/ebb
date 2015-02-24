
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

-------------------------------------------------------------------------------
--[[                          Legion environment                           ]]--
-------------------------------------------------------------------------------

local LE = rawget(_G, '_legion_env')
local struct LegionEnv {
  runtime : LW.legion_runtime_t,
  ctx     : LW.legion_context_t
}
LE.legion_env = global(LegionEnv)
local legion_env = LE.legion_env:get()

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
  local future = LW.legion_future_from_buffer(legion_env.runtime, data,
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
  local l = {
              type      = 'unstructured',
              relation  = params.relation,
              field_ids = 0,
              n_rows    = params.n_rows,
            }
  -- index space
  l.is  = Create1DGridIndexSpace(l.n_rows)
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
  if not params.dimensions then
    error('Expects dimensions argument', 2)
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
  local pr = LW.legion_inline_launcher_execute(legion_env.runtime,
                                               legion_env.ctx, il)

  local launchobj = setmetatable({
    inline_launcher = il,
    physical_region = pr,
    fields          = fields,
    dimensions      = params.dimensions,
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
    subrect            = global(LW.legion_rect_1d_t)
    rect               = global(LW.legion_rect_1d_t)
    get_raw_rect_ptr   = LW.legion_accessor_generic_raw_rect_ptr_1d
  elseif #dims == 2 then
    subrect            = global(LW.legion_rect_2d_t)
    rect               = global(LW.legion_rect_2d_t)
    get_raw_rect_ptr   = LW.legion_accessor_generic_raw_rect_ptr_2d
  elseif #dims == 3 then
    subrect            = global(LW.legion_rect_3d_t)
    rect               = global(LW.legion_rect_3d_t)
    get_raw_rect_ptr   = LW.legion_accessor_generic_raw_rect_ptr_3d
  else
    error('INTERNAL n_dimensions > 3')
  end
  for d=1,#dims do
    rect:get().lo.x[d-1] = 0
    rect:get().hi.x[d-1] = dims[d]-1
  end

  -- initialize field-dependent data for iteration
  for k,fid in ipairs(self.fields) do
    accs[k] = LW.legion_physical_region_get_field_accessor_generic(
                self.physical_region, fid)
    local offtemp = global(LW.legion_byte_offset_t[#dims])
    ptrs[k] = terralib.cast(&int8, get_raw_rect_ptr(
                accs[k], rect, subrect:getpointer(),
                terralib.cast(&LW.legion_byte_offset_t, offtemp:getpointer())
              ))
    offs[k] = {}
    for d=1,#dims do
      offs[k][d] = offtemp:get()[d-1].offset
    end
  end

  -- define what to do when the iteration terminates
  local function close_up()
    for k=1,#self.fields do
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
      for fi=1,#self.fields do 
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
      for fi=1,#self.fields do 
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
      for fi=1,#self.fields do 
        callptrs[fi] = ptrs[fi] +
                       zi*offs[fi][3] + yi*offs[fi][2] + xi*offs[fi][1]
      end
      return {xi,yi,zi}, callptrs
    end
  end
end


function LW.ControlScanner:close()
  LW.legion_runtime_unmap_region(legion_env.runtime,
                                 legion_env.ctx,
                                 self.physical_region)
  LW.legion_physical_region_destroy(self.physical_region)
  LW.legion_inline_launcher_destroy(self.inline_launcher)
end





