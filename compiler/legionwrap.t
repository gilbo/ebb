
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

struct LW.KernelLauncherTemplate {
  Launch : { LW.TaskArgs } -> {};
}

terra LW.NewKernelLauncher( kernel_code : { LW.TaskArgs } -> {} )
  var l : LW.KernelLauncherTemplate
  l.Launch = kernel_code
  return l
end

LW.KernelLauncherSize = terralib.sizeof(LW.KernelLauncherTemplate)

-- Pack kernel launcher into a task argument for Legion.
terra LW.KernelLauncherTemplate:PackToTaskArg()
  var sub_args = LW.legion_task_argument_t {
    args       = [&opaque](self),
    arglen     = LW.KernelLauncherSize
  }
  return sub_args
end


-------------------------------------------------------------------------------
--[[                             Legion Tasks                              ]]--
-------------------------------------------------------------------------------
--[[ A simple task is a task that does not have any return value. A fut_task
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
  assert(arglen == LW.KernelLauncherSize)
  var kernel_launcher =
    [&LW.KernelLauncherTemplate](LW.legion_task_get_args(task))
  kernel_launcher.Launch( LW.TaskArgs {
    task, regions, num_regions, ctx, runtime
  } )
  C.printf("Completed executing simple task\n")
end

LW.TID_SIMPLE = 200

terra LW.fut_task(
  task        : LW.legion_task_t,
  regions     : &LW.legion_physical_region_t,
  num_regions : uint32,
  ctx         : LW.legion_context_t,
  runtime     : LW.legion_runtime_t
) : LW.legion_task_result_t
  C.printf("Executing future task\n")
  var arglen = LW.legion_task_get_arglen(task)
  assert(arglen == LW.KernelLauncherSize)
  var kernel_launcher =
    [&LW.KernelLauncherTemplate](LW.legion_task_get_args(task))
  kernel_launcher.Launch( LW.TaskArgs {
    task, regions, num_regions, ctx, runtime
  } )
  -- TODO: dummy seems likely broken.  It should refer to this task?
  var dummy : int = 9
  var result = LW.legion_task_result_create(&dummy, terralib.sizeof(int))
  C.printf("Completed executing future task task\n")
  return result
end

LW.TID_FUT = 300

-- GLB: Why do we need this table?
LW.TaskTypes = { simple = 'simple', fut = 'fut' }




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
              type = 'unstructured',
              relation  = params.relation,
              field_ids = 0,
              rows_max  = params.rows_max,
              rows_live = 0,
            }
  -- index space
  l.is  = LW.legion_index_space_create(LE.runtime, LE.ctx, l.rows_max)
  l.isa = LW.legion_index_allocator_create(LE.runtime, LE.ctx, l.is)
  -- field space
  l.fs  = LW.legion_field_space_create(LE.runtime, LE.ctx)
  l.fsa = LW.legion_field_allocator_create(LE.runtime, LE.ctx, l.fs)
  -- logical region
  l.handle = LW.legion_logical_region_create(LE.runtime, LE.ctx, l.is, l.fs)
  setmetatable(l, LogicalRegion)
  l:AllocateRows(params.rows_init)
  return l
end

-- Allocate a structured logical region
-- NOTE: Call from top level task only.
function LW.NewGridLogicalRegion(params)
  local l = {
              type = 'grid',
              relation  = params.relation,
              field_ids = 0,
              bounds = params.bounds,
              dimensions = params.dimensions,
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




-------------------------------------------------------------------------------
--[[                         Legion task launcher                          ]]--
-------------------------------------------------------------------------------


-- Creates a map of region requirements, to be used when creating region
-- requirements for task launcher, and when codegen-ing task executable.
function LW.SetUpTaskArgs(params)
  -- TODO: make this into a class
  local field_use = params.bran.kernel.field_use
  params.bran.arg_layout = {}
  local arg_layout = params.bran.arg_layout

  -- field to region number
  arg_layout.field_to_rnum = {}
  local field_to_rnum = arg_layout.field_to_rnum

  -- field location
  arg_layout.field_num = {}
  local field_num = arg_layout.field_num
  local num_fields = 1

  -- region metadata
  arg_layout.regions = {}
  local regions = arg_layout.regions

  -- only one region for now
  regions[1] = {}
  regions[1].relation = params.bran.relset
  regions[1].fields = {}
  regions[1].num_fields = 0

  for field, access in pairs(field_use) do
    local reg_num = 1
    field_num[field] = num_fields
    field_to_rnum[field] = reg_num
    regions[reg_num].num_fields = regions[reg_num].num_fields + 1
    regions[reg_num].fields[regions[reg_num].num_fields] = field
    num_fields = num_fields + 1
  end

  arg_layout.num_regions = 1
  arg_layout.num_fields = num_fields - 1
end


-- Creates a task launcher with task region requirements.
-- Implementation details:
--  * This creates a separate region requirement for each accessed field. We
--  can group multiple fields into one region requirement, based on stencil
--  and access privileges.
--  * A region requirement with no fields is created as region req 0. This is
--  for iterating over the index space. We can instead do book-keeping about
--  which region can be used for performing iteration, or something else?
function LW.CreateTaskLauncher(params)
  local args = params.bran.kernel_launcher:PackToTaskArg()
  -- Simple task that does not return any values
  if params.task_type == LW.TaskTypes.simple then
    -- task launcher
    local task_launcher = LW.legion_task_launcher_create(
                             LW.TID_SIMPLE, args,
                             LW.legion_predicate_true(), 0, 0)
    local field_use = params.bran.kernel.field_use
    local relset = params.bran.relset
    local field_to_rnum = params.bran.arg_layout.field_to_rnum
    local regions = params.bran.arg_layout.regions
    local num_regions = params.bran.arg_layout.num_regions
    local reg_req = {}
    for r = 1, num_regions do
      local region = regions[r]
      local rel = region.relation
      reg_req[r] =
        LW.legion_task_launcher_add_region_requirement_logical_region(
          task_launcher, rel._logical_region_wrapper.handle,
          LW.READ_WRITE, LW.EXCLUSIVE,
          rel._logical_region_wrapper.handle, 0, false )
      for f = 1, region.num_fields do
        local field = region.fields[f]
        local access = field_use[field]
        local rel = field.owner
        print("In create task launcher, adding field " .. field.fid .. " to region req " .. r)
        LW.legion_task_launcher_add_field(
          task_launcher, reg_req[r], field.fid, true )
      end
    end
    return task_launcher
  elseif params.task_type == LW.TaskTypes.fut then
    error("INTERNAL ERROR: Liszt does not handle tasks with future values yet")
  else
    error("INTERNAL ERROR: Unknown task type")
  end
end

-- Launches Legion task and returns.
function LW.LaunchTask(p, leg_args)
  print("Launching legion task")
   LW.legion_task_launcher_execute(leg_args.runtime, leg_args.ctx,
                                   p.task_launcher)
  print("Launched task")
end







