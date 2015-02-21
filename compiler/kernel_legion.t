local K = {}
package.loaded["compiler.kernel_legion"] = K
local Kc = require "compiler.kernel_common"
local L = require "compiler.lisztlib"
local C = require "compiler.c"

local codegen = require "compiler.codegen_legion"
local LW      = require "compiler.legionwrap"

-------------------------------------------------------------------------------
--[[                            Kernels, Brans                             ]]--
-------------------------------------------------------------------------------

local Bran = Kc.Bran
local Seedbank = Kc.Seedbank
local seedbank_lookup = Kc.seedbank_lookup
local SimpleKernelLauncherTemplate = LW.SimpleKernelLauncherTemplate
local FutureKernelLauncherTemplate = LW.FutureKernelLauncherTemplate


-------------------------------------------------------------------------------
--[[                          Legion environment                           ]]--
-------------------------------------------------------------------------------

local LE = rawget(_G, '_legion_env')
local legion_env = LE.legion_env:get()

-------------------------------------------------------------------------------
--[[                Argument layout (known during codegen)                 ]]--
-------------------------------------------------------------------------------

-- information about fields in a region, privileges etc
-- this is constant across invocations, used only once during codegen.
-- legion task arguments on the other hand will have to be computed every time
-- a kernel is invoked.
local ArgRegion = {}
ArgRegion.__index = ArgRegion
local function NewArgRegion(rel)
  local reg = {
    relation   = rel,
    -- fields in this region
    fields     = {},
    -- number of fields this region contains
    num_fields = 0,
  }
  setmetatable(reg, ArgRegion)
  return reg
end
K.NewArgRegion = NewArgRegion

function ArgRegion:Relation()
  return self.relation
end

function ArgRegion:Fields()
  return self.fields
end

function ArgRegion:NumFields()
  return self.num_fields
end

function ArgRegion:AddField(field)
  self.num_fields = self.num_fields + 1
  self.fields[self.num_fields] = field
end

-- information about regions passed by Legion, number of fields etc
local ArgLayout = {}
ArgLayout.__index = ArgLayout
K.ArgLayout = ArgLayout
local function NewArgLayout()
  local arg = {
    -- regions
    num_regions = 0,
    region_idx  = {},
    regions     = {}, -- there may not be a one-to-many map from relations to regions
    -- fields
    num_fields  = 0,
    field_idx   = {},
    -- gloabls (futures)
    num_globals = 0,
    global_idx  = {},
    globals     = {},
    global_red  = nil
  }
  setmetatable(arg, ArgLayout)
  return arg
end
K.NewArgLayout = NewArgLayout

function ArgLayout:NumRegions()
  return self.num_regions
end

function ArgLayout:Regions()
  return self.regions
end

function ArgLayout:Globals()
  return self.globals
end

function ArgLayout:GlobalToReduce()
  return self.global_red
end

function ArgLayout:NumFields()
  return self.num_fields
end

function ArgLayout:NumGlobals()
  return self.num_globals
end

function ArgLayout:AddRegion(reg)
  self.num_regions = self.num_regions + 1
  self.region_idx[reg] = self.num_regions
  self.regions[self.num_regions] = reg
end

-- Check if region corresponding to a relation is present.
-- If not, create one, based on params.
-- This creates one region for every relation right now.
function ArgLayout:GetRegion(relation)
  for _, reg in pairs(self:Regions()) do
    if reg:Relation() == relation then
      return reg
    end
  end
  local reg = NewArgRegion(relation)
  self:AddRegion(reg)
  return reg
end

function ArgLayout:AddFieldToRegion(field, reg)
  self.num_fields = self.num_fields + 1
  self.field_idx[field] = self.num_fields
  reg:AddField(field)
end

function ArgLayout:AddGlobal(global, phase)
  self.num_globals = self.num_globals + 1
  self.global_idx[global] = self.num_globals
  self.globals[self.num_globals] = global
  if phase:isReduce() then
    self.global_red = global
  end
end

function ArgLayout:RegIdx(reg)
  return self.region_idx[reg]
end

function ArgLayout:FieldIdx(field)
  return self.field_idx[field]
end

function ArgLayout:GlobalIdx(global)
  return self.global_idx[global]
end


-------------------------------------------------------------------------------
--[[                                Kernel                                 ]]--
-------------------------------------------------------------------------------
--[[ When a Liszt kernel is invoked, Liszt runtime creates a function to
--   execute application code, wraps it into a kernel launcher, and launches a
--   Legion task with a pointer to this executable. A Legion task unwraps the
--   kernel launcher from its local arguments, and invokes the function (which
--   is the entry point for Liszt generated code). The generated code uses
--   physical regions, instead of any direct data pointers.
--   SetupArgLayout computes region-field layout, generate() (codegen) uses
--   this computed layout, and CreateTaskLauncher uses the layout as well as
--   the generated executable. The launcher is stored in bran for launching
--   tasks over the same relation in future, without having to compute the
--   launcher and region requirements again.
--]]--

-- Setup and Launch Legion task when a Liszt kernel is invoked from an
-- application.
L.LKernel.__call  = function (kobj, relset)
  if not (relset and (L.is_relation(relset) or L.is_subset(relset)))
  then
      error("A kernel must be called on a relation or subset.", 2)
  end

  local proc = L.default_processor

  -- retreive the correct bran or create a new one
  local bran = seedbank_lookup({ kernel = kobj,
                                 relset = relset,
                                 proc   = proc
                               })

  -- GENERATE CODE (USING BRAN?) FOR LEGION TASK
  if not bran.kernel_launcher then
    bran.relset = relset
    bran.kernel = kobj
    bran.location = proc
    bran:SetUpArgLayout()
    bran:generate()
    bran:CreateTaskLauncher()
  end

  -- Launch task
  bran:LaunchTask({ ctx = legion_env.ctx, runtime = legion_env.runtime })

end


-------------------------------------------------------------------------------
--[[                                 Brans                                 ]]--
-------------------------------------------------------------------------------

-- Computes a layout for region requirements and futures.
-- implementation details:
-- This computes a trivial region requirement with just one region per
-- relation, and one future for very global being used.
function Bran:SetUpArgLayout()

  local field_use  = self.kernel.field_use
  local global_use = self.kernel.global_use

  -- arg layout
  self.arg_layout = NewArgLayout()
  local arg_layout = self.arg_layout

  -- regions
  for field, access in pairs(field_use) do
    local reg = arg_layout:GetRegion(field.owner)
    arg_layout:AddFieldToRegion(field, reg)
  end

  -- globals/ futures
  for global, access in pairs(global_use) do
    arg_layout:AddGlobal(global, access)
  end

end

function Bran:generate()
  -- GENERATE CODE FOR LEGION TASK
  -- signature for legion code includes
  --   * physical regions, a map from accessed fields to physical regions
  --   * globals?
  --   * pointer to legion ctx - may not need this
  --   * pointer to legion runtime - may not need this
  local bran      = self
  local kernel    = bran.kernel
  local typed_ast = bran.kernel.typed_ast

  if L.is_relation(bran.relset) then
    bran.relation = bran.relset
  else
    error("Subsets not implemented with Legion runtime")
  end

  -- type checking the kernel signature against the invocation
  if typed_ast.relation ~= bran.relation then
    error('Kernels may only be called on a relation they were typed with', 3)
  end

  -- compile an executable (placeholder right now)
  bran.executable = codegen.codegen(typed_ast, bran)
  -- create a liszt kernel launcher, to invoke from legion task (a convenience
  -- data structure to pass a function type around, since there are no typedefs)
  -- TODO: Cannot pass a terra function to another terra function.
  -- Doing so throws error "cannot convert 'table' to 'bool (*)()'".
  -- Should fix this, and remove the wrapper terra function defined below.
  if bran:LegionTaskType() == LW.TaskTypes.simple then
    local terra NewSimpleKernelLauncher()
      return LW.NewSimpleKernelLauncher(bran.executable)
    end
    bran.kernel_launcher = NewSimpleKernelLauncher()
  else
    local terra NewFutureKernelLauncher()
      return LW.NewFutureKernelLauncher(bran.executable)
    end
    bran.kernel_launcher = NewFutureKernelLauncher()
  end
end

function Bran:LegionTaskType()
  assert(self.arg_layout)
  if self.arg_layout.global_red then
    return LW.TaskTypes.future
  else
    return LW.TaskTypes.simple
  end
end

-- Creates a task launcher with task region requirements.
-- Implementation details:
-- * Creates a separate region requirement for every region in arg_layout. 
-- * Adds futures corresponding to globals to launcher.
-- * NOTE: This is not combined with SetUpArgLayout because of a needed codegen
--   phase in between : codegen can happen only after SetUpArgLayout, and the
--   launcher can be created only after executable from codegen is available.
function Bran:CreateTaskLauncher()

  local args = self.kernel_launcher:PackToTaskArg()

  local task_launcher
  -- Simple task that does not return any values
  if self:LegionTaskType() == LW.TaskTypes.simple then
    task_launcher = LW.legion_task_launcher_create(
                       LW.TID_SIMPLE, args,
                       LW.legion_predicate_true(), 0, 0)
  -- Tasks with futures, for handling global reductions
  else
    task_launcher = LW.legion_task_launcher_create(
                       LW.TID_FUTURE, args,
                       LW.legion_predicate_true(), 0, 0)
  end

  local arg_layout = self.arg_layout

  -- region requirements
  local regions = arg_layout:Regions()
  for _, region in ipairs(regions) do
    local r = arg_layout:RegIdx(region)
    local rel = region:Relation()
    -- Just use READ_WRITE and EXCLUSIVE for now.
    -- Will need to update this when doing partitions.
    local reg_req =
      LW.legion_task_launcher_add_region_requirement_logical_region(
        task_launcher, rel._logical_region_wrapper.handle,
        LW.READ_WRITE, LW.EXCLUSIVE,
        rel._logical_region_wrapper.handle, 0, false )
    -- add all fields that should belong to this region req (computed in
    -- SetupArgLayout)
    for _, field in ipairs(region:Fields()) do
      local f = arg_layout:FieldIdx(field, region)
      local rel = field.owner
      print("In create task launcher, adding field " .. field.fid .. " to region req " .. r)
      LW.legion_task_launcher_add_field(
        task_launcher, reg_req, field.fid, true )
    end
  end

  -- futures
  local globals = arg_layout:Globals()
  for _, global in ipairs(globals) do
    LW.legion_task_launcher_add_future(task_launcher, global.data)
  end

  self.task_launcher = task_launcher
end

-- Launches Legion task and returns.
function Bran:LaunchTask(leg_args)
  print("Launching legion task")
  if self:LegionTaskType() == LW.TaskTypes.simple then
    LW.legion_task_launcher_execute(leg_args.runtime, leg_args.ctx,
                                    self.task_launcher)
  else
    local global = self.arg_layout:GlobalToReduce()
    local future = LW.legion_task_launcher_execute(leg_args.runtime,
                                                   leg_args.ctx,
                                                   self.task_launcher)
    local res = LW.legion_future_get_result(future)
    -- Wait till value is available. We can remove this once apply and
    -- fold operations are implemented using legion API, and we figure out
    -- how to safely delete old future : Is it safe to call DestroyFuture
    -- immediately after launching the tasks that use the future?
    -- TODO: We must apply this return value to old value - necessary for
    -- multiple partitions. Work around right now applies reduction in the
    -- task (Liszt kernel) itself, so we can simply replace the old future.
    global.data = LW.legion_future_from_buffer(leg_args.runtime,
                                               res.value, res.value_size)
    LW.legion_task_result_destroy(res)
  end
  print("Launched task")
end
