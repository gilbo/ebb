local K = {}
package.loaded["compiler.kernel_legion"] = K
local Kc = terralib.require "compiler.kernel_common"
local L = terralib.require "compiler.lisztlib"
local C = terralib.require "compiler.c"

local Tc = terralib.require "compiler.typedefs"
terralib.require "legionlib-terra"
local Lc = terralib.includecstring([[
#include "legion_c.h"
]])

local codegen = terralib.require "compiler.codegen"
local Ld      = terralib.require "compiler.legion_data"
local Lt      = terralib.require "compiler.legion_tasks"

-------------------------------------------------------------------------------
--[[                            Kernels, Brans                             ]]--
-------------------------------------------------------------------------------

local Bran = Kc.Bran
local Seedbank = Kc.Seedbank
local seedbank_lookup = Kc.seedbank_lookup
local KernelLauncherTemplate = Lt.KernelLauncherTemplate


-------------------------------------------------------------------------------
--[[                          Legion environment                           ]]--
-------------------------------------------------------------------------------

local legion_env = rawget(_G, '_legion_env')
local ctx = legion_env.ctx
local runtime = legion_env.runtime


-------------------------------------------------------------------------------
--[[                                Kernel                                 ]]--
-------------------------------------------------------------------------------
--[[ When a Liszt kernel is invoked, Liszt runtime creates a function to
--   execute application code, wraps it into a kernel launcher, and launches a
--   Legion task with a pointer to this executable. A Legion task unwraps the
--   kernel launcher from its local arguments, and invokes the function (which
--   is the entry point for Liszt generated code). The generated code uses
--   physical regions, instead of any direct data pointers. Liszt runtime
--   creates the region requirements for the field and global uses every time
--   the kernel is invoked, to pass these requirements to the task launcher.
--   (We could cache the region requirements later if that makes sense.)
--]]--

-- Placeholder for kernel executable.
local terra code_unimplemented(regions : &Lc.legion_physical_region_t,
                               num_region : uint32, ctx : Lc.legion_context_t,
                               runtime : Lc.legion_runtime_t)
  C.printf("Kernel executable unimplemented\n")
  return false
end

-- Set up pointer to Liszt generated code to be invoked from a Legion task.
local terra SetUpKernelLauncher()
  var kernel_launcher = Lt.NewKernelLauncher(code_unimplemented)
  return kernel_launcher
end

-- Setup physical regions and arguments for a Legion task, and launch the
-- Legion task.
-- NOTE: call from top level task only.
local function SetUpAndLaunchTask(params)
  local launcher = Lt.CreateTaskLauncher(
                      { task_type        = Lt.TaskTypes.simple,
                         kernel_launcher = params.kernel_launcher } )
  Lt.LaunchTask( { task_launcher = launcher,
                   ctx = ctx,
                   runtime =  runtime } )
end

-- Setup and Launch Legion task when a Liszt kernel is invoked from an
-- application.
L.LKernel.__call  = function (kobj, relset)
    if not (relset and (L.is_relation(relset) or L.is_subset(relset)))
    then
        error("A kernel must be called on a relation or subset.", 2)
    end

    local proc = L.default_processor

    -- GENERATE CODE (USING BRAN?) FOR LEGION TASK
    -- signature for legion code includes
    --   * physical regions, a map from accessed fields to physical regions
    --   * globals?
    --   * pointer to legion ctx - may not need this
    --   * pointer to legion runtime - may not need this
    local kernel_launcher = SetUpKernelLauncher()

    -- PREPARE LEGION TASK ARGUMENTS (physical regions, globals)
    SetUpAndLaunchTask({ kernel_launcher = kernel_launcher })

    -- PREPARE LEGION TASK REGION REQUIREMENTS

    -- LAUNCH LEGION "KERNEL" TASK, WITH A POINTER TO THE GEBERATED EXECUTABLE
    print("Unimplemented kernel call for legion runtime")
end

-- LEGION "KERNEL" TASK
-- build signature
-- run executable


-------------------------------------------------------------------------------
--[[                                 Brans                                 ]]--
-------------------------------------------------------------------------------
