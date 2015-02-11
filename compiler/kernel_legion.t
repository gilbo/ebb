local K = {}
package.loaded["compiler.kernel_legion"] = K
local Kc = require "compiler.kernel_common"
local L = require "compiler.lisztlib"
local C = require "compiler.c"

local Tc = require "compiler.typedefs"
require "legionlib"
local Lc = terralib.includecstring([[
#include "legion_c.h"
]])

local codegen = require "compiler.codegen_legion"
local Ld      = require "compiler.legion_data"
local Lt      = require "compiler.legion_tasks"
local Tt      = require "compiler.legion_task_types"

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
    Lt.SetUpArgLayout( { bran = bran} )
    bran:generate()
    bran.task_launcher = Lt.CreateTaskLauncher(
                            {
                              task_type        = Tt.TaskTypes.simple,
                              bran             = bran,
                            } )
  end

  -- Launch task
  Lt.LaunchTask( { task_launcher = bran.task_launcher },
                 { ctx = ctx, runtime = runtime } )

end

-- LEGION "KERNEL" TASK
-- build signature
-- run executable


-------------------------------------------------------------------------------
--[[                                 Brans                                 ]]--
-------------------------------------------------------------------------------

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
  local terra NewKernelLauncher()
    return Tt.NewKernelLauncher(bran.executable)
  end
  bran.kernel_launcher = NewKernelLauncher()
end
