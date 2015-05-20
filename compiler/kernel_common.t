local K = {}
package.loaded["compiler.kernel_common"] = K

local L = require "compiler.lisztlib"
local specialization = require "compiler.specialization"
local semant         = require "compiler.semant"
local phase          = require "compiler.phase"

-- Use the following to produce
-- deterministic order of table entries
-- From the Lua Documentation
function pairs_sorted(tbl, compare)
  local arr = {}
  for k in pairs(tbl) do table.insert(arr, k) end
  table.sort(arr, compare)

  local i = 0
  local iter = function() -- iterator
    i = i + 1
    if arr[i] == nil then return nil
    else return arr[i], tbl[arr[i]] end
  end
  return iter
end


-------------------------------------------------------------------------------
--[[ Kernels, Brans, Signatures                                            ]]--
-------------------------------------------------------------------------------
--[[

We use a Kernel as the primary unit of computation.
  For internal use, we define the related concept of a Bran

((
etymology:
  a Bran is the outer part of a kernel, encasing the germ and endosperm
))

A Bran -- a Lua table
          It provides metadata about a particular kernel specialization.
          e.g. one bran for each (kernel, runtime, subset) tuple
          Examples entries:
            - specialization params: (relation, subset)
            - the argument layout
            - executable function
            - field/phase signature

Each Kernel may have many Brans, each a compile-time specialization
Each Bran may have a different binding of argument values for each execution

]]--

local Bran = {}
Bran.__index = Bran
K.Bran = Bran

function Bran.New()
  return setmetatable({}, Bran)
end

-- Seedbank is a cache of brans
local Seedbank = {}

function K.seedbank_lookup(sig)
  local str_sig = ''
  for k,v in pairs_sorted(sig) do
    str_sig = str_sig .. k .. '=' .. tostring(v) .. ';'
  end
  local bran = Seedbank[str_sig]
  if not bran then
    bran = Bran.New()
    for k,v in pairs(sig) do bran[k] = v end
    Seedbank[str_sig] = bran
  end
  return bran
end


-------------------------------------------------------------------------------
--[[ Kernels                                                               ]]--
-------------------------------------------------------------------------------

function L.NewKernel(kernel_ast, env)
    local new_kernel = setmetatable({}, L.LKernel)

    -- All declaration time processing here
    local specialized    = specialization.specialize(env, kernel_ast)
    new_kernel.typed_ast = semant.check(env, specialized)

    local phase_results   = phase.phasePass(new_kernel.typed_ast)
    new_kernel.field_use  = phase_results.field_use
    new_kernel.global_use = phase_results.global_use
    new_kernel.inserts    = phase_results.inserts
    new_kernel.deletes    = phase_results.deletes

    return new_kernel
end
