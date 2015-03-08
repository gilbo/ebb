local K = {}
package.loaded["compiler.kernel_common"] = K

local L = require "compiler.lisztlib"
local specialization = require "compiler.specialization"
local semant         = require "compiler.semant"
local phase          = require "compiler.phase"
local AST            = require "compiler.ast"

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
    new_kernel.specialized_ast = specialization.specialize(env, kernel_ast)

    --new_kernel:TypeCheck()

    return new_kernel
end

function L.NewKernelFromFunction(user_func, relation)
  local new_kernel = setmetatable({}, L.LKernel)

  --user_func.ast:pretty_print()
  local func_ast  = user_func.ast:alpha_rename() -- ensure this is a safe copy
  --func_ast:pretty_print()
  local ast_root  = AST.LisztKernel:DeriveFrom(func_ast)
  ast_root.id     = func_ast.id
  ast_root.name   = func_ast.params[1]
  ast_root.body   = func_ast.body

  ast_root.set    = AST.LuaObject:DeriveFrom(func_ast)
  ast_root.set.node_type = L.internal(relation)

  new_kernel.specialized_ast = ast_root

  return new_kernel
end

function L.LKernel:TypeCheck()
  self.typed_ast = semant.check(self.specialized_ast)

  -- TODO: throw an error if more than one global is being reduced

  local phase_results   = phase.phasePass(self.typed_ast)
  self.field_use  = phase_results.field_use
  self.global_use = phase_results.global_use
  self.inserts    = phase_results.inserts
  self.deletes    = phase_results.deletes
end



