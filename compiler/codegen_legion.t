local Codegen = {}
package.loaded["compiler.codegen_legion"] = Codegen

local ast = require "compiler.ast"

local C = terralib.require 'compiler.c'
local L = terralib.require 'compiler.lisztlib'
local Cc = terralib.require 'compiler.codegen_common'

-- Legion dependencies
terralib.require "compiler.legion_data"
terralib.require "legionlib"
local Lc = terralib.includecstring([[
#include "legion_c.h"
]])


-------------------------------------------------------------------------------
--[[                         Codegen Entrypoint                            ]]--
-------------------------------------------------------------------------------

function Codegen.codegen (kernel_ast, bran)
  local env  = terralib.newenvironment(nil)
  local ctxt = Context.new(env, bran)

  if ctxt:onGPU() then
    error("Unimplemented GPU codegen with Legion runtime")
  else
    return cpu_codegen_legion(kernel_ast, ctxt)
  end
end


-------------------------------------------------------------------------------
--[[                   Context Object for Compiler Pass                    ]]--
-------------------------------------------------------------------------------

local Context = Cc.Context

function Context:IndexSpaceIterator()
  -- TODO: 
  -- > Return a symbol for index space iterator corresponding to logical reion
  --   for relation over which kernel is invoked.
end

function Context:FieldAccessor(relation, field)
  -- TODO:
  -- > Return a symbol for accessing field from physical region for given pair
  --   of relation, field.
end


-------------------------------------------------------------------------------
--[[                         Codegen Entrypoint                            ]]--
-------------------------------------------------------------------------------

function Codegen.codegen (kernel_ast, bran)
  local env  = terralib.newenvironment(nil)
  local ctxt = Context.new(env, bran)

  if ctxt:onGPU() then
    error("Unimplemented GPU codegen with Legion runtime")
  else
    return cpu_codegen(kernel_ast, ctxt)
  end

end


----------------------------------------------------------------------------
--[[                         CPU Codegen                                ]]--
----------------------------------------------------------------------------

function cpu_codegen (kernel_ast, ctxt)
  ctxt:enterblock()
    -- declare the symbol for iteration

    -- body of kernel loop
    local body = quote
      -- [kernel_ast.body:codegen(ctxt)]
    end

    -- Placeholder for kernel executable.
    local kernel_body = quote
      C.printf("Kernel executable unimplemented\n")
      C.printf("In codegen_legion.t\n")
      return false
    end

    -- special iteration logic for subset-mapped kernels
    if ctxt.bran.subset then
      error("Unimplemented subsets codegen with Legion runtime")
    else
      -- kernel_body = quote
      --   TODO:
      --   > Define index space iterator using logical region for the relation
      --   > Iterate over index space
      --     [body]
      --   end
    end
  ctxt:leaveblock()

  local k = terra (regions : &Lc.legion_physical_region_t,
                   num_regions : uint32,
                   ctx : Lc.legion_context_t,
                   runtime : Lc.legion_runtime_t)
    [kernel_body]
  end
  k:setname(kernel_ast.id)
  return k
end
