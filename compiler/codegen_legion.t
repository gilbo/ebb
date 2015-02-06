local Codegen = {}
package.loaded["compiler.codegen_legion"] = Codegen

local ast = require "compiler.ast"

local C = require 'compiler.c'
local L = require 'compiler.lisztlib'
local Cc = require 'compiler.codegen_common'

-- Legion dependencies
require "compiler.legion_data"
require "legionlib"
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

    local kernel_body = quote
      C.printf("Kernel executable unimplemented\n")
    end

    -- special iteration logic for subset-mapped kernels
    if ctxt.bran.subset then
      C.printf("This is a subset??")
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
                   leg_ctx : Lc.legion_context_t,
                   leg_runtime : Lc.legion_runtime_t)
    -- Plain terra code, needs to be converted to code that uses symbols
    -- and probably moved up to kernel body
    var r = Lc.legion_physical_region_get_logical_region(regions[0])
    var is = r.index_space
    var d = Lc.legion_index_space_get_domain(leg_runtime, leg_ctx, is)
    -- IF STRUCTURED
    -- Assuming rect == subrect
    if [ ctxt.bran.relset._typestate.grid ] then
      if [ ctxt.bran.relset._typestate.dimensions == 1 ] then
        var rect = Lc.legion_domain_get_rect_1d(d)
        for i = rect.lo.x[0], rect.hi.x[0] do
          [kernel_body]
        end
      end
      if [ ctxt.bran.relset._typestate.dimensions == 2 ] then
        var rect = Lc.legion_domain_get_rect_2d(d)
        for i = rect.lo.x[0], rect.hi.x[0] do
          for j = rect.lo.x[1], rect.hi.x[1] do
            [kernel_body]
          end
        end
      end
      if [ ctxt.bran.relset._typestate.dimensions == 3 ] then
        var rect = Lc.legion_domain_get_rect_3d(d)
        for i = rect.lo.x[0], rect.hi.x[0]+1 do
          for j = rect.lo.x[1], rect.hi.x[1]+1 do
            for k = rect.lo.x[2], rect.hi.x[2]+1 do
              [kernel_body]
            end
          end
        end
      end
    -- ELSE
    else
        C.printf("Codegen for unstructured relations unimplemented\n")
    end
  end
  k:setname(kernel_ast.id)
  return k
end
