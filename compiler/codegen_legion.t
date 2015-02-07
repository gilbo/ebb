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
--[[                    Internal types used in codegen                     ]]--
-------------------------------------------------------------------------------

local domIndex = {}
for dim = 1, 3 do
  domIndex[dim] = struct { x : int[dim] ; }
end

local fieldData = {}
for dim = 1, 3 do
  fieldData[dim] = struct {
    ptr : &int8,
    strides : Lc.legion_byte_offset_t[dim] ;
  }
end


-------------------------------------------------------------------------------
--[[                   Context Object for Compiler Pass                    ]]--
-------------------------------------------------------------------------------

local Context = Cc.Context

function Context:NumRegions()
  for k, v in pairs(self.bran) do
    print(tostring(k))
  end
  return self.bran.arg_layout.num_regions
end

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
  kernel_ast:pretty_print()
  ctxt:enterblock()

    -- symbol for iteration (corresponding to row)
    local iter
    if ctxt.bran.relset._typestate.grid then
      iter = symbol(domIndex[ctxt.bran.relset._typestate.dimensions])
      ctxt:localenv()[kernel_ast.name] = iter
    else
      print("Unstructured case not handled")
    end

    -- list of fieldData (pointer to field)
    -- replace with physical regions if fieldData is insufficient
    local field_ptrs = symbol(fieldData[2][ctxt:NumRegions()-1])
    
    -- one iteration code inside Liszt kernel
    local kernel_body = quote
      C.printf("Kernel body unimplemented\n")
    end

    -- generate iteration code over domain
    local function body(dom) end

    -- grids
    if ctxt.bran.relset._typestate.grid then
      -- 1 dimensional
      if ctxt.bran.relset._typestate.dimensions == 1 then
        body = function(dom)
          return quote
            var rect = Lc.legion_domain_get_rect_1d([dom])
            for i = rect.lo.x[0], rect.hi.x[0]+1 do
              var [iter] = [ domIndex[1] ] { arrayof(int, i) }
              -- add a symbol for it to local env
              [kernel_body]
            end
          end
        end
      end
      -- 2 dimensional
      if ctxt.bran.relset._typestate.dimensions == 2 then
        body = function(dom)
          return quote
            var rect = Lc.legion_domain_get_rect_2d(dom)

            --var acc  = Lc.legion_physical_region_get_field_accessor_generic(
            --  regions[0], 1)
            --var subrect : Lc.legion_rect_2d_t
            --var offsets : Lc.legion_byte_offset_t[2]
            --var base = [&int8](Lc.legion_accessor_generic_raw_rect_ptr_2d(
            --  acc, rect, &subrect, offsets))

            for i = rect.lo.x[0], rect.hi.x[0]+1 do
              for j = rect.lo.x[1], rect.hi.x[1]+1 do
                var [iter] = [ domIndex[2] ] { arrayof(int, i, j) }
                -- add a symbol for it to local env
                [kernel_body]
              end
            end
          end
        end
      end
      -- 3 dimensional
      if ctxt.bran.relset._typestate.dimensions == 3 then
        body = function(dom)
          return quote
            var rect = Lc.legion_domain_get_rect_3d(dom)
            for i = rect.lo.x[0], rect.hi.x[0]+1 do
              for j = rect.lo.x[1], rect.hi.x[1]+1 do
                for k = rect.lo.x[2], rect.hi.x[2]+1 do
                  var [iter] = [ domIndex[3] ] { arrayof(int, i, j, k) }
                  -- add a symbol for it to local env
                  [kernel_body]
                end
              end
            end
          end
        end
      end
    -- unstructured domains
    else
      C.printf("Codegen for unstructured relations unimplemented\n")
    end

    local k = terra (regions : &Lc.legion_physical_region_t,
                     num_regions : uint32,
                     leg_ctx : Lc.legion_context_t,
                     leg_runtime : Lc.legion_runtime_t)
      -- Plain terra code, needs to be converted to code that uses symbols
      -- and probably moved up to kernel body
      var r   = Lc.legion_physical_region_get_logical_region(regions[0])
      var is  = r.index_space
      var dom = Lc.legion_index_space_get_domain(leg_runtime, leg_ctx, is)
      var nr = [ctxt:NumRegions()]
      -- Add symbol for physical regions &/ or field data
      [body(dom)]
    end

  ctxt:leaveblock()

  k:setname(kernel_ast.id)
  return k
end


----------------------------------------------------------------------------
--[[          Codegen Pass Cases involving data access                  ]]--
----------------------------------------------------------------------------

function ast.FieldAccess:codegen (ctxt)
  local index = self.row:codegen(ctxt)
  local dataptr = ctxt:DataPtr(self.field)
end
