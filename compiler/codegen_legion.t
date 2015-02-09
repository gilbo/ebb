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

function Context:IsGrid()
  return self.bran.relset._typestate.grid
end

function Context:GridDimensions()
  return self.bran.relset._typestate.dimensions
end

function Context:NumRegions()
  return self.bran.arg_layout.num_regions
end

function Context:FieldData(field)
  local findex = self.bran.arg_layout.field_to_index[field]
  local fd     = self:localenv()['_field_ptrs']
  assert(terralib.issymbol(fd))
  terralib.tree.printraw(fd)
  -- field data does not contain region 0, which is used only for iterating
  return `([fd][ findex - 1 ])
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

    -- symbol for physical regions
    local pregions = symbol(&Lc.legion_physical_region_t)
    ctxt:localenv()['_pregions'] = pregions
    -- symbols for iteration and field data
    local iter, field_ptrs
    if ctxt.bran.relset._typestate.grid then
      iter = symbol(domIndex[ctxt.bran.relset._typestate.dimensions])
      ctxt:localenv()[kernel_ast.name] = iter
      field_ptrs = symbol(fieldData[ctxt.bran.relset._typestate.dimensions][ctxt:NumRegions()])
      ctxt:localenv()['_field_ptrs'] = field_ptrs
    else
      error("Unstructured field ptrs/ pregions case not handled")
    end

    -- one iteration code inside Liszt kernel
    local kernel_body = quote
      [ kernel_ast.body:codegen(ctxt) ]
      C.printf("Kernel body incomplete\n")
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
            -- add ptrs to field data and corresponding offsets
            var [field_ptrs]
            for r = 1, [ctxt:NumRegions()+1] do
              C.printf("Adding region %i to field_ptrs\n", r)
              var acc  = Lc.legion_physical_region_get_field_accessor_generic(
                [pregions][r], 1)
              var subrect : Lc.legion_rect_1d_t
              var strides : Lc.legion_byte_offset_t[1]
              var base = [&int8](Lc.legion_accessor_generic_raw_rect_ptr_1d(
                acc, rect, &subrect, strides))
              C.printf("Strides = (%i)\n", strides[0].offset)
              [field_ptrs][r-1] = [ fieldData[1] ] { base, strides }
            end
            -- iterate over region
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
            -- add ptrs to field data and corresponding offsets
            var [field_ptrs]
            for r = 1, [ctxt:NumRegions()+1] do
              C.printf("Adding region %i to field_ptrs\n", r)
              var acc  = Lc.legion_physical_region_get_field_accessor_generic(
                [pregions][r], 1)
              var subrect : Lc.legion_rect_2d_t
              var strides : Lc.legion_byte_offset_t[2]
              var base = [&int8](Lc.legion_accessor_generic_raw_rect_ptr_2d(
                acc, rect, &subrect, strides))
              C.printf("Strides = (%i, %i)\n", strides[0].offset, strides[1].offset)
              [field_ptrs][r-1] = [ fieldData[2] ] { base, strides }
            end
            -- iterate over region
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
            -- add ptrs to field data and corresponding offsets
            var [field_ptrs]
            for r = 1, [ctxt:NumRegions()+1] do
              C.printf("Adding region %i to field_ptrs\n", r)
              var acc  = Lc.legion_physical_region_get_field_accessor_generic(
                [pregions][r], 1)
              var subrect : Lc.legion_rect_3d_t
              var strides : Lc.legion_byte_offset_t[3]
              var base = [&int8](Lc.legion_accessor_generic_raw_rect_ptr_3d(
                acc, rect, &subrect, strides))
              C.printf("Strides = (%i, %i, %i)\n", strides[0].offset, strides[1].offset, strides[2].offset)
              [field_ptrs][r-1] = [ fieldData[3] ] { base, strides }
            end
            -- iterate over region
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
      error("Codegen for unstructured relations unimplemented")
    end

    local k = terra (regions : &Lc.legion_physical_region_t,
                     num_regions : uint32,
                     leg_ctx : Lc.legion_context_t,
                     leg_runtime : Lc.legion_runtime_t)
      -- Plain terra code, needs to be converted to code that uses symbols
      -- and probably moved up to kernel body
      var [pregions] = regions
      var r   = Lc.legion_physical_region_get_logical_region(regions[0])
      var is  = r.index_space
      var dom = Lc.legion_index_space_get_domain(leg_runtime, leg_ctx, is)
      C.printf("Regions = %i\n", [ctxt:NumRegions()])

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

local function IndexToOffset(ctxt, index, strides)
  if ctxt:IsGrid() then
    if ctxt:GridDimensions() == 1 then
      return `([index].x[0] * [strides][0].offset)
    end
    if ctxt:GridDimensions() == 2 then
      return  `(
        [index].x[0] * [strides][0].offset +
        [index].x[1] * [strides][1].offset
      )
    end
    if ctxt:GridDimensions() == 3 then
      return `(
        [index].x[0] * [strides][0].offset +
        [index].x[1] * [strides][1].offset +
        [index].x[2] * [strides][2].offset
      )
    end
  else
    error("Field access to unstructured not implemented")
  end
end

function ast.FieldWrite:codegen (ctxt)
  local phase = ctxt:fieldPhase(self.fieldaccess.field)
  if ctxt:onGPU() then
    error("Field write on GPU not implemented with Legion")
  else
    -- just re-direct to an assignment statement otherwise
    local assign = ast.Assignment:DeriveFrom(self)
    assign.lvalue = self.fieldaccess
    assign.exp    = self.exp
    if self.reduceop then
      error("Reductions not implemented with Legion")
    end
    return assign:codegen(ctxt)
  end
end

function ast.FieldAccess:codegen (ctxt)
  self:pretty_print()
  local index = self.row:codegen(ctxt)
  local fdata = ctxt:FieldData(self.field)
  terralib.tree.printraw(fdata)
  local fttype = self.field:Type().terratype
  local access = quote
    var strides = [fdata].strides
    var ptr = [&fttype]([fdata].ptr + [IndexToOffset(ctxt, index, strides)] )
  in
    @ptr
  end
  return access
end
