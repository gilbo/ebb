local Codegen = {}
package.loaded["compiler.codegen_legion"] = Codegen

local ast = require "compiler.ast"

local C = require 'compiler.c'
local L = require 'compiler.lisztlib'
local Cc = require 'compiler.codegen_common'

-- Legion dependencies
local LW = require "compiler.legionwrap"


-------------------------------------------------------------------------------
--[[                         Codegen Entrypoint                            ]]--
-------------------------------------------------------------------------------

function Codegen.codegen (kernel_ast, bran)
  local env  = terralib.newenvironment(nil)
  local ctxt = Context.new(env, bran)

  if ctxt:onGPU() then
    error("INTERNAL ERROR: Unimplemented GPU codegen with Legion runtime")
  else
    return cpu_codegen_legion(kernel_ast, ctxt)
  end
end


-------------------------------------------------------------------------------
--[[                Function/ types templated on dimension                 ]]--
-------------------------------------------------------------------------------

local domIndex = {}
for dim = 1, 3 do
  domIndex[dim] = struct { x : int[dim] ; }
end

local fieldData = {}
for dim = 1, 3 do
  fieldData[dim] = struct {
    ptr : &int8,
    strides : LW.legion_byte_offset_t[dim] ;
  }
end

local LegionRect = {}
LegionRect[1] = LW.legion_rect_1d_t
LegionRect[2] = LW.legion_rect_2d_t
LegionRect[3] = LW.legion_rect_3d_t

local LegionGetRectFromDom = {}
LegionGetRectFromDom[1] = LW.legion_domain_get_rect_1d
LegionGetRectFromDom[2] = LW.legion_domain_get_rect_2d
LegionGetRectFromDom[3] = LW.legion_domain_get_rect_3d

local LegionRawPtrFromAcc = {}
LegionRawPtrFromAcc[1] = LW.legion_accessor_generic_raw_rect_ptr_1d
LegionRawPtrFromAcc[2] = LW.legion_accessor_generic_raw_rect_ptr_2d
LegionRawPtrFromAcc[3] = LW.legion_accessor_generic_raw_rect_ptr_3d


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

function Context:NumFields()
  return self.bran.arg_layout.num_fields
end

function Context:Region(reg_num)
  return self.bran.arg_layout.regions[reg_num]
end

function Context:NumFieldsForRegion(reg_num)
  return self.bran.arg_layout.regions[reg_num].num_fields
end

function Context:FieldData(field)
  local findex = self.bran.arg_layout.field_num[field]
  local fd     = self:localenv()['_field_ptrs']
  assert(terralib.issymbol(fd))
  -- field data does not contain region 0, which is used only for iterating
  return `([fd][ findex - 1 ])
end

function Context:FieldFromArg(reg_num, field_num)
  return self.bran.arg_layout.regions[reg_num].fields[fnum]
end

function Context:FieldIdFromArg(reg_num, field_num)
  return self.bran.arg_layout.regions[reg_num].fields[field_num].fid
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
    error("INTERNAL ERROR: Unimplemented GPU codegen with Legion runtime")
  else
    return cpu_codegen(kernel_ast, ctxt)
  end

end


----------------------------------------------------------------------------
--[[                         CPU Codegen                                ]]--
----------------------------------------------------------------------------

function cpu_codegen (kernel_ast, ctxt)
  ctxt:enterblock()

    -- symbols for arguments to executable
    local Largs = symbol(LW.TaskArgs)
    ctxt:localenv()['_legion_args'] = Largs

    -- symbols for iteration and field data
    local iter, rect, field_ptrs
    if ctxt:IsGrid() then
      iter = symbol(domIndex[ctxt:GridDimensions()])
      ctxt:localenv()[kernel_ast.name] = iter
      rect = symbol(LegionRect[ctxt:GridDimensions()])
      ctxt:localenv()['_rect'] = rect
      field_ptrs = symbol(fieldData[ctxt:GridDimensions()][ctxt:NumFields()])
      ctxt:localenv()['_field_ptrs'] = field_ptrs
    else
      error("INTERNAL ERROR: Unstructured field ptrs/ pregions case not handled")
    end

    -- code for one iteration inside Liszt kernel
    local kernel_body = quote
      [ kernel_ast.body:codegen(ctxt) ]
      C.printf("Kernel body incomplete\n")
    end

    -- generate iteration code over domain
    local body = quote end

    -- GRIDS
    if ctxt:IsGrid() then
      -- 1 dimensional
      local dim =  ctxt:GridDimensions()

      -- add ptrs to field data and corresponding offsets
      local field_init = quote
        var [field_ptrs]
      end
      local field_init_f = quote end
      local fields_added = 0
      for r = 1, ctxt:NumRegions() do
        local reg = ctxt:Region(r)
        for f = 1, reg.num_fields do
          field_init_f = quote
            [field_init_f]
            do
              var preg = [Largs].regions[r-1]
              var is   = LW.legion_physical_region_get_logical_region(preg).index_space
              var dom  = LW.legion_index_space_get_domain([Largs].lg_runtime, [Largs].lg_ctx, is)
              var rect = [ LegionGetRectFromDom[dim] ](dom)
              do
                var acc  = LW.legion_physical_region_get_field_accessor_generic(
                  preg, [ ctxt:FieldIdFromArg(r, f) ])
                var subrect : LegionRect[dim]
                var strides : LW.legion_byte_offset_t[dim]
                var base = [&int8]([ LegionRawPtrFromAcc[dim] ](
                  acc, rect, &subrect, strides))
                C.printf("In legion task - setup, adding field id %i from region %i\n", [ctxt:FieldIdFromArg(r, f)], r-1)
                [field_ptrs][fields_added] = [ fieldData[dim] ] { base, strides }
              end
            end
          end
          fields_added = fields_added + 1
        end
      end

      -- setup loop bounds
      local setup = quote
        [field_init]
        [field_init_f]
        var r   = LW.legion_physical_region_get_logical_region([Largs].regions[0])
        var is  = r.index_space
        var dom = LW.legion_index_space_get_domain([Largs].lg_runtime, [Largs].lg_ctx, is)
        C.printf(" --- Begin loop ---\n")
        var [rect] = [ LegionGetRectFromDom[dim] ](dom)
      end

      -- loop over domain
      if dim == 1 then
        body = quote
          [setup]
          for i = [rect].lo.x[0], [rect].hi.x[0]+1 do
            -- C.printf("Loop iteration %i\n", i)
            var [iter] = [ domIndex[1] ] { arrayof(int, i) }
            [kernel_body]
          end
        end
      end
      if dim == 2 then
        body = quote
          [setup]
          for i = [rect].lo.x[0], [rect].hi.x[0]+1 do
            for j  = [rect].lo.x[1], [rect].hi.x[1]+1 do
              -- C.printf("Loop iteration %i, %i\n", i, j)
              var [iter] = [ domIndex[2] ] { arrayof(int, i, j) }
              [kernel_body]
            end
          end
        end
      end
      if dim == 3 then
        body = quote
          [setup]
          for i = [rect].lo.x[0], [rect].hi.x[0]+1 do
            for j  = [rect].lo.x[1], [rect].hi.x[1]+1 do
              for k  = [rect].lo.x[2], [rect].hi.x[2]+1 do
                -- C.printf("Loop iteration %i, %i\n", i, j)
                var [iter] = [ domIndex[3] ] { arrayof(int, i, j, k) }
                [kernel_body]
              end
            end
          end
        end
      end

    -- UNSTRUCTURED DOMAINS
    else
      error("INTERNAL ERROR: Codegen for unstructured relations unimplemented")
    end

    local k = terra (leg_args : LW.TaskArgs)
      -- Plain terra code, needs to be converted to code that uses symbols
      -- and probably moved up to kernel body
      C.printf("------------ Executing a legion task -------------\n")
      var [Largs] = leg_args

      -- Add symbol for physical regions &/ or field data
      [body]
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
    error("INTERNAL ERROR: Field access to unstructured not implemented")
  end
end

function ast.FieldWrite:codegen (ctxt)
  local phase = ctxt:fieldPhase(self.fieldaccess.field)
  if ctxt:onGPU() then
    error("INTERNAL ERROR: Field write on GPU not implemented with Legion")
  else
    -- just re-direct to an assignment statement otherwise
    local assign = ast.Assignment:DeriveFrom(self)
    assign.lvalue = self.fieldaccess
    assign.exp    = self.exp
    if self.reduceop then
      assign.reduceop = self.reduceop
    end
    return assign:codegen(ctxt)
  end
end

function ast.FieldAccess:codegen (ctxt)
  local index = self.row:codegen(ctxt)
  local fdata = ctxt:FieldData(self.field)
  local fttype = self.field:Type().terratype
  local access = quote
    var strides = [fdata].strides
    var ptr = [&fttype]([fdata].ptr + [IndexToOffset(ctxt, index, strides)] )
   -- C.printf("Data was %i\n", @ptr)
  in
    @ptr
  end
  return access
end
