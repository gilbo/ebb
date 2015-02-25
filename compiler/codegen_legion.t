local Codegen = {}
package.loaded["compiler.codegen_legion"] = Codegen

local ast = require "compiler.ast"

local C = require 'compiler.c'
local L = require 'compiler.lisztlib'
local Cc = require 'compiler.codegen_common'

-- Legion dependencies
local LW = require "compiler.legionwrap"


-------------------------------------------------------------------------------
--[[                Function/ types templated on dimension                 ]]--
-------------------------------------------------------------------------------

local iterKey = {}
for dim = 1, 3 do
  iterKey[dim] = struct { a : uint64[dim] ; }
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

function Context:Dimensions()
  return self.bran.relset:nDims()
end

function Context:NumRegions()
  return self.bran.arg_layout:NumRegions()
end

function Context:NumFields()
  return self.bran.arg_layout:NumFields()
end

function Context:NumGlobals()
  return self.bran.arg_layout:NumGlobals()
end

function Context:Regions()
  return self.bran.arg_layout:Regions()
end

function Context:GetRegion(relation)
  return self.bran.arg_layout:GetRegion(relation)
end

function Context:Fields(reg)
  return reg:Fields()
end

function Context:Globals()
  return self.bran.arg_layout:Globals()
end

function Context:GlobalToReduce()
  return self.bran.arg_layout:GlobalToReduce()
end

function Context:FieldData(field)
  local fidx   = self.bran.arg_layout:FieldIdx(field)
  local fd     = self:localenv()['_field_ptrs']
  assert(terralib.issymbol(fd))
  return `([fd][ fidx - 1 ])
end

function Context:GlobalData(global)
  local gidx   = self.bran.arg_layout:GlobalIdx(global)
  local gd     = self:localenv()['_global_ptrs']
  assert(terralib.issymbol(gd))
  return `([&global.type:terraType()](([gd][ gidx - 1 ]).value))
end

function Context:RegIdx(reg)
  return self.bran.arg_layout:RegIdx(reg)
end

function Context:FieldIdx(field, reg)
  return self.bran.arg_layout:FieldIdx(field, reg)
end

function Context:GlobalIdx(global)
  return self.bran.arg_layout:GlobalIdx(global)
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
--[[ Codegen uses arg_layout to get base pointers for every region-field pair.
--   This computation is in the setup phase, before the loop. Storing these 
--   field pointers (along with their offsets) makes field accesses easier
--   (also potentially avoiding repeated function calls to get the base pointer
--   and offsets).
--]]--

function cpu_codegen (kernel_ast, ctxt)
  ctxt:enterblock()

    -- symbols for arguments to executable
    local Largs = symbol(LW.TaskArgs)
    ctxt:localenv()['_legion_args'] = Largs

    -- symbols for iteration and global/ field data
    local iter, rect, field_ptrs, global_ptrs
    iter = symbol(iterKey[ctxt:Dimensions()])
    ctxt:localenv()[kernel_ast.name] = iter
    rect = symbol(LegionRect[ctxt:Dimensions()])
    ctxt:localenv()['_rect'] = rect
    field_ptrs = symbol(fieldData[ctxt:Dimensions()][ctxt:NumFields()])
    ctxt:localenv()['_field_ptrs'] = field_ptrs
    global_ptrs = symbol(LW.legion_task_result_t[ctxt:NumGlobals()])
    ctxt:localenv()['_global_ptrs'] = global_ptrs

    -- code for one iteration inside Liszt kernel
    local kernel_body = quote
      [ kernel_ast.body:codegen(ctxt) ]
    end

    -- generate iteration code over domain
    local body = quote end

    assert(ctxt:NumRegions() > 0, "Liszt kernel" .. tostring(kernel_ast.id) ..
          " should have at least 1 physical region")

    local dim =  ctxt:Dimensions()

    -- add ptrs to field data and corresponding offsets
    local field_init = quote
      var [field_ptrs]
    end
    for _, reg in ipairs(ctxt:Regions()) do
      local r = ctxt:RegIdx(reg)
      for _, field in ipairs(ctxt:Fields(reg)) do
        local f = ctxt:FieldIdx(field, reg)
        field_init = quote
          [field_init]
          do
            var preg = [Largs].regions[r-1]
            var is   = LW.legion_physical_region_get_logical_region(preg).index_space
            var dom  = LW.legion_index_space_get_domain([Largs].lg_runtime, [Largs].lg_ctx, is)
            var rect = [ LegionGetRectFromDom[dim] ](dom)
            do
              var acc =
                LW.legion_physical_region_get_field_accessor_generic(
                  preg, [field.fid] )
              var subrect : LegionRect[dim]
              var strides : LW.legion_byte_offset_t[dim]
              var base = [&int8]([ LegionRawPtrFromAcc[dim] ](
                acc, rect, &subrect, strides))
              -- C.printf("In legion task - setup, adding field id %i from region %i\n", [ctxt:FieldIdx(field, reg)], r-1)
              [field_ptrs][f-1] = [ fieldData[dim] ] { base, strides }
            end
          end
        end
      end
    end

    -- Read in global data from futures: assumption that the caller gets
    -- ownership of returned legion_result_t. Need to do a deep-copy (copy
    -- value from result) otherwise.
    local global_init = quote
      var [global_ptrs]
    end
    for _, global in ipairs(ctxt:Globals()) do
      local g = ctxt:GlobalIdx(global)
      global_init = quote
        [global_init]
        do
          var fut = LW.legion_task_get_future([Largs].task, g-1)
          [global_ptrs][g-1] = LW.legion_future_get_result(fut)
        end
      end
    end

    -- Return reduced task result and destroy other future task results
    local cleanup_and_ret = quote end
    local global_to_reduce = ctxt:GlobalToReduce()
    for _, global in ipairs(ctxt:Globals()) do
      if global ~= global_to_reduce then
        local g = ctxt:GlobalIdx(global)
        cleanup_and_ret = quote
          [cleanup_and_ret]
          do
            LW.legion_task_result_destroy([global_ptrs][g-1])
          end
        end
      end
    end
    local gred = ctxt:GlobalIdx(global_to_reduce)
    if gred then
      cleanup_and_ret = quote
        [cleanup_and_ret]
        return [global_ptrs][ gred-1 ]
      end
    end

    -- setup loop bounds
    local it_idx = (ctxt:RegIdx(ctxt:GetRegion(ctxt.bran.relset)) - 1)
    local setup = quote
      [field_init]
      [global_init]
      var r   = LW.legion_physical_region_get_logical_region([Largs].regions[it_idx])
      var is  = r.index_space
      var dom = LW.legion_index_space_get_domain([Largs].lg_runtime, [Largs].lg_ctx, is)
      var [rect] = [ LegionGetRectFromDom[dim] ](dom)
    end

    -- loop over domain
    if dim == 1 then
      body = quote
        for i = [rect].lo.x[0], [rect].hi.x[0]+1 do
          var [iter] = [ iterKey[1] ] { array(uint64(i)) }
          [kernel_body]
        end
      end
    end
    if dim == 2 then
      body = quote
        for i = [rect].lo.x[0], [rect].hi.x[0]+1 do
          for j  = [rect].lo.x[1], [rect].hi.x[1]+1 do
            var [iter] = [ iterKey[2] ] { array(uint64(i), uint64(j)) }
            [kernel_body]
          end
        end
      end
    end
    if dim == 3 then
      body = quote
        for i = [rect].lo.x[0], [rect].hi.x[0]+1 do
          for j  = [rect].lo.x[1], [rect].hi.x[1]+1 do
            for k  = [rect].lo.x[2], [rect].hi.x[2]+1 do
              var [iter] = [ iterKey[3] ] { array(uint64(i), uint64(j), uint64(k)) }
              [kernel_body]
            end
          end
        end
      end
    end

    -- assemble everything
    body = quote
      [setup]
      [body]
      [cleanup_and_ret]
    end

    local k = terra (leg_args : LW.TaskArgs)
      -- Plain terra code, needs to be converted to code that uses symbols
      -- and probably moved up to kernel body
      var [Largs] = leg_args
      [body]
    end

  ctxt:leaveblock()

  k:setname(kernel_ast.id)
  return k
end


----------------------------------------------------------------------------
--[[          Codegen Pass Cases involving data access                  ]]--
----------------------------------------------------------------------------

function ast.Global:codegen(ctxt)
  local dataptr = ctxt:GlobalData(self.global)
  return `@dataptr
end

function ast.GlobalReduce:codegen(ctxt)
  if ctxt:onGPU() then
    error("INTERNAL ERROR: reductions on gpu not implemented")
  else
    local assign = ast.Assignment:DeriveFrom(self)
    assign.lvalue = self.global
    assign.exp    = self.exp
    assign.reduceop = self.reduceop
    return assign:codegen(ctxt)
  end
end

local function IndexToOffset(ctxt, index, strides)
  if ctxt:Dimensions() == 1 then
    return `([index].a[0] * [strides][0].offset)
  end
  if ctxt:Dimensions() == 2 then
    return  `(
      [index].a[0] * [strides][0].offset +
      [index].a[1] * [strides][1].offset
    )
  end
  if ctxt:Dimensions() == 3 then
    return `(
      [index].a[0] * [strides][0].offset +
      [index].a[1] * [strides][1].offset +
      [index].a[2] * [strides][2].offset
    )
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
  local index = self.key:codegen(ctxt)
  local fdata = ctxt:FieldData(self.field)
  local fttype = self.field:Type().terratype
  local access = quote
    var strides = [fdata].strides
    var ptr = [&fttype]([fdata].ptr + [IndexToOffset(ctxt, index, strides)] )
  in
    @ptr
  end
  return access
end
