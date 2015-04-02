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

local fieldData = {}
for dim = 1, 3 do
  fieldData[dim] = struct {
    ptr : &int8,
    strides : LW.legion_byte_offset_t[dim]
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
  return self.bran.relation:nDims()
end

function Context:NumRegions()
  return self.bran.arg_layout:NumRegions()
end

function Context:NumFields(dim)
  return self.bran.arg_layout:NumFields(dim)
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
  local rdim   = field:Relation():nDims()
  local fd     = self:localenv()['_field_ptrs_'..tostring(rdim)]
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

-- 1/ 2/ 3 dimensional iteration
local function terraIterNd(ndims, rect, func)
  local atyp = L.addr_terra_types[ndims]
  local addr = symbol(atyp)
  local iters = {}
  for d=1,ndims do iters[d] = symbol(uint64) end
  local loop = quote
    var [addr] = [atyp]({ a = array( iters ) })
    [func(addr)]
  end
  for d=1,ndims do
    loop = quote
      for [iters[d]] = [rect].lo.x[d-1], [rect].hi.x[d-1]+1 do
        [loop]
      end
    end
  end
  return loop
end

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

    local dim = ctxt:Dimensions()
    -- symbols for iteration and global/ field data
    local iter, rect, field_ptrs, global_ptrs
    iter = symbol(L.addr_terra_types[dim])
    ctxt:localenv()[kernel_ast.name] = iter
    rect = symbol(LegionRect[dim])
    ctxt:localenv()['_rect'] = rect
    field_ptrs = {}
    for d = 1, 3 do
      field_ptrs[d] = symbol(fieldData[d][ctxt:NumFields(d)])
      ctxt:localenv()['_field_ptrs_'..tostring(d)] = field_ptrs[d]
    end
    global_ptrs = symbol(LW.legion_task_result_t[ctxt:NumGlobals()])
    ctxt:localenv()['_global_ptrs'] = global_ptrs

    -- code for one iteration inside Liszt kernel
    local kernel_body = quote
      [ kernel_ast.body:codegen(ctxt) ]
    end

    assert(ctxt:NumRegions() > 0, "Liszt kernel" .. tostring(kernel_ast.id) ..
          " should have at least 1 physical region")

    -- add ptrs to field data and corresponding offsets
    local field_init = quote end
    for d = 1, 3 do
      field_init = quote 
        [field_init]
        var [field_ptrs[d]]
      end
    end
    for _, reg in ipairs(ctxt:Regions()) do
      local r = ctxt:RegIdx(reg)
      local rdim = reg:Relation():nDims()
      for _, field in ipairs(ctxt:Fields(reg)) do
        local f = ctxt:FieldIdx(field, reg)
        field_init = quote
          [field_init]
          do
            var preg = [Largs].regions[r-1]
            var is   = LW.legion_physical_region_get_logical_region(preg).index_space
            var dom  = LW.legion_index_space_get_domain([Largs].lg_runtime, [Largs].lg_ctx, is)
            var rect = [ LegionGetRectFromDom[rdim] ](dom)
            do
              var acc =
                LW.legion_physical_region_get_field_accessor_generic(
                  preg, [field.fid] )
              var subrect : LegionRect[rdim]
              var strides : LW.legion_byte_offset_t[rdim]
              var base = [&int8]([ LegionRawPtrFromAcc[rdim] ](
                acc, rect, &subrect, strides))
              -- C.qrintf("In legion task - setup, adding field id %i from region %i\n", [ctxt:FieldIdx(field, reg)], r-1)
              [field_ptrs[rdim]][f-1] = [ fieldData[rdim] ] { base, strides }
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

    -- Return reduced task result and destroy other task results
    -- (corresponding to futures)
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
    local it_idx = (ctxt:RegIdx(ctxt:GetRegion(ctxt.bran.relation)) - 1)
    local setup = quote
      [field_init]
      [global_init]
      var r   = LW.legion_physical_region_get_logical_region([Largs].regions[it_idx])
      var is  = r.index_space
      var dom = LW.legion_index_space_get_domain([Largs].lg_runtime, [Largs].lg_ctx, is)
      var [rect] = [ LegionGetRectFromDom[dim] ](dom)
    end

    -- loop over domain
    local body_loop = quote end
    if ctxt.bran.subset then
      body_loop = terraIterNd(dim, rect, function(param)
        local boolmask_data = ctxt:FieldData(ctxt.bran.subset._boolmask)
        return quote
          var [iter] = param
          var boolmask_stride = [boolmask_data].strides
          var boolmask_ptr = [&bool]([boolmask_data].ptr +
                             [IndexToOffset(ctxt, param, boolmask_stride)])
          if @boolmask_ptr then
            [kernel_body]
          end
        end
      end)
    else
      body_loop = terraIterNd(dim, rect, function(param)
        return quote
          var [iter] = param
          [kernel_body]
        end
      end)
    end

    -- assemble everything
    local body = quote
      [setup]
      [body_loop]
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

function IndexToOffset(ctxt, index, strides)
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
