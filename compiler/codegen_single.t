local Codegen = {}
package.loaded["compiler.codegen_single"] = Codegen

local ast = require "compiler.ast"
local T   = require "compiler.types"

local C   = require 'compiler.c'
local L   = require 'compiler.lisztlib'
local G   = require 'compiler.gpu_util'
local Cc  = require 'compiler.codegen_common'

L._INTERNAL_DEV_OUTPUT_PTX = false

--[[--------------------------------------------------------------------]]--
--[[                 Context Object for Compiler Pass                   ]]--
--[[--------------------------------------------------------------------]]--

-- container class for context attributes specific to the GPU runtime: --
local GPUContext = Cc.GPUContext

function GPUContext:tid()
  if not self._tid then self._tid = symbol(uint32) end
  return self._tid
end

function GPUContext:bid()
  if not self._bid then self._bid = symbol(uint32) end
  return self._bid
end

function GPUContext:blockSize()
  return self.block_size
end

-- container class that manages extra state needed to support reductions
-- on GPUs
local ReductionCtx = Cc.ReductionCtx

function ReductionCtx:reduceRequired()
  return self.ctxt.bran:UsesGPUReduce()
end
function ReductionCtx:sharedMemPtr(globl)
  local tid = self.ctxt.gpu:tid()
  local shared_ptr = self.ctxt.bran:getTerraReduceSharedMemPtr(globl)
  return `[shared_ptr][tid]
end

function ReductionCtx:globalSharedIter()
  -- HACK HACK HACK
  local globl_iter = nil
  return function()
    local data
    globl_iter, data = next(self.ctxt.bran.gpu_reductions, globl_iter)
    if globl_iter then return globl_iter, data.sharedmem
                  else return nil end
  end
end

function ReductionCtx:sharedMemSize()
  return self.ctxt.bran:nBytesSharedMem() --.shared_mem_size
end

function ReductionCtx:GlobalScratchPtr(global)
  return self.ctxt.bran:getTerraReduceGlobalMemPtr(
                                self.ctxt:runtimeSignature(), global)
end

-- The Context class manages state common to the GPU and CPU runtimes.  GPU-specific
-- state should be stored in ctxt.gpu and ctxt.reduce
local Context = Cc.Context

function Context:argsType()
  return self.bran:argsType()
end
function Context:FieldPtr(field)
  return self.bran:getTerraFieldPtr(self:runtimeSignature(), field)
end

function Context:GlobalPtr(global)
  return self.bran:getTerraGlobalPtr(self:runtimeSignature(), global)
end

function Context:runtimeSignature()
  if not self.signature_ptr then
    self.signature_ptr = symbol(self:argsType())
  end
  return self.signature_ptr
end
function Context:cpuSignature()
  return self.bran.args:ptr()
end
function Context:isLiveCheck(param_var)
  local ptr = self:FieldPtr(self.bran.relation._is_live_mask)
  return `ptr[param_var.a[0]]
end
function Context:deleteSizeVar()
  local dd = self.bran.delete_data
  if dd then
    return `@[self:GlobalPtr(dd.updated_size)]
  end
end
function Context:getInsertIndex()
  return `[self:runtimeSignature()].insert_write
end
function Context:incrementInsertIndex()
  local insert_index = self:getInsertIndex()
  local counter = self:GlobalPtr(self.bran.insert_data.n_inserted)

  return quote
    insert_index = insert_index + 1
    @counter = @counter + 1
  end
end


--[[--------------------------------------------------------------------]]--
--[[                        Codegen Entrypoint                          ]]--
--[[--------------------------------------------------------------------]]--


function Codegen.codegen (kernel_ast, bran)
  local env  = terralib.newenvironment(nil)
  local ctxt = Context.new(env, bran)

  if ctxt:onGPU() then
    return gpu_codegen(kernel_ast, ctxt)
  else
    return cpu_codegen(kernel_ast, ctxt)
  end

end


--[[--------------------------------------------------------------------]]--
--[[                         CPU Codegen                                ]]--
--[[--------------------------------------------------------------------]]--

function terraIterNd(dims, func)
  local atyp = L.addr_terra_types[#dims]
  local addr = symbol(atyp)
  local iters = {}
  for d=1,#dims do iters[d] = symbol(uint64) end
  local loop = quote
    var [addr] = [atyp]({ a = array( iters ) })
    [func(addr)]
  end
  for drev=1,#dims do
    local d = #dims-drev + 1
    loop = quote for [iters[d]] = 0, [dims[d]] do [loop] end end
  end
  return loop
end

function cpu_codegen (kernel_ast, ctxt)
  ctxt:enterblock()
    -- declare the symbol for iteration
    local param = symbol(L.key(ctxt.bran.relation):terraType())
    ctxt:localenv()[kernel_ast.name] = param

    -- insert a check for the live row mask

    local body  = quote
      if [ctxt:isLiveCheck(param)] then
        [kernel_ast.body:codegen(ctxt)]
      end
    end

    -- by default on CPU just iterate over all the possible rows
    local dims = ctxt:dims()
    if ctxt:isElastic() then dims = { `[ctxt:runtimeSignature()].n_rows } end
    local kernel_body = terraIterNd(dims, function(iter)
      return quote
        var [param] = iter
        [body]
      end
    end)

    -- special iteration logic for subset-mapped kernels
    if ctxt.bran.subset then
      local boolmask = symbol('boolmask')
      local boolloop = terraIterNd(dims, function(iter)
        return quote
          if [boolmask][ [T.linAddrTerraGen(dims)](iter) ] then
            var [param] = iter
            [body]
          end
        end
      end)
      boolloop = quote
        var [boolmask] = [ctxt:runtimeSignature()].boolmask
        [boolloop]
      end

      -- should never execute this stub
      local indexloop = quote assert(false) end
      -- the following code will not typecheck for grids, so don't
      -- try to build the code in that case
      if #ctxt:dims() == 1 then
        local indexsym  = symbol('index')
        local sizesym   = symbol('index_size')
        indexloop = terraIterNd({ sizesym }, function(iter)
          return quote
            var [param] = [indexsym][iter.a[0]]
            [body]
          end
        end)
        indexloop = quote
          var [indexsym] = [ctxt:runtimeSignature()].index
          var [sizesym]  = [ctxt:runtimeSignature()].index_size
          [indexloop]
        end
      end

      kernel_body = quote
        if [ctxt:runtimeSignature()].use_boolmask then
          [boolloop]
        else
          [indexloop]
        end
      end
    end
  ctxt:leaveblock()

  local k = terra (signature_ptr : &ctxt:argsType())
    var [ctxt:runtimeSignature()] = @signature_ptr
    [kernel_body]
  end
  k:setname(kernel_ast.id)
  return k
end


--[[--------------------------------------------------------------------]]--
--[[                         GPU Codegen                                ]]--
--[[--------------------------------------------------------------------]]--

Codegen.reduction_identity = Cc.reduction_identity
Codegen.reduction_binop    = Cc.reduction_binop

function terraGPUId_to_Nd(dims, size, id, func)
  local atyp = L.addr_terra_types[#dims]
  local addr = symbol(atyp)
  local translate
  if #dims == 1 then
    translate = quote var [addr] = [atyp]({ a = array(id) }) end
  elseif #dims == 2 then
    translate = quote
      var xid : uint64 = id % [dims[1]]
      var yid : uint64 = id / [dims[1]]
      var [addr] = [atyp]({ a = array(xid,yid) })
    end
  elseif #dims == 3 then
    translate = quote
      var xid : uint64 = id % [dims[1]]
      var yid : uint64 = (id / [dims[1]]) % [dims[2]]
      var zid : uint64 = id / [dims[1]*dims[2]]
      var [addr] = [atyp]({ a = array(xid,yid,zid) })
    end
  else
    error('INTERNAL: #dims > 3')
  end

  return quote
    if id < size then
      [translate]
      [func(addr)]
    end
  end
end

function gpu_codegen (kernel_ast, ctxt)
  local BLOCK_SIZE   = ctxt.bran.blocksize
  local MAX_GRID_DIM = 65536

  -----------------------------
  --[[ Codegen CUDA kernel ]]--
  -----------------------------
  ctxt:initializeGPUState(BLOCK_SIZE)
  ctxt:enterblock()
    -- declare the symbol for iteration
    local param = symbol(L.key(ctxt.bran.relation):terraType())
    ctxt:localenv()[kernel_ast.name] = param
    local id  = symbol(uint64)

    if ctxt:isElastic() then error("INTERNAL: ELASTIC ON GPU UNSUPPORTED") end
    local dims = ctxt:dims()

    local body = kernel_ast.body:codegen(ctxt)

    if ctxt.bran.subset then
      --body = quote
      --  if [ctxt:runtimeSignature()].use_boolmask then
      --    if id < [ctxt:runtimeSignature()].n_rows and
      --       [ctxt:runtimeSignature()].boolmask[id]
      --    then
      --      var [param] = [L.addr_terra_types[1]]({ a = array(id) })
      --      [body]
      --    end
      --  else
      --    if id < [ctxt:runtimeSignature()].index_size then
      --      var i = [ctxt:runtimeSignature()].index[id]
      --      var [param] = [L.addr_terra_types[1]]({ a = array(i) })
      --      [body]
      --    end
      --  end
      --end

      body = terraGPUId_to_Nd(dims,
      `[ctxt:runtimeSignature()].n_rows, id, function(addr)
        return quote
          -- set param
          var [param]
          if [ctxt:runtimeSignature()].use_boolmask then
            param = addr
          else
            if id < [ctxt:runtimeSignature()].index_size then
              param = [ctxt:runtimeSignature()].index[id]
            end
          end

          -- conditionally execute
          if    not [ctxt:runtimeSignature()].use_boolmask
             or [ctxt:runtimeSignature()].boolmask[id]
          then
            [body]
          end
        end
      end)
    else
      body = terraGPUId_to_Nd(dims,
      `[ctxt:runtimeSignature()].n_rows, id, function(addr)
        return quote
          var [param] = [addr]
          [body]
        end
      end)
    end

    local kernel_body = quote
      var [ctxt.gpu:tid()] = G.thread_id()
      var [ctxt.gpu:bid()] = G.block_id()
      var [id] = [ctxt.gpu:bid()] * BLOCK_SIZE + [ctxt.gpu:tid()]

      -- Initialize shared memory for global reductions
      --  for kernels that require it
      escape if ctxt.reduce:reduceRequired() then emit quote
        [ctxt.bran:GenerateSharedMemInitialization(ctxt.gpu:tid())]
        G.barrier()
      end end end
      
      [body]

      -- reduce block reduction temporaries to one value and copy back to GPU
      -- global memory for kernels that require it
      escape if ctxt.reduce:reduceRequired() then emit quote
        G.barrier()
        [ctxt.bran:GenerateSharedMemReduceTree(ctxt:runtimeSignature(),
                                               ctxt.gpu:tid(),
                                               ctxt.gpu:bid())]
      end end end
    end

  ctxt:leaveblock()

  local cuda_kernel = terra ([ctxt:runtimeSignature()])
    [kernel_body]
  end
  cuda_kernel:setname(kernel_ast.id)
  -- using kernel_ast.id here to set debug name of kernel for tuning/debugging
  
  -- limit 
  local annotations = { {"maxntidx",64}, {"minctasm",6} }
  cuda_kernel = G.kernelwrap(cuda_kernel, L._INTERNAL_DEV_OUTPUT_PTX, annotations)

  --------------------------
  --[[ Codegen launcher ]]--
  --------------------------
  -- signature type will have a use_boolmask field only if it
  -- was generated for a subset kernel
  local launcher = terra (n_blocks : uint, signature_ptr : &ctxt:argsType())
    var [ctxt:runtimeSignature()] = @signature_ptr
    var grid_x : uint,
        grid_y : uint,
        grid_z : uint   = G.get_grid_dimensions(n_blocks, MAX_GRID_DIM)
    var params = terralib.CUDAParams {
      grid_x, grid_y, grid_z,
      BLOCK_SIZE, 1, 1,
      [ctxt.reduce:sharedMemSize()], nil
    }

    cuda_kernel(&params, [ctxt:runtimeSignature()])
    G.sync() -- flush print streams
  end

  return launcher
end









function atomic_gpu_red_exp (op, typ, lvalptr, update)
  local internal_error = 'unsupported reduction, internal error; '..
                         'this should be guarded against in the typechecker'
  if typ == L.float then
    if     op == '+'   then return `G.atomic_add_float(lvalptr,  update)
    elseif op == '-'   then return `G.atomic_add_float(lvalptr, -update)
    elseif op == '*'   then return `G.atomic_mul_float_SLOW(lvalptr, update)
    elseif op == '/'   then return `G.atomic_div_float_SLOW(lvalptr, update)
    elseif op == 'min' then return `G.atomic_min_float_SLOW(lvalptr, update)
    elseif op == 'max' then return `G.atomic_max_float_SLOW(lvalptr, update)
    end

  elseif typ == L.double then
    if     op == '+'   then return `G.atomic_add_double_SLOW(lvalptr,  update)
    elseif op == '-'   then return `G.atomic_add_double_SLOW(lvalptr, -update)
    elseif op == '*'   then return `G.atomic_mul_double_SLOW(lvalptr, update)
    elseif op == '/'   then return `G.atomic_div_double_SLOW(lvalptr, update)
    elseif op == 'min' then return `G.atomic_min_double_SLOW(lvalptr, update)
    elseif op == 'max' then return `G.atomic_max_double_SLOW(lvalptr, update)
    end

  elseif typ == L.int then
    if     op == '+'   then return `G.reduce_add_int32(lvalptr,  update)
    elseif op == '-'   then return `G.reduce_add_int32(lvalptr, -update)
    elseif op == '*'   then return `G.atomic_mul_int32_SLOW(lvalptr, update)
    elseif op == 'max' then return `G.reduce_max_int32(lvalptr, update)
    elseif op == 'min' then return `G.reduce_min_int32(lvalptr, update)
    end

  elseif typ == L.bool then
    if     op == 'and' then return `G.reduce_and_b32(lvalptr, update)
    elseif op == 'or'  then return `G.reduce_or_b32(lvalptr, update)
    end
  end
  error(internal_error)
end


function atomic_gpu_mat_red_exp(op, result_typ, lval, rhe, rhtyp)
  if result_typ:isScalar() then
    return atomic_gpu_red_exp(op, result_typ, `&lval, rhe)
  elseif result_typ:isVector() then

    local N = result_typ.N
    local rhbind, rhcoords = let_vec_binding(rhtyp, N, rhe)

    local v = symbol() -- pointer to vector location of reduction result

    local result = quote end
    for i = 0, N-1 do
      result = quote
        [result]
        [atomic_gpu_red_exp(op, result_typ:baseType(), `v+i, rhcoords[i+1])]
      end
    end
    return quote
      var [v] : &result_typ:terraBaseType() = [&result_typ:terraBaseType()](&[lval])
      [rhbind]
    in
      [result]
    end
  else -- small matrix

    local N = result_typ.Nrow
    local M = result_typ.Ncol
    local rhbind, rhcoords = let_mat_binding(rhtyp, N, M, rhe)

    local m = symbol()

    local result = quote end
    for i = 0, N-1 do
      for j = 0, M-1 do
        result = quote
          [result]
          [atomic_gpu_red_exp(op, result_typ:baseType(), `&([m].d[i][j]), rhcoords[i+1][j+1])]
        end
      end
    end
    return quote
      var [m] : &result_typ:terraType() = [&result_typ:terraType()](&[lval])
      [rhbind]
      in
      [result]
    end
  end
end



--[[--------------------------------------------------------------------]]--
--[[          Codegen Pass Cases involving data access                  ]]--
--[[--------------------------------------------------------------------]]--


function ast.Global:codegen (ctxt)
  local dataptr = ctxt:GlobalPtr(self.global)
  return `@dataptr
end

function ast.Where:codegen(ctxt)
    local key         = self.key:codegen(ctxt)
    local sType       = self.node_type:terraType()
    local keydims     = self.key.node_type.relation:Dims()
    local indexarith  = T.linAddrTerraGen(keydims)

    local dstrel  = self.relation
    local offptr  = ctxt:FieldPtr(dstrel:_INTERNAL_GroupedOffset())
    local lenptr  = ctxt:FieldPtr(dstrel:_INTERNAL_GroupedLength())
    --local indexdata = self.relation._grouping.index:DataPtr()
    local v = quote
        var k   = [key]
        var off = offptr[ indexarith(k) ]
        var len = lenptr[ indexarith(k) ]
        --var idx = [indexdata]
    in 
        sType { off, off+len }
        --sType { idx[k.a[0]].a[0], idx[k.a[0]+1].a[0] }
    end
    return v
end

function doProjection(key,field,ctxt)
    assert(L.is_field(field))
    local dataptr     = ctxt:FieldPtr(field)
    local keydims     = field:Relation():Dims()
    local indexarith  = T.linAddrTerraGen(keydims)
    return `dataptr[ indexarith(key) ]
end


function ast.GlobalReduce:codegen(ctxt)
  -- GPU impl:
  if ctxt:onGPU() then
    local lval = ctxt.reduce:sharedMemPtr(self.global.global)
    local rexp = self.exp:codegen(ctxt)
    local rhs = mat_bin_exp(self.reduceop, self.global.node_type, lval, rexp, self.global.node_type, self.exp.node_type)
    return quote [lval] = [rhs] end
  end

  -- CPU impl forwards to assignment codegen
  local assign = ast.Assignment:DeriveFrom(self)
  assign.lvalue = self.global
  assign.exp    = self.exp
  assign.reduceop = self.reduceop

  return assign:codegen(ctxt)
end


function ast.FieldWrite:codegen (ctxt)
  local phase = ctxt:fieldPhase(self.fieldaccess.field)
  if ctxt:onGPU() and self.reduceop and not phase:isCentered() then
    local lval = self.fieldaccess:codegen(ctxt)
    local rexp = self.exp:codegen(ctxt)
    return atomic_gpu_mat_red_exp(self.reduceop, self.fieldaccess.node_type, lval, rexp, self.exp.node_type)
  else
    -- just re-direct to an assignment statement otherwise
    local assign = ast.Assignment:DeriveFrom(self)
    assign.lvalue = self.fieldaccess
    assign.exp    = self.exp
    if self.reduceop then assign.reduceop = self.reduceop end

    return assign:codegen(ctxt)
  end
end

function ast.FieldAccess:codegen (ctxt)
  local key         = self.key:codegen(ctxt)
  local dataptr     = ctxt:FieldPtr(self.field)
  local keydims     = self.field:Relation():Dims()
  local indexarith  = T.linAddrTerraGen(keydims)
  return `dataptr[ indexarith(key) ]
end


--[[--------------------------------------------------------------------]]--
--[[                          INSERT/ DELETE                            ]]--
--[[--------------------------------------------------------------------]]--


function ast.DeleteStatement:codegen (ctxt)
  local relation  = self.key.node_type.relation

  local key       = self.key:codegen(ctxt)
  local live_mask = ctxt:FieldPtr(relation._is_live_mask)
  local set_mask_stmt = quote live_mask[key.a[0]] = false end

  local updated_size     = ctxt:deleteSizeVar()
  local size_update_stmt = quote [updated_size] = [updated_size]-1 end

  return quote set_mask_stmt size_update_stmt end
end

function ast.InsertStatement:codegen (ctxt)
  local relation = self.relation.node_type.value -- to insert into

  -- index to write to
  local index = ctxt:getInsertIndex()

  -- start with writing the live mask
  local live_mask  = ctxt:FieldPtr(relation._is_live_mask)
  local write_code = quote live_mask[index] = true end

  -- the rest of the fields should be assigned values based on the
  -- record literal specified as an argument to the insert statement
  for field,i in pairs(self.fieldindex) do
    local exp_code = self.record.exprs[i]:codegen(ctxt)
    local fieldptr = ctxt:FieldPtr(field)

    write_code = quote
      write_code
      fieldptr[index] = exp_code
    end
  end

  local inc_stmt = ctxt:incrementInsertIndex()

  return quote
    write_code
    inc_stmt
  end
end



