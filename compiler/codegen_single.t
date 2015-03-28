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

local Context = {}
Context.__index = Context

function Context.New(env, bran)
    local ctxt = setmetatable({
        env  = env,
        bran = bran,
    }, Context)
    return ctxt
end

function Context:localenv()
  return self.env:localenv()
end
function Context:enterblock()
  self.env:enterblock()
end
function Context:leaveblock()
  self.env:leaveblock()
end

-- -- -- -- -- -- -- -- -- -- -- -- -- --
-- Info about the relation mapped over

function Context:dims()
  if not self.bran.dims then self.bran.dims = self.bran.relation:Dims() end
  return self.bran.dims
end

function Context:argKeyTerraType()
  return L.key(self.bran.relation):terraType()
end

-- This one is the odd one out, generates some code
function Context:isLiveCheck(param_var)
  assert(self:isOverElastic())
  local ptr = self:FieldPtr(self.bran.relation._is_live_mask)
  -- Assuming 1D address is ok, b/c elastic relations must be 1D
  return `ptr[param_var.a[0]]
end

-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
-- Argument Struct related context functions

function Context:argsType()
  return self.bran:argsType()
end

function Context:argsym()
  if not self.arg_symbol then
    self.arg_symbol = symbol(self:argsType())
  end
  return self.arg_symbol
end

-- -- -- -- -- -- -- -- -- -- -- --
-- Modal Data

function Context:onGPU()
  return self.bran:isOnGPU()
end
function Context:hasGPUReduce()
  return self.bran:UsesGPUReduce()
end
function Context:isOverElastic() -- meaning the relation mapped over
  return self.bran:overElasticRelation()
end
function Context:isOverSubset() -- meaning a subset of the relation mapped over
  return self.bran:isOverSubset()
end

-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
-- Generic Field / Global context functions

function Context:hasExclusivePhase(field)
  return self.bran.kernel.field_use[field]:isCentered()
end

function Context:FieldPtr(field)
  return self.bran:getTerraFieldPtr(self:argsym(), field)
end

function Context:GlobalPtr(global)
  return self.bran:getTerraGlobalPtr(self:argsym(), global)
end

-- -- -- -- -- -- -- -- -- -- -- --
-- GPU related context functions

function Context:gpuBlockSize()
  return self.bran:getBlockSize()
end
function Context:gpuSharedMemBytes()
  return self.bran:nBytesSharedMem()
end

function Context:tid()
  if not self._tid then self._tid = symbol(uint32) end
  return self._tid
end

function Context:bid()
  if not self._bid then self._bid = symbol(uint32) end
  return self._bid
end

-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
-- GPU reduction related context functions

function Context:gpuReduceSharedMemPtr(globl)
  local tid = self:tid()
  local shared_ptr = self.bran:getTerraReduceSharedMemPtr(globl)
  return `[shared_ptr][tid]
end

-- Two odd functions to ask the bran to generate a bit of code
-- TODO: Should we refactor the actual codegen functions in the Bran
-- into a "codegen support" file, also containing the arithmetic
-- expression dispatching.
--    RULE FOR REFACTOR: take all codegen that does not depend on
--      The AST structure, and factor that into one file apart
--      from this AST driven codegeneration file
function Context:codegenSharedMemInit()
  return self.bran:GenerateSharedMemInitialization(self:tid())
end
function Context:codegenSharedMemTreeReduction()
  return self.bran:GenerateSharedMemReduceTree(self:argsym(),
                                               self:tid(),
                                               self:bid())
end

-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
-- Insertion / Deletion related context functions

function Context:deleteSizeVar()
  local dd = self.bran.delete_data
  if dd then
    return `@[self:GlobalPtr(dd.updated_size)]
  end
end

function Context:getInsertIndex()
  return `[self:argsym()].insert_write
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
--[[                         CPU Codegen                                ]]--
--[[--------------------------------------------------------------------]]--

-- We use the separate size parameter to pass in a symbol
-- whose value may be dynamic, thus ensuring that elastic
-- relations are properly iterated over as they change size
local function terraIterNd(dims, size, func)
  local atyp = L.addr_terra_types[#dims]
  local addr = symbol(atyp)
  local iters = {}
  for d=1,#dims do iters[d] = symbol(uint64) end
  local loop = quote
    var [addr] = [atyp]({ a = array( iters ) })
    [func(addr)]
  end
  if #dims == 1 then -- use potentially dynamic size for Non-Grids
    loop = quote for [iters[1]] = 0, size do [loop] end end
  else
    for drev=1,#dims do
      local d = #dims-drev + 1 -- flip loop order of dimensions
      loop = quote for [iters[d]] = 0, [dims[d]] do [loop] end end
    end
  end
  return loop
end

local function terraGPUId_to_Nd(dims, size, id, func)
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

local function cpu_codegen (kernel_ast, ctxt)

  ctxt:enterblock()
    -- declare the symbol for the parameter key
    local param = symbol(ctxt:argKeyTerraType())
    ctxt:localenv()[kernel_ast.name] = param

    local dims                  = ctxt:dims()
    local nrow_sym              = `[ctxt:argsym()].n_rows
    local linid
    if ctxt:onGPU() then linid  = symbol(uint64) end

    local body = kernel_ast.body:codegen(ctxt)

    -- Handle Masking of dead rows when mapping
    -- Over an Elastic Relation
    if ctxt:isOverElastic() then
      if ctxt:onGPU() then
        error("INTERNAL: ELASTIC ON GPU CURRENTLY UNSUPPORTED")
      else
        body = quote
          if [ctxt:isLiveCheck(param)] then [body] end
        end
      end
    end

    -- GENERATE FOR SUBSETS
    if ctxt:isOverSubset() then

      -- GPU SUBSET VERSION
      if ctxt:onGPU() then
        body = terraGPUId_to_Nd(dims, nrow_sym, linid, function(addr)
          return quote
            -- set param
            var [param]
            var use_index = not [ctxt:argsym()].use_boolmask
            if use_index then
              if id < [ctxt:argsym()].index_size then
                param = [ctxt:argsym()].index[id]
              end
            else -- use_boolmask
              param = addr
            end

            -- conditionally execute
            if use_index or [ctxt:argsym()].boolmask[id] then
              [body]
            end
          end
        end)

      -- CPU SUBSET VERSION
      else
        body = quote
          if [ctxt:argsym()].use_boolmask then
          -- BOOLMASK SUBSET BRANCH
            [terraIterNd(dims, nrow_sym, function(iter) return quote
              if [ctxt:argsym()].boolmask[ [T.linAddrTerraGen(dims)](iter) ]
              then
                var [param] = iter
                [body]
              end
            end end)]
          else
          -- INDEX SUBSET BRANCH
            -- ONLY GENERATE FOR NON-GRID RELATIONS
            escape if #ctxt:dims() > 1 then
              local size_sym    = symbol('index_size')
              emit quote
                var [size_sym]  = [ctxt:argsym()].index_size
                [terraIterNd({ size_sym }, nrow_sym, function(iter)
                  return quote
                    var [param] = [ctxt:argsym()].index[iter.a[0]]
                    [body]
                  end
                end)]
              end
            end end
          end
        end
      end

    -- GENERATE FOR FULL RELATION
    else

      -- GPU FULL RELATION VERSION
      if ctxt:onGPU() then
        body = terraGPUId_to_Nd(dims, nrow_sym, linid, function(addr)
          return quote
            var [param] = [addr]
            [body]
          end
        end)

      -- CPU FULL RELATION VERSION
      else
        body = terraIterNd(dims, nrow_sym, function(iter) return quote
          var [param] = iter
          [body]
        end end)
      end

    end

    -- Extra GPU wrapper
    if ctxt:onGPU() then

      -- Extra GPU Reduction setup/post-process
      if ctxt:hasGPUReduce() then
        body = quote
          [ctxt:codegenSharedMemInit()]
          G.barrier()
          [body]
          G.barrier()
          [ctxt:codegenSharedMemTreeReduction()]
        end
      end

      body = quote
        var [ctxt:tid()] = G.thread_id()
        var [ctxt:bid()] = G.block_id()
        var [linid]      = [ctxt:bid()] * [ctxt:gpuBlockSize()] + [ctxt:tid()]

        [body]
      end
    end

  ctxt:leaveblock()

  -- BUILD GPU LAUNCHER
  if ctxt:onGPU() then
    local cuda_kernel = terra([ctxt:argsym()]) [body] end
    cuda_kernel:setname(kernel_ast.id .. '_cudakernel')
    cuda_kernel = G.kernelwrap(cuda_kernel, L._INTERNAL_DEV_OUTPUT_PTX,
                               { {"maxntidx",64}, {"minctasm",6} })

    local MAX_GRID_DIM = 65536
    local launcher = terra (n_blocks : uint, args_ptr : &ctxt:argsType())
      var [ctxt:argsym()] = @args_ptr
      var grid_x : uint,    grid_y : uint,    grid_z : uint   =
          G.get_grid_dimensions(n_blocks, MAX_GRID_DIM)
      var params = terralib.CUDAParams {
        grid_x, grid_y, grid_z,
        [ctxt:gpuBlockSize()], 1, 1,
        [ctxt:gpuSharedMemBytes()], nil
      }
      cuda_kernel(&params, [ctxt:argsym()])
      G.sync() -- flush print streams
      -- TODO: Does this sync cause any performance problems?
    end
    launcher:setname(kernel_ast.id)
    return launcher

  -- BUILD CPU LAUNCHER
  else
    local k = terra (args_ptr : &ctxt:argsType())
      var [ctxt:argsym()] = @args_ptr
      [body]
    end
    k:setname(kernel_ast.id)
    return k
  end
end


--[[--------------------------------------------------------------------]]--
--[[                         GPU Codegen                                ]]--
--[[--------------------------------------------------------------------]]--

Codegen.reduction_identity = Cc.reduction_identity
Codegen.reduction_binop    = Cc.reduction_binop


local function gpu_codegen (kernel_ast, ctxt)

  -----------------------------
  --[[ Codegen CUDA kernel ]]--
  -----------------------------
  ctxt:enterblock()
    -- declare the symbol for iteration
    local param = symbol(ctxt:argKeyTerraType())
    ctxt:localenv()[kernel_ast.name] = param
    local id  = symbol(uint64)

    if ctxt:isOverElastic() then
      error("INTERNAL: ELASTIC ON GPU CURRENTLY UNSUPPORTED")
    end
    local dims = ctxt:dims()

    local body = kernel_ast.body:codegen(ctxt)

    if ctxt:isOverSubset() then

      body = terraGPUId_to_Nd(dims,
      `[ctxt:argsym()].n_rows, id, function(addr)
        return quote
          -- set param
          var [param]
          if [ctxt:argsym()].use_boolmask then
            param = addr
          else
            if id < [ctxt:argsym()].index_size then
              param = [ctxt:argsym()].index[id]
            end
          end

          -- conditionally execute
          if    not [ctxt:argsym()].use_boolmask
             or [ctxt:argsym()].boolmask[id]
          then
            [body]
          end
        end
      end)
    else
      body = terraGPUId_to_Nd(dims,
      `[ctxt:argsym()].n_rows, id, function(addr)
        return quote
          var [param] = [addr]
          [body]
        end
      end)
    end

    local kernel_body = quote
      var [ctxt:tid()] = G.thread_id()
      var [ctxt:bid()] = G.block_id()
      var [id] = [ctxt:bid()] * [ctxt:gpuBlockSize()] + [ctxt:tid()]

      -- Initialize shared memory for global reductions
      --  for kernels that require it
      escape if ctxt:hasGPUReduce() then emit quote
        [ctxt:codegenSharedMemInit()]
        G.barrier()
      end end end
      
      [body]

      -- reduce block reduction temporaries to one value and copy back to GPU
      -- global memory for kernels that require it
      escape if ctxt:hasGPUReduce() then emit quote
        G.barrier()
        [ctxt:codegenSharedMemTreeReduction()]
      end end end
    end

  ctxt:leaveblock()

  local cuda_kernel = terra ([ctxt:argsym()])
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
  local MAX_GRID_DIM = 65536
  local launcher = terra (n_blocks : uint, args_ptr : &ctxt:argsType())
    var [ctxt:argsym()] = @args_ptr
    var grid_x : uint,
        grid_y : uint,
        grid_z : uint   = G.get_grid_dimensions(n_blocks, MAX_GRID_DIM)
    var params = terralib.CUDAParams {
      grid_x, grid_y, grid_z,
      [ctxt:gpuBlockSize()], 1, 1,
      [ctxt:gpuSharedMemBytes()], nil
    }

    cuda_kernel(&params, [ctxt:argsym()])
    G.sync() -- flush print streams
  end

  return launcher
end



--[[--------------------------------------------------------------------]]--
--[[                        Codegen Entrypoint                          ]]--
--[[--------------------------------------------------------------------]]--


function Codegen.codegen (kernel_ast, bran)
  local env  = terralib.newenvironment(nil)
  local ctxt = Context.New(env, bran)

  if ctxt:onGPU() then
    return gpu_codegen(kernel_ast, ctxt)
  else
    return cpu_codegen(kernel_ast, ctxt)
  end

end



--[[--------------------------------------------------------------------]]--
--[[                             GPU Atomics                            ]]--
--[[--------------------------------------------------------------------]]--




local function atomic_gpu_red_exp (op, typ, lvalptr, update)
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


local function atomic_gpu_mat_red_exp(op, result_typ, lval, rhe, rhtyp)
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

local function doProjection(key,field,ctxt)
    assert(L.is_field(field))
    local dataptr     = ctxt:FieldPtr(field)
    local keydims     = field:Relation():Dims()
    local indexarith  = T.linAddrTerraGen(keydims)
    return `dataptr[ indexarith(key) ]
end


function ast.GlobalReduce:codegen(ctxt)
  -- GPU impl:
  if ctxt:onGPU() then
    local lval = ctxt:gpuReduceSharedMemPtr(self.global.global)
    local rexp = self.exp:codegen(ctxt)
    local rhs = mat_bin_exp(self.reduceop, self.global.node_type,
                            lval, rexp,
                            self.global.node_type, self.exp.node_type)
    return quote [lval] = [rhs] end

  -- CPU impl: forwards to assignment codegen
  else
    local assign = ast.Assignment:DeriveFrom(self)
    assign.lvalue = self.global
    assign.exp    = self.exp
    assign.reduceop = self.reduceop

    return assign:codegen(ctxt)
  end
end


function ast.FieldWrite:codegen (ctxt)
  -- If this is a field-reduction on the GPU
  if ctxt:onGPU() and
     self.reduceop and
     not ctxt:hasExclusivePhase(self.fieldaccess.field)
  then
    local lval = self.fieldaccess:codegen(ctxt)
    local rexp = self.exp:codegen(ctxt)
    return atomic_gpu_mat_red_exp(self.reduceop,
                                  self.fieldaccess.node_type,
                                  lval, rexp, self.exp.node_type)
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



