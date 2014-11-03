local Codegen = {}
package.loaded["compiler.codegen"] = Codegen

local ast = require "compiler.ast"

local C = terralib.require 'compiler.c'
local L = terralib.require 'compiler.lisztlib'
local G = terralib.require 'compiler.gpu_util'


----------------------------------------------------------------------------

local Context = {}
Context.__index = Context

function Context.new(env, bran)
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

function Context:onGPU()
  return self.bran.location == L.GPU
end

function Context:FieldPtr(field)
  return self.bran:getRuntimeFieldPtr(field)
end
function Context:GlobalScratchPtr(global, on_cpu)
  if on_cpu then
    return self.bran:getCPUScratchTablePtr(global)
  else
    return self.bran:getGPUScratchTablePtr(global)
  end
end
function Context:ScratchTableSize()
  return self.bran.cpu_scratch_table:byteSize()
end
function Context:cpuScratchTable()
  return self.bran.cpu_scratch_table:ptr()
end
function Context:gpuScratchTable()
  return self.bran.gpu_scratch_table:ptr()
end
function Context:GlobalPtr(global)
  return self.bran:getRuntimeGlobalPtr(global)
end
function Context:SetGlobalSharedPtrs(shared_ptrs)
  self.global_shared_ptrs = shared_ptrs
end
function Context:GlobalSharedPtr(global, tid)
  return quote
    var tid = G.thread_id()
  in
    [self.global_shared_ptrs[global]][tid]
  end
end
function Context:runtimeGerm()
  return self.bran.runtime_germ:ptr()
end
function Context:cpuGerm()
  return self.bran.cpu_germ:ptr()
end
function Context:fieldPhase(field)
  return self.bran.kernel.field_use[field]
end
function Context:globalPhase(global)
  return self.bran.kernel.global_use[global]
end
function Context:isLiveCheck(param_var)
  local ptr = self:FieldPtr(self.bran.relation._is_live_mask)
  return `ptr[param_var]
end
function Context:deleteSizeVar()
  local dd = self.bran.delete_data
  if dd then
    return `@[self:GlobalPtr(dd.updated_size)]
  end
end
function Context:getInsertIndex()
  return `[self:runtimeGerm()].insert_write
end
function Context:incrementInsertIndex()
  local insert_index = self:getInsertIndex()
  local counter = self:GlobalPtr(self.bran.insert_data.n_inserted)

  return quote
    insert_index = insert_index + 1
    @counter = @counter + 1
  end
end




local function vec_mapgen(typ,func)
  local arr = {}
  for i=1,typ.N do arr[i] = func(i-1) end
  return `[typ:terraType()]({ array([arr]) })
end
local function mat_mapgen(typ,func)
  local rows = {}
  for i=1,typ.Nrow do
    local r = {}
    for j=1,typ.Ncol do r[j] = func(i-1,j-1) end
    rows[i] = `array([r])
  end
  return `[typ:terraType()]({ array([rows]) })
end

local function vec_foldgen(N, init, binf)
  local acc = init
  for ii = 1, N do local i = N - ii -- count down to 0
    acc = binf(i, acc) end
  return acc
end
local function mat_foldgen(N,M, init, binf)
  local acc = init
  for ii = 1, N do local i = N - ii -- count down to 0
    for jj = 1, M do local j = M - jj -- count down to 0
      acc = binf(i,j, acc) end end
  return acc
end




--[[--------------------------------------------------------------------]]--
--[[                         CPU Codegen                                ]]--
--[[--------------------------------------------------------------------]]--

local function cpu_codegen (kernel_ast, ctxt)
  ctxt:enterblock()
    -- declare the symbol for iteration
    local param = symbol(L.row(ctxt.bran.relation):terraType())
    ctxt:localenv()[kernel_ast.name] = param

    -- insert a check for the live row mask
    local body  = quote
      if [ctxt:isLiveCheck(param)] then
        [kernel_ast.body:codegen(ctxt)]
      end
    end

    -- by default on CPU just iterate over all the possible rows
    local kernel_body = quote
      for [param] = 0, [ctxt:runtimeGerm()].n_rows do
        [body]
      end
    end

    -- special iteration logic for subset-mapped kernels
    if ctxt.bran.subset then
      kernel_body = quote
        if [ctxt:runtimeGerm()].use_boolmask then
          var boolmask = [ctxt:runtimeGerm()].boolmask
          for [param] = 0, [ctxt:runtimeGerm()].n_rows do
            if boolmask[param] then -- subset guard
              [body]
            end
          end
        else
          var index = [ctxt:runtimeGerm()].index
          var size = [ctxt:runtimeGerm()].index_size
          for itr = 0,size do
            var [param] = index[itr]
            [body]
          end
        end
      end
    end
  ctxt:leaveblock()

  local k = terra () [kernel_body] end
  k:setname(kernel_ast.id)
  return k
end


--[[--------------------------------------------------------------------]]--
--[[                         GPU Codegen                                ]]--
--[[--------------------------------------------------------------------]]--
local checkCudaError = macro(function(code)
    return quote
        if code ~= 0 then
            C.printf("CUDA ERROR: ")
            error(C.cudaGetErrorString(code))
        end
    end
end)

local function scalar_reduce_identity (ltype, reduceop)
  if ltype == L.int then
    if reduceop == '+' or reduceop == '-' then
      return `0
    elseif reduceop == '*' or reduceop == '/' then
      return `1
    elseif reduceop == 'min' then
      return `[C.INT_MAX]
    elseif reduceop == 'max' then
      return `[C.INT_MIN]
    end
  elseif ltype == L.uint64 then
    if reduceop == '+' or reduceop == '-' then
      return `0
    elseif reduceop == '*' or reduceop == '/' then
      return `1
    elseif reduceop == 'min' then
      return `[C.ULONG_MAX]
    elseif reduceop == 'max' then
      return `0
    end
  elseif ltype == L.float then
    if reduceop == '+' or reduceop == '-' then
      return `0.0f
    elseif reduceop == '*' or reduceop == '/' then
      return `1.0f
    elseif reduceop == 'min' then
      return `[C.FLT_MAX]
    elseif reduceop == 'max' then
      return `[C.FLT_MIN]
    end
  elseif ltype == L.double then
    if reduceop == '+' or reduceop == '-' then
      return `0.0
    elseif reduceop == '*' or reduceop == '/' then
      return `1.0
    elseif reduceop == 'min' then
      return `[C.DBL_MAX]
    elseif reduceop == 'max' then
      return `[C.DBL_MIN]
    end
  elseif ltype == L.bool then
    if reduceop == 'and' then
      return `true
    elseif reduceop == 'or' then
      return `false
    end
  end
  -- we should never reach this
  error("scalar identity for reduction operator " .. reduceop .. 'on type '
        .. tostring(ltype) ' not implemented')
end

function tree_reduce_op (reduceop)
  if reduceop == '-' then return '+' end
  if reduceop == '/' then return '*' end
  return reduceop
end

local function reduce_identity(ltype, reduceop)
  if not ltype:isVector() then
    return scalar_reduce_identity(ltype, reduceop)
  end
  local scalar_id = scalar_reduce_identity(ltype:baseType(), reduceop)
  return quote
    var rid : ltype:terraType()
    var tmp : &ltype:terraBaseType() = [&ltype:terraBaseType()](&rid)
    for i = 0, [ltype.N] do
      tmp[i] = [scalar_id]
    end
  in
    [rid]
  end
end

local function initialize_global_shared_memory (global_shared_ptrs, ctxt)
  local tid = symbol(uint32)
  local init_code = quote
    var [tid] = G.thread_id()
  end

  for g, shared in pairs(global_shared_ptrs) do
    local gtype = g.type
    local reduceop = ctxt:globalPhase(g).reduceop

    init_code = quote
      [init_code]
      [shared][tid] = [reduce_identity(gtype,reduceop)]
    end
  end
  return init_code
end

local function unrolled_block_reduce (op, typ, ptr, tid, block_size)
    local expr = quote end
    local step = block_size

    op = tree_reduce_op(op)
    while (step > 1) do
        step = step/2
        expr = quote
            [expr]
            if tid < step then
              [ptr][tid] = [vec_bin_exp(op, typ, `[ptr][tid], `[ptr][tid + step], typ, typ)]
            end
            G.barrier()
        end

        -- Pairwise reductions over > 32 threads need to be synchronized b/c
        -- they aren't guaranteed to be executed in lockstep, as they are
        -- running in multiple warps.  But the store must be volatile or the
        -- compiler might re-order them!
        --if step > WARP_SIZE then
        --    expr = quote [expr] G.barrier()
        --end
    end
    return expr
end

local function reduce_global_shared_memory (global_shared_ptrs, ctxt, block_size, commit_final_value)
  local tid = symbol(uint64) -- thread id
  local bid = symbol(uint64) -- block id
  local reduce_code = quote
    var [tid] = G.thread_id()
    var [bid] = G.block_id()
  end

  for g, shared in pairs(global_shared_ptrs) do

    local gtype = g.type
    local reduceop = ctxt:globalPhase(g).reduceop
    local scratch_ptr = ctxt:GlobalScratchPtr(g)

   reduce_code = quote
      [reduce_code]

      [unrolled_block_reduce(reduceop, gtype, shared, tid, block_size)]
      if tid == 0 then
        escape
          if commit_final_value then
            local finalptr = ctxt:GlobalPtr(g)
            emit quote
              @[finalptr] = [vec_bin_exp(reduceop, gtype,`@[finalptr],`[shared][0],gtype,gtype)]
            end
          else
            emit quote
              [scratch_ptr][bid] = shared[0]
            end
          end
        end
      end
    end
  end

  return reduce_code
end

local function generate_final_reduce (kernel_ast, global_shared_ptrs, ctxt, block_size)
  local array_len  = symbol(uint64)

  -- read in all global reduction values corresponding to this (global) thread
  local final_reduce = quote
    var t  = G.thread_id()
    var b  = G.block_id()
    var gt = t + [block_size] * b
    var num_blocks : uint64 = G.num_blocks()

    [initialize_global_shared_memory(global_shared_ptrs, ctxt)]

    for global_index = gt, [array_len], num_blocks*[block_size] do
      escape
        for g, shared in pairs(global_shared_ptrs) do
          local op = ctxt:globalPhase(g).reduceop
          local typ = g.type
          local gptr = ctxt:GlobalScratchPtr(g)
          emit quote
            [shared][t] = [vec_bin_exp(op, typ, `[shared][t], `[gptr][global_index], typ, typ)]
          end
        end
      end
    end
    [reduce_global_shared_memory(global_shared_ptrs, ctxt, block_size, true)]
  end

  final_reduce = terra ([array_len])
    [final_reduce] 
  end
  local id = kernel_ast.id .. '_reduce'
  -- using id here to set debug name of kernel for tuning/debugging
  return terralib.cudacompile({[id]=final_reduce})[id]
end

local function allocate_reduction_space (n_blocks, global_shared_ptrs, ctxt)
  local alloc_code = quote end

  for g, _ in pairs(global_shared_ptrs) do
    local ptr = `[ctxt:GlobalScratchPtr(g,true)]
    alloc_code = quote
      [alloc_code]
      var sz = [sizeof(g.type:terraType())]*n_blocks
      checkCudaError(C.cudaMalloc([&&opaque](&ptr), sz))
    end
  end

  -- after initializing global scratch pointers in CPU memory,
  -- copy pointers to GPU to be accessed during kernel invocation
  alloc_code = quote
    [alloc_code]
    C.cudaMemcpy([ctxt:gpuScratchTable()],
                 [ctxt:cpuScratchTable()],
                 [ctxt:ScratchTableSize()],
                 C.cudaMemcpyHostToDevice)
  end

  return alloc_code
end

local function free_reduction_space (global_shared_ptrs, ctxt)
  local free_code = quote end

  for g, _ in pairs(global_shared_ptrs) do
    free_code = quote
      [free_code]
      C.cudaFree([ctxt:GlobalScratchPtr(g,true)])
    end
  end

  return free_code
end

local function compute_global_reduction_data (ctxt, block_size)
  local shared_mem_size = 0
  local global_shared_ptrs = { }
  local kernel = ctxt.bran.kernel
  local codegen_reduce = false
  for g, phase in pairs(kernel.global_use) do
    if phase.reduceop then
      codegen_reduce = true
      global_shared_ptrs[g] = cudalib.sharedmemory(g.type:terraType(), block_size)
      shared_mem_size = shared_mem_size + sizeof(g.type:terraType()) * block_size
    end
  end
  return codegen_reduce, global_shared_ptrs, shared_mem_size
end

local function gpu_codegen (kernel_ast, ctxt)
  local BLOCK_SIZE   = 64
  local MAX_GRID_DIM = 65536

  local codegen_reduce, global_shared_ptrs, shared_mem_size = compute_global_reduction_data(ctxt, BLOCK_SIZE)
  ctxt:SetGlobalSharedPtrs(global_shared_ptrs)
  ctxt:enterblock()
    -- declare the symbol for iteration
    local param = symbol(L.row(ctxt.bran.relation):terraType())
    ctxt:localenv()[kernel_ast.name] = param

    local id = symbol(uint64)
    local body = quote
      if [ctxt:isLiveCheck(param)] then
        [kernel_ast.body:codegen(ctxt)]
      end
    end

    if ctxt.bran.subset then
      body = quote
        if [ctxt:runtimeGerm()].use_boolmask then
          var [param] = id
          if [param] < [ctxt:runtimeGerm()].n_rows and
             [ctxt:runtimeGerm()].boolmask[param]
          then
            [body]
          end
        else
          if id < [ctxt:runtimeGerm()].index_size then
            var [param] = [ctxt:runtimeGerm()].index[id]
            [body]
          end
        end
      end
    else
      body = quote
        var [param] = id
        if [param] < [ctxt:runtimeGerm()].n_rows then
          [body]
        end
      end
    end

    local kernel_body = quote
      var [id] : uint64 = G.block_id() * BLOCK_SIZE + G.thread_id()

      -- initialize shared memory for global reductions for kernels that require it
      escape if codegen_reduce then emit quote
        [initialize_global_shared_memory(global_shared_ptrs, ctxt)]
        G.barrier()
      end end end
      
      [body]

      -- reduce L.globals to one value and copy back to GPU global memory for kernels that require it
      escape if codegen_reduce then emit quote
        G.barrier()
        [reduce_global_shared_memory(global_shared_ptrs, ctxt, BLOCK_SIZE)]
      end end end
    end

  ctxt:leaveblock()

  local cuda_kernel = terra ()
    [kernel_body]
  end
  -- using kernel_ast.id here to set debug name of kernel for tuning/debugging
  cuda_kernel = terralib.cudacompile { [kernel_ast.id] = cuda_kernel }[kernel_ast.id]
  local reduce_global_scratch_values = generate_final_reduce(kernel_ast, global_shared_ptrs,ctxt,BLOCK_SIZE)
  -- germ type will have a use_boolmask field only if it
  -- was generated for a subset kernel
  if ctxt.bran.subset then
    local launcher = terra ()
      var n_blocks = uint(C.ceil(
        [ctxt:cpuGerm()].n_rows / float(BLOCK_SIZE)))
      var grid_x : uint, grid_y : uint, grid_z : uint = 
        G.get_grid_dimensions(n_blocks, MAX_GRID_DIM)
      var params = terralib.CUDAParams {
        grid_x, grid_y, grid_z,
        BLOCK_SIZE, 1, 1,
        shared_mem_size, nil
      }

      if not [ctxt:cpuGerm()].use_boolmask then
        var n_blocks = uint(C.ceil(
          [ctxt:cpuGerm()].index_size / float(BLOCK_SIZE)))
        var grid_x : uint, grid_y : uint, grid_z : uint = 
          G.get_grid_dimensions(n_blocks, MAX_GRID_DIM)
        params = terralib.CUDAParams {
          grid_x, grid_y, grid_z,
          BLOCK_SIZE, 1, 1,
          shared_mem_size, nil
        }
      end

      [allocate_reduction_space(n_blocks,global_shared_ptrs, ctxt)]
      cuda_kernel(&params)
      G.sync() -- flush print streams

      var reduce_params = terralib.CUDAParams {
        1,1,1,
        BLOCK_SIZE,1,1,
        shared_mem_size, nil
      }
      reduce_global_scratch_values(&reduce_params,n_blocks)
      [free_reduction_space(global_shared_ptrs, ctxt)]
    end
    return launcher

  else
    local launcher = terra ()
      var n_blocks = uint(C.ceil(
        [ctxt:cpuGerm()].n_rows / float(BLOCK_SIZE)))
      var grid_x : uint, grid_y : uint, grid_z : uint =
        G.get_grid_dimensions(n_blocks, MAX_GRID_DIM)
      var params = terralib.CUDAParams {
        grid_x, grid_y, grid_z,
        BLOCK_SIZE, 1, 1,
        shared_mem_size, nil
      }
      [allocate_reduction_space(n_blocks,global_shared_ptrs, ctxt)]
      cuda_kernel(&params)
      G.sync() -- flush print streams

      var reduce_params = terralib.CUDAParams {
        1,1,1,
        BLOCK_SIZE,1,1,
        shared_mem_size, nil
      }
      reduce_global_scratch_values(&reduce_params,n_blocks)
      [free_reduction_space(global_shared_ptrs, ctxt)]
    end
    return launcher
  end
end

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
--[[                      Shared Codegen                                ]]--
--[[--------------------------------------------------------------------]]--

function ast.AST:codegen (ctxt)
  error("Codegen not implemented for AST node " .. self.kind)
end

function ast.ExprStatement:codegen (ctxt)
  return self.exp:codegen(ctxt)
end

-- complete no-op
function ast.Quote:codegen (ctxt)
  return self.code:codegen(ctxt)
end

function ast.LetExpr:codegen (ctxt)
  ctxt:enterblock()
  local block = self.block:codegen(ctxt)
  local exp   = self.exp:codegen(ctxt)
  ctxt:leaveblock()

  return quote [block] in [exp] end
end

-- DON'T CODEGEN THE KERNEL THIS WAY; HANDLE IN Codegen.codegen()
--function ast.LisztKernel:codegen (ctxt)
--end

function ast.Block:codegen (ctxt)
  -- start with an empty ast node, or we'll get an error when appending new quotes below
  local code = quote end
  for i = 1, #self.statements do
    local stmt = self.statements[i]:codegen(ctxt)
    code = quote code stmt end
  end
  return code
end

function ast.CondBlock:codegen(ctxt, cond_blocks, else_block, index)
  index = index or 1

  local cond  = self.cond:codegen(ctxt)
  ctxt:enterblock()
  local body = self.body:codegen(ctxt)
  ctxt:leaveblock()

  if index == #cond_blocks then
    if else_block then
      return quote if [cond] then [body] else [else_block:codegen(ctxt)] end end
    else
      return quote if [cond] then [body] end end
    end
  else
    ctxt:enterblock()
    local nested = cond_blocks[index + 1]:codegen(ctxt, cond_blocks, else_block, index + 1)
    ctxt:leaveblock()
    return quote if [cond] then [body] else [nested] end end
  end
end

function ast.IfStatement:codegen (ctxt)
  return self.if_blocks[1]:codegen(ctxt, self.if_blocks, self.else_block)
end

function ast.WhileStatement:codegen (ctxt)
  local cond = self.cond:codegen(ctxt)
  ctxt:enterblock()
  local body = self.body:codegen(ctxt)
  ctxt:leaveblock()
  return quote while [cond] do [body] end end
end

function ast.DoStatement:codegen (ctxt)
  ctxt:enterblock()
  local body = self.body:codegen(ctxt)
  ctxt:leaveblock()
  return quote do [body] end end
end

function ast.RepeatStatement:codegen (ctxt)
  ctxt:enterblock()
  local body = self.body:codegen(ctxt)
  local cond = self.cond:codegen(ctxt)
  ctxt:leaveblock()

  return quote repeat [body] until [cond] end
end

function ast.NumericFor:codegen (ctxt)
  -- min and max expression should be evaluated in current scope,
  -- iter expression should be in a nested scope, and for block
  -- should be nested again -- that way the loop var is reset every
  -- time the loop runs.
  local minexp  = self.lower:codegen(ctxt)
  local maxexp  = self.upper:codegen(ctxt)
  local stepexp = self.step and self.step:codegen(ctxt) or nil

  ctxt:enterblock()
  local iterstr = self.name
  local itersym = symbol()
  ctxt:localenv()[iterstr] = itersym

  ctxt:enterblock()
  local body = self.body:codegen(ctxt)
  ctxt:leaveblock()
  ctxt:leaveblock()

  if stepexp then
    return quote for [itersym] = [minexp], [maxexp], [stepexp] do [body] end end
  end

  return quote for [itersym] = [minexp], [maxexp] do [body] end end
end

function ast.Break:codegen(ctxt)
  return quote break end
end

function ast.Name:codegen(ctxt)
  local s = ctxt:localenv()[self.name]
  assert(terralib.issymbol(s))
  return `[s]
end

local minexp = function(lhe, rhe)
  return quote
    var a = [lhe]
    var b = [rhe]
    var result = a
    if result > b then result = b end
  in
    result
  end
end

local maxexp = function(lhe, rhe)
  return quote
    var a = [lhe]
    var b = [rhe]
    var result = a
    if result < b then result = b end
  in
    result
  end
end

local function bin_exp (op, lhe, rhe)
  if     op == '+'   then return `[lhe] +   [rhe]
  elseif op == '-'   then return `[lhe] -   [rhe]
  elseif op == '/'   then return `[lhe] /   [rhe]
  elseif op == '*'   then return `[lhe] *   [rhe]
  elseif op == '%'   then return `[lhe] %   [rhe]
  elseif op == '^'   then return `[lhe] ^   [rhe]
  elseif op == 'or'  then return `[lhe] or  [rhe]
  elseif op == 'and' then return `[lhe] and [rhe]
  elseif op == '<'   then return `[lhe] <   [rhe]
  elseif op == '>'   then return `[lhe] >   [rhe]
  elseif op == '<='  then return `[lhe] <=  [rhe]
  elseif op == '>='  then return `[lhe] >=  [rhe]
  elseif op == '=='  then return `[lhe] ==  [rhe]
  elseif op == '~='  then return `[lhe] ~=  [rhe]
  elseif op == 'max' then return maxexp(lhe, rhe)
  elseif op == 'min' then return minexp(lhe, rhe)
  end
end


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


-- ONLY ONE PLACE...
function let_vec_binding(typ, N, exp)
  local val = symbol(typ:terraType())
  local let_binding = quote var [val] = [exp] end

  local coords = {}
  if typ:isVector() then
    for i=1, N do coords[i] = `val.d[i-1] end
  else
    for i=1, N do coords[i] = `val end
  end

  return let_binding, coords
end



function symgen_bind(typ, exp, f)
  local s = symbol(typ:terraType())
  return quote var s = exp in [f(s)] end
end
function symgen_bind2(typ1, typ2, exp1, exp2, f)
  local s1 = symbol(typ1:terraType())
  local s2 = symbol(typ2:terraType())
  return quote
    var s1 = exp1
    var s2 = exp2
  in [f(s1,s2)] end
end

function vec_bin_exp(op, result_typ, lhe, rhe, lhtyp, rhtyp)
  if lhtyp:isPrimitive() and rhtyp:isPrimitive() then
    return bin_exp(op, lhe, rhe)
  end

  -- handles equality and inequality of rows
  if lhtyp:isRow() and rhtyp:isRow() then
    return bin_exp(op, lhe, rhe)
  end

  -- ALL THE CASES

  -- OP: Ord (scalars only)
  -- OP: Mod (scalars only)
  -- BOTH HANDLED ABOVE

  -- OP: Eq (=> DIM: == , BASETYPE: == )
    -- pairwise comparisons, and/or collapse
  local eqinitval = { ['=='] = `true, ['~='] = `false }
  if op == '==' or op == '~=' then
    if lhtyp:isVector() then -- rhtyp:isVector()
      return symgen_bind2(lhtyp, rhtyp, lhe, rhe, function(lvec,rvec)
        return vec_foldgen(lhtyp.N, eqinitval[op], function(i, acc)
          if op == '==' then return `acc and lvec.d[i] == rvec.d[i]
                        else return `acc or  lvec.d[i] ~= rvec.d[i] end
        end) end)

    elseif lhtyp:isSmallMatrix() then -- rhtyp:isSmallMatrix()
      return symgen_bind2(lhtyp, rhtyp, lhe, rhe, function(lmat,rmat)
        return mat_foldgen(lhtyp.Nrow, lhtyp.Ncol, eqinitval[op],
          function(i,j, acc)
            if op == '==' then return `acc and lmat.d[i][j] == rmat.d[i][j]
                          else return `acc or  lmat.d[i][j] ~= rmat.d[i][j] end
          end) end)

    end
  end

  -- OP: Logical (and or)
    -- map the OP
  -- OP: + - min max
    -- map the OP
  if op == 'and'  or op == 'or' or
     op == '+'    or op == '-'  or
     op == 'min'  or op == 'max'
  then
    if lhtyp:isVector() then -- rhtyp:isVector()
      return symgen_bind2(lhtyp, rhtyp, lhe, rhe, function(lvec,rvec)
        return vec_mapgen(result_typ, function(i)
          return bin_exp( op, (`lvec.d[i]), `(rvec.d[i]) ) end) end)

    elseif lhtyp:isSmallMatrix() then
      return symgen_bind2(lhtyp, rhtyp, lhe, rhe, function(lmat,rmat)
        return mat_mapgen(result_typ, function(i,j)
          return bin_exp( op, (`lmat.d[i][j]), `(rmat.d[i][j]) ) end) end)

    end
  end

  -- OP: *
    -- DIM: Scalar _
    -- DIM: _ Scalar
      -- map the OP with expanding one side
  -- OP: /
    -- DIM: _ Scalar
      -- map the OP with expanding one side
  if op == '/' or
    (op == '*' and lhtyp:isPrimitive() or rhtyp:isPrimitive())
  then
    if lhtyp:isVector() then -- rhtyp:isPrimitive()
      return symgen_bind2(lhtyp, rhtyp, lhe, rhe, function(lvec,r)
        return vec_mapgen(result_typ, function(i)
          return bin_exp( op, (`lvec.d[i]), r ) end) end)

    elseif rhtyp:isVector() then -- lhtyp:isPrimitive()
      return symgen_bind2(lhtyp, rhtyp, lhe, rhe, function(l,rvec)
        return vec_mapgen(result_typ, function(i)
          return bin_exp( op, l, `(rvec.d[i]) ) end) end)

    elseif lhtyp:isSmallMatrix() then -- rhtyp:isPrimitive()
      return symgen_bind2(lhtyp, rhtyp, lhe, rhe, function(lmat,r)
        return mat_mapgen(result_typ, function(i,j)
          return bin_exp( op, (`lmat.d[i][j]), r ) end) end)

    elseif rhtyp:isSmallMatrix() then -- rhtyp:isPrimitive()
      return symgen_bind2(lhtyp, rhtyp, lhe, rhe, function(l,rmat)
        return mat_mapgen(result_typ, function(i,j)
          return bin_exp( op, l, `(rmat.d[i][j]) ) end) end)

    end
  end

  -- OP: *
    -- DIM: Vector(n) Matrix(n,_)
    -- DIM: Matrix(_,m) Vector(m)
    -- DIM: Matrix(_,m) Matrix(m,_)
      -- vector-matrix, matrix-vector, or matrix-matrix products
--  if op == '*' then
--    if lhtyp:isVector() and rhtyp:isSmallMatrix() then
--      return symgen_bind2(lhtyp, rhtyp, lhe, rhe, function(lvec,rmat)
--        return vec_mapgen(result_typ, function(j)
--          return vec_foldgen(rmat.Ncol, `0, function(i, acc)
--            return `acc + lvec.d[i] * rmat.d[i][j] end) end) end)
--
--    elseif lhtyp:isSmallMatrix() and rhtyp:isVector() then
--      return symgen_bind2(lhtyp, rhtyp, lhe, rhe, function(lmat,rvec)
--        return vec_mapgen(result_typ, function(i)
--          return vec_foldgen(lmat.Nrow, `0, function(j, acc)
--            return `acc + lmat.d[i][j] * rvec.d[j] end) end) end)
--
--    elseif lhtyp:isSmallMatrix() and rhtyp:isSmallMatrix() then
--      return symgen_bind2(lhtyp, rhtyp, lhe, rhe, function(lmat,rmat)
--        return mat_mapgen(result_typ, function(i,j)
--          return vec_foldgen(rmat.Ncol, `0, function(k, acc)
--            return `acc + lmat.d[i][k] * rmat.d[k][j] end) end) end)
--
--    end
--  end

  -- If we fell through to here we've run into an unhandled branch
  error('Internal Error: Could not find any code to generate for '..
        'binary operator '..op..' with opeands of type '..lhtyp:toString()..
        ' and '..rhtyp:toString())
end

function atomic_gpu_vec_red_exp(op, result_typ, lval, rhe, rhtyp)
  if not result_typ:isVector() then
    return atomic_gpu_red_exp(op, result_typ, `&lval, rhe)
  end

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
end

function ast.Assignment:codegen (ctxt)
  local lhs   = self.lvalue:codegen(ctxt)
  local rhs   = self.exp:codegen(ctxt)

  local ltype, rtype = self.lvalue.node_type, self.exp.node_type

  if self.reduceop then
    rhs = vec_bin_exp(self.reduceop, ltype, lhs, rhs, ltype, rtype)
  end
  return quote [lhs] = rhs end
end

function ast.GlobalReduce:codegen(ctxt)
  -- GPU impl:
  if ctxt:onGPU() then
    local lval = ctxt:GlobalSharedPtr(self.global.global)
    local rexp = self.exp:codegen(ctxt)
    local rhs = vec_bin_exp(self.reduceop, self.global.node_type, lval, rexp, self.global.node_type, self.exp.node_type)
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
    return atomic_gpu_vec_red_exp(self.reduceop, self.fieldaccess.node_type, lval, rexp, self.exp.node_type)
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
  local index = self.row:codegen(ctxt)
  local dataptr = ctxt:FieldPtr(self.field)
  return `@(dataptr + [index])
end

function ast.Cast:codegen(ctxt)
  local typ = self.node_type
  local bt  = typ:terraBaseType()
  local valuecode = self.value:codegen(ctxt)

  if typ:isPrimitive() then
    return `[typ:terraType()](valuecode)

  elseif typ:isVector() then
    local vec = symbol(self.value.node_type:terraType())
    return quote var [vec] = valuecode in
      [ vec_mapgen(typ, function(i) return `[bt](vec.d[i]) end) ] end

  elseif typ:isSmallMatrix() then
    local mat = symbol(self.value.node_type:terraType())
    return quote var [mat] = valuecode in
      [ mat_mapgen(typ, function(i,j) return `[bt](mat.d[i][j]) end) ] end

  else
    error("Internal Error: Type unrecognized "..typ:toString())
  end
end

-- By the time we make it to codegen, Call nodes are only used to represent builtin function calls.
function ast.Call:codegen (ctxt)
    return self.func.codegen(self, ctxt)
end


function ast.DeclStatement:codegen (ctxt)
  local varname = self.name
  local tp      = self.node_type:terraType()
  local varsym  = symbol(tp)

  if self.initializer then
    local exp = self.initializer:codegen(ctxt)
    ctxt:localenv()[varname] = varsym -- MUST happen after init codegen
    return quote 
      var [varsym] = [exp]
    end
  else
    ctxt:localenv()[varname] = varsym -- MUST happen after init codegen
    return quote var [varsym] end
  end
end

function ast.MatrixLiteral:codegen (ctxt)
  local typ = self.node_type

  return mat_mapgen(typ, function(i,j)
    return self.elems[i*self.m + j + 1]:codegen(ctxt)
  end)
end

function ast.VectorLiteral:codegen (ctxt)
  local typ = self.node_type

  return vec_mapgen(typ, function(i)
    return self.elems[i+1]:codegen(ctxt)
  end)
end

function ast.Global:codegen (ctxt)
  local dataptr = ctxt:GlobalPtr(self.global)
  return `@dataptr
end

function ast.SquareIndex:codegen (ctxt)
  local base  = self.base:codegen(ctxt)
  local index = self.index:codegen(ctxt)

  -- Vector case
  if self.index2 == nil then
    return `base.d[index]
  -- Matrix case
  else
    local index2 = self.index2:codegen(ctxt)

    return `base.d[index][index2]
  end
end

function ast.Number:codegen (ctxt)
  return `[self.value]
end

function ast.Bool:codegen (ctxt)
  if self.value == true then
    return `true
  else 
    return `false
  end
end


function ast.UnaryOp:codegen (ctxt)
  local expr = self.exp:codegen(ctxt)
  local typ  = self.node_type

  if typ:isPrimitive() then
    if self.op == '-' then return `-[expr]
                      else return `not [expr] end
  elseif typ:isVector() then
    local vec = symbol(typ:terraType())

    if self.op == '-' then
      return quote var [vec] = expr in
        [ vec_mapgen(typ, function(i) return `-vec.d[i] end) ] end
    else
      return quote var [vec] = expr in
        [ vec_mapgen(typ, function(i) return `not vec.d[i] end) ] end
    end
  elseif typ:isSmallMatrix() then
    local mat = symbol(typ:terraType())

    if self.op == '-' then
      return quote var [mat] = expr in
        [ mat_mapgen(typ, function(i,j) return `-mat.d[i][j] end) ] end
    else
      return quote var [mat] = expr in
        [ mat_mapgen(typ, function(i,j) return `not mat.d[i][j] end) ] end
    end

  else
    error("Internal Error: Type unrecognized "..typ:toString())
  end
end

function ast.BinaryOp:codegen (ctxt)
  local lhe = self.lhs:codegen(ctxt)
  local rhe = self.rhs:codegen(ctxt)

  -- handle case of two primitives
  return vec_bin_exp(self.op, self.node_type,
      lhe, rhe, self.lhs.node_type, self.rhs.node_type)
end

function ast.LuaObject:codegen (ctxt)
    return `{}
end
function ast.Where:codegen(ctxt)
    local key   = self.key:codegen(ctxt)
    local sType = self.node_type:terraType()
    local indexdata = self.relation._grouping.index:DataPtr()
    local v = quote
        var k   = [key]
        var idx = [indexdata]
    in 
        sType { idx[k], idx[k+1] }
    end
    return v
end

local function doProjection(obj,field,ctxt)
    assert(L.is_field(field))
    local dataptr = ctxt:FieldPtr(field)
    return `dataptr[obj]
end

function ast.GenericFor:codegen (ctxt)
    local set       = self.set:codegen(ctxt)
    local iter      = symbol("iter")
    local rel       = self.set.node_type.relation
    local projected = iter

    for i,p in ipairs(self.set.node_type.projections) do
        local field = rel[p]
        projected   = doProjection(projected,field,ctxt)
        rel         = field.type.relation
        assert(rel)
    end
    local sym = symbol(L.row(rel):terraType())
    ctxt:enterblock()
        ctxt:localenv()[self.name] = sym
        local body = self.body:codegen(ctxt)
    ctxt:leaveblock()
    local code = quote
        var s = [set]
        for [iter] = s.start,s.finish do
            var [sym] = [projected]
            [body]
        end
    end
    return code
end


----------------------------------------------------------------------------

function ast.DeleteStatement:codegen (ctxt)
  local relation  = self.row.node_type.relation

  local row       = self.row:codegen(ctxt)
  local live_mask = ctxt:FieldPtr(relation._is_live_mask)
  local set_mask_stmt = quote live_mask[row] = false end

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


