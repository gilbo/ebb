local Codegen = {}
package.loaded["compiler.codegen_single"] = Codegen

local ast = require "compiler.ast"
local T   = require "compiler.types"

local C   = require 'compiler.c'
local L   = require 'compiler.lisztlib'
local G   = require 'compiler.gpu_util'
local Support = require 'compiler.codegen_support'

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
--[[            Iteration / Dimension Abstraction Helpers               ]]--
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


--[[--------------------------------------------------------------------]]--
--[[                        Codegen Entrypoint                          ]]--
--[[--------------------------------------------------------------------]]--

function Codegen.codegen (kernel_ast, bran)
  local env  = terralib.newenvironment(nil)
  local ctxt = Context.New(env, bran)

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
              param = [ctxt:argsym()].index[linid]
            else -- use_boolmask
              param = addr
            end

            -- conditionally execute
            if use_index or [ctxt:argsym()].boolmask[linid] then
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
            escape if #ctxt:dims() > 1 then emit quote
              [terraIterNd({ nrow_sym }, nrow_sym, function(iter) return quote
                var [param] = [ctxt:argsym()].index[iter.a[0]]
                [body]
              end end)]
            end end end
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
--[[                       Codegen Pass Cases                           ]]--
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

-- DON'T CODEGEN A KERNEL DIRECTLY; HANDLE IN Codegen.codegen()
--function ast.LisztKernel:codegen (ctxt)
--end

function ast.Block:codegen (ctxt)
  -- start with an empty ast node, or we'll get an error when appending new quotes below
  local code = quote end
  for i = 1, #self.statements do
    local stmt = self.statements[i]:codegen(ctxt)
    code = quote
      [code]
      [stmt]
    end
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
    return quote for [itersym] = [minexp], [maxexp], [stepexp] do
      [body]
    end end
  else
    return quote for [itersym] = [minexp], [maxexp] do [body] end end
  end
end

function ast.Break:codegen(ctxt)
  return quote break end
end

function ast.Name:codegen(ctxt)
  local s = ctxt:localenv()[self.name]
  assert(terralib.issymbol(s))
  return `[s]
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
      [ Support.vec_mapgen(typ, function(i)
          return `[bt](vec.d[i])
      end) ] end

  elseif typ:isMatrix() then
    local mat = symbol(self.value.node_type:terraType())
    return quote var [mat] = valuecode in
      [ Support.mat_mapgen(typ, function(i,j)
          return `[bt](mat.d[i][j])
      end) ] end

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

  return Support.mat_mapgen(typ, function(i,j)
    return self.elems[i*self.m + j + 1]:codegen(ctxt)
  end)
end

function ast.VectorLiteral:codegen (ctxt)
  local typ = self.node_type

  return Support.vec_mapgen(typ, function(i)
    return self.elems[i+1]:codegen(ctxt)
  end)
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

  return Support.unary_exp(self.op, typ, expr)
end

function ast.BinaryOp:codegen (ctxt)
  local lhe = self.lhs:codegen(ctxt)
  local rhe = self.rhs:codegen(ctxt)

  return Support.bin_exp(self.op, self.node_type,
      lhe, rhe, self.lhs.node_type, self.rhs.node_type)
end

function ast.LuaObject:codegen (ctxt)
    return `{}
end

function ast.GenericFor:codegen (ctxt)
    local set       = self.set:codegen(ctxt)
    local iter      = symbol("iter")
    local rel       = self.set.node_type.relation
    -- the key being used to drive the where query should
    -- come from a grouped relation, which is necessarily 1d
    local projected = `[L.addr_terra_types[1]]({array([iter])})

    for i,p in ipairs(self.set.node_type.projections) do
        local field = rel[p]
        projected   = doProjection(projected,field,ctxt)
        rel         = field.type.relation
        assert(rel)
    end
    local sym = symbol(L.key(rel):terraType())
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

function ast.Assignment:codegen (ctxt)
  local lhs   = self.lvalue:codegen(ctxt)
  local rhs   = self.exp:codegen(ctxt)

  local ltype, rtype = self.lvalue.node_type, self.exp.node_type

  if self.reduceop then
    rhs = Support.bin_exp(self.reduceop, ltype, lhs, rhs, ltype, rtype)
  end
  return quote [lhs] = rhs end
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
    local rhs  = Support.bin_exp(self.reduceop, self.global.node_type,
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
    return Support.gpu_atomic_exp(self.reduceop,
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



