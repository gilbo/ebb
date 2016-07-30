-- The MIT License (MIT)
-- 
-- Copyright (c) 2015 Stanford University.
-- All rights reserved.
-- 
-- Permission is hereby granted, free of charge, to any person obtaining a
-- copy of this software and associated documentation files (the "Software"),
-- to deal in the Software without restriction, including without limitation
-- the rights to use, copy, modify, merge, publish, distribute, sublicense,
-- and/or sell copies of the Software, and to permit persons to whom the
-- Software is furnished to do so, subject to the following conditions:
-- 
-- The above copyright notice and this permission notice shall be included
-- in all copies or substantial portions of the Software.
-- 
-- THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
-- IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
-- FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
-- AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
-- LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
-- FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
-- DEALINGS IN THE SOFTWARE.

local UF   = {}
package.loaded["ebb.src.ufversions"] = UF

local use_exp    = not not rawget(_G, 'EBB_USE_EXPERIMENTAL_SIGNAL')
local use_single = not use_exp

local Pre   = require "ebb.src.prelude"
local C     = require "ebb.src.c"
local G     = require "ebb.src.gpu_util"
local T     = require "ebb.src.types"
local Util  = require 'ebb.src.util'

local CPU       = Pre.CPU
local GPU       = Pre.GPU
local keyT      = T.key

local codegen         = require "ebb.src.codegen"
local codesupport     = require "ebb.src.codegen_support"
local DataArray       = require('ebb.src.rawdata').DataArray

local ewrap     = use_exp and require 'ebb.src.ewrap'

local R         = require 'ebb.src.relations'
local F         = require 'ebb.src.functions'
local UFunc     = F.Function
local UFVersion = F.UFVersion
local _INTERNAL_DEV_OUTPUT_PTX = F._INTERNAL_DEV_OUTPUT_PTX

local newlist = terralib.newlist

-------------------------------------------------------------------------------
-------------------------------------------------------------------------------
--[[ Terra Signature                                                       ]]--
-------------------------------------------------------------------------------
-------------------------------------------------------------------------------

local struct bounds_struct { lo : int64, hi : int64 }
local exp_field_struct = {}
for d=1,3 do
  exp_field_struct[d] = terralib.types.newstruct('exp_field_struct_'..d)
  exp_field_struct[d].entries:insertall {
    { 'ptr',        &opaque },
    { 'strides',    int64[d] },
  }
  exp_field_struct[d]:complete()
end

--[[
args = {
  relation,     -- relation being executed over
  use_subset,   -- boolean
  name,         -- name for struct
  fields,       -- list of fields
  globals,      -- list of globals
  global_reductions,  -- list of globals to reduce
}
--]]
local function BuildTerraSignature(args)
  local arg_keyT        = keyT(args.relation):terratype()
  local n_dims          = #args.relation:Dims()

  local name = (args.name or '') .. '_terra_signature'
  local terrasig = terralib.types.newstruct(name)
  local _field_num,             f_count = {}, 0
  local _global_num,            g_count = {}, 0
  local _global_reduction_num,  r_count = {}, 0
  local field_names             = {}
  local global_names            = {}
  local global_reduction_names  = {}

  -- add counter
  terrasig.entries:insert{field='bounds', type=(bounds_struct[n_dims])}
  -- add subset data
  if args.use_subset then -- make sure it's available
    terrasig.entries:insert{field='index',      type=&arg_keyT}
    terrasig.entries:insert{field='index_size', type=int64}
  end
  -- add fields
  for i,f in ipairs(args.fields) do
    _field_num[f] = i
    local name = 'field_'..i..'_'..string.gsub(f:FullName(),'%W','_')
    field_names[f] = name
    if use_single then
      terrasig.entries:insert{ field=name, type=&( f:Type():terratype() ) }
    elseif use_exp then
      local f_dims = #f:Relation():Dims()
      terrasig.entries:insert { field=name, type=exp_field_struct[f_dims] }
    end
  end
  -- add globals
  for i,g in ipairs(args.globals) do
    _global_num[g] = i
    local name = 'global_'..i..'_'..
                 string.gsub(tostring(g:Type()),'%W','_')
    global_names[g] = name
    terrasig.entries:insert{ field=name, type=&( g:Type():terratype() ) }
  end
  -- add global reductions (possibly secondary location)
  for i,gr in ipairs(args.global_reductions) do
    _global_reduction_num[gr] = i
    local name = 'reduce_global_'..i..'_'..
                 string.gsub(tostring(gr:Type()),'%W','_')
    global_reduction_names[gr] = name
    terrasig.entries:insert{ field=name, type=&( gr:Type():terratype() ) }
  end

  terrasig:complete()
  terrasig._field_num             = _field_num
  terrasig._global_num            = _global_num
  terrasig._global_reduction_num  = _global_reduction_num

  function terrasig.luaget(sig, fg)
    if        R.is_field(fg) then return sig[field_names[fg]]
    elseif Pre.is_global(fg) then return sig[global_names[fg]]
    else error('luaget() expects a field or global') end
  end
  function terrasig.luaget_reduction(sig, g)
    return sig[global_reduction_names[g]]
  end
  function terrasig.luaset(sig, fg, val)
    if        R.is_field(fg) then sig[field_names[fg]] = val
    elseif Pre.is_global(fg) then sig[global_names[fg]] = val
    else error('luaset() expects a field or global') end
  end
  function terrasig.luaset_reduction(sig, g, val)
    sig[global_reduction_names[g]] = val
  end
  function terrasig.terraptr(sig, fg)
    if        R.is_field(fg) then
      local name = field_names[fg]
      return `sig.[name]
    elseif Pre.is_global(fg) then
      local name = global_names[fg]
      return `sig.[name]
    else
      error('terraptr() expects a field or global')
    end
  end
  function terrasig.terraptr_reduction(sig, g)
    local name = global_reduction_names[g]
    return `sig.[name]
  end

  return terrasig
end



-------------------------------------------------------------------------------
-------------------------------------------------------------------------------
--[[ UFVersion                                                             ]]--
-------------------------------------------------------------------------------
-------------------------------------------------------------------------------

function UFVersion:Execute(exec_args)
  exec_args = exec_args or {}

  if not self:isCompiled() then
    self:Compile()
  end

  UFVersion._total_function_launch_count:increment()

  self._exec_timer:start()

  -- Regardless of the caching, we need to make sure
  -- that the current state of the system and relevant data safely allows
  -- us to launch the UFunc.  This might include checks to make sure that
  -- all of the relevant data does in fact exist, and that any invariants
  -- we assumed when the UFunc was compiled and cached are still true.
  self:_DynamicChecks()

  -- Next, we bind all the necessary data into the version.
  -- This involves looking up appropriate pointers, argument values,
  -- data location, CUDA parameters, and packing
  -- appropriate structures
  self:_BindData()

  --local prelaunch = terralib.currenttimeinseconds()
  -- Then, once all the data is bound and marshalled, we
  -- can actually launch the computation.  Oddly enough, this
  -- may require some amount of further marshalling and binding
  -- of data depending on what runtime this version is being launched on.
  self:_Launch(exec_args)
  --LAUNCH_TIMER = LAUNCH_TIMER + (terralib.currenttimeinseconds() - prelaunch)

  -- Finally, some features may require some post-processing after the
  -- launch of the UFunc.  This hook provides a place for
  -- any such computations.
  self:_PostLaunchCleanup()

  self._exec_timer:stop()
end

-- Define ways of inspecting high-level UFVersion state/modes
function UFVersion:isCompiled()
  return nil ~= self._executable
end
function UFVersion:UsesInsert()
  return nil ~= self._insert_data
end
function UFVersion:UsesDelete()
  return nil ~= self._delete_data
end
function UFVersion:UsesGlobalReduce()
  return next(self._global_reductions) ~= nil
end
function UFVersion:onGPU()
  return self._proc == GPU
end
function UFVersion:overElasticRelation()
  return self._is_elastic
end
function UFVersion:isOverSubset()
  return nil ~= self._subset
end
function UFVersion:isBoolMaskSubset()
  return nil ~= self._subset._boolmask
end
function UFVersion:isIndexSubset()
  return nil ~= self._subset._index
end


--                  ---------------------------------------                  --
--[[ UF Compilation                                                        ]]--
--                  ---------------------------------------                  --


function UFVersion:Compile()
  self._compile_timer:start()

  local typed_ast   = self._typed_ast

  -- Build Signatures defining interface boundaries for
  -- constructing various kinds of task wrappers
  self:_CompileTerraSignature()
  -- NOTE: CompileTerraSignature also updates field uses and global uses
  -- For instance, add boolmask field to _field_use if over subset.
  if use_exp    then self:_CompileExpSignature() end

  -- handle GPU specific compilation
  if self:onGPU() and self:UsesGlobalReduce() then
    self._sharedmem_size = 0
    self:_CompileGPUReduction()
  end

  if use_single then
    -- allocate memory for the arguments struct on the CPU.  It will be used
    -- to hold the parameter values that will be passed to the Ebb function.
    self._args = DataArray.New{
      size = 1,
      type = self._terra_signature,
      processor = CPU, -- DON'T MOVE
    }
    
    -- compile an executable
    self._executable = codegen.codegen(typed_ast, self)

  elseif use_exp then
    self:_CompileExpAndGetLauncher(typed_ast)
  else
    error("INTERNAL: IMPOSSIBLE BRANCH")
  end

  self._compile_timer:stop()
end

function UFVersion:_CompileTerraSignature()
  local fields, globals   = newlist(), newlist()
  self._global_reductions = {}
  local global_reductions = newlist()

  for f, use in pairs(self._field_use) do
    fields:insert(f)
  end

  for g, phase in pairs(self._global_use) do
    globals:insert(g)
    if phase.reduceop then
      self._global_reductions[g] = {
        phase = phase,
      }
      self._uses_global_reduce = true
      global_reductions:insert(g)
    end
  end

  self._terra_signature = BuildTerraSignature{
    relation          = self._relation,
    use_subset        = self:isOverSubset(),
    name              = self._name,
    fields            = fields,
    field_use         = self._field_use,
    globals           = globals,
    global_reductions = global_reductions,
  }
end

--                  ---------------------------------------                  --
--[[ UFVersion Interface for Codegen / Compilation                         ]]--
--                  ---------------------------------------                  --

function UFVersion:_argsType ()
  return self._terra_signature
end

function UFVersion:_getReduceData(global)
  local data = self._global_reductions[global]
  return assert(self._global_reductions[global],
                'reduction was not predeclared')
end

function UFVersion:_setFieldPtr(field)
  if use_exp then
    error('INTERNAL: Do not call _setFieldPtr() when using exp') end
  self._terra_signature.luaset(self._args:_raw_ptr(),
                               field,
                               field:_Raw_DataPtr())
end
function UFVersion:_setGlobalPtr(global)
  if use_exp then
    error('INTERNAL: Do not call _setGlobalPtr() when using exp') end
  self._terra_signature.luaset(self._args:_raw_ptr(),
                               global,
                               global:_Raw_DataPtr())
end

function UFVersion:_getTerraField(args_sym, field)
  return self._terra_signature.terraptr(args_sym, field)
end
function UFVersion:_getTerraGlobalPtr(args_sym, global)
  return self._terra_signature.terraptr(args_sym, global)
end
function UFVersion:_getTerraGreductionPtr(args_sym, global)
  return self._terra_signature.terraptr_reduction(args_sym, global)
end


--                  ---------------------------------------                  --
--[[ UFVersion Dynamic Checks                                              ]]--
--                  ---------------------------------------                  --

function UFVersion:_DynamicChecks()
  if use_single then
    -- Check that the fields are resident on the correct processor
    local underscore_field_fail = nil
    for field, _ in pairs(self._field_use) do
      if field._array:location() ~= self._proc then
        if field:Name():sub(1,1) == '_' then
          underscore_field_fail = field
        else
          error("cannot execute function because field "..field:FullName()..
                " is not currently located on "..tostring(self._proc), 3)
        end
      end
    end
    if underscore_field_fail then
      error("cannot execute function because hidden field "..
            underscore_field_fail:FullName()..
            " is not currently located on "..tostring(self._proc), 3)
    end
  end

  if self:UsesInsert()  then  self:_DynamicInsertChecks()  end
  if self:UsesDelete()  then  self:_DynamicDeleteChecks()  end
end


--                  ---------------------------------------                  --
--[[ UFVersion Data Binding                                                ]]--
--                  ---------------------------------------                  --

function UFVersion:_BindData()
  -- Bind inserts and deletions before anything else, because
  -- the binding may trigger computations to re-size/re-allocate
  -- data in some cases, invalidating previous data pointers
  if self:UsesInsert()  then  self:_bindInsertData()       end
  if self:UsesDelete()  then  self:_bindDeleteData()       end

  -- Bind the rest of the data
  self:_bindFieldGlobalSubsetArgs()
end

function UFVersion:_bindFieldGlobalSubsetArgs()
  -- Don't worry about binding on distributed, since we need
  -- to handle that a different way anyways
  if use_exp then return end

  local argptr    = self._args:_raw_ptr()

  -- Case 1: subset indirection index
  if self._subset and self._subset._index then
    argptr.index        = self._subset._index:_Raw_DataPtr()
    -- Spoof the number of entries in the index, which is what
    -- we actually want to iterate over
    argptr.bounds[0].lo = 0
    argptr.bounds[0].hi = self._subset._index:Size() - 1 

  -- Case 2: elastic relation
  elseif self:overElasticRelation() then
    argptr.bounds[0].lo = 0
    argptr.bounds[0].hi = self._relation:ConcreteSize() - 1

  -- Case 3: generic staticly sized relation
  else
    local dims = self._relation:Dims()
    for d=1,#dims do
      argptr.bounds[d-1].lo = 0
      argptr.bounds[d-1].hi = dims[d] - 1
    end

  end

  -- set field and global pointers
  for field, _ in pairs(self._field_use) do
    self:_setFieldPtr(field)
  end
  for globl, _ in pairs(self._global_use) do
    self:_setGlobalPtr(globl)
  end
end

--                  ---------------------------------------                  --
--[[ UFVersion Launch                                                      ]]--
--                  ---------------------------------------                  --

function UFVersion:_Launch(exec_args)
  if use_exp then
    self._executable()
  else
    self._executable(self._args:_raw_ptr())
  end
end

--                  ---------------------------------------                  --
--[[ UFVersion Postprocess / Cleanup                                       ]]--
--                  ---------------------------------------                  --

function UFVersion:_PostLaunchCleanup()
  -- GPU Reduction finishing and cleanup
  --if self:onGPU() then
  --  if self:UsesGlobalReduce() then  self:postprocessGPUReduction()  end
  --end

  -- Handle post execution Insertion and Deletion Behaviors
  if self:UsesInsert()         then   self:_postprocessInsertions()    end
  if self:UsesDelete()         then   self:_postprocessDeletions()     end
end


-------------------------------------------------------------------------------
-------------------------------------------------------------------------------
--[[ Insert / Delete Extensions                                            ]]--
-------------------------------------------------------------------------------
-------------------------------------------------------------------------------


--                  ---------------------------------------                  --
--[[ Insert Processing ; all 4 stages (-launch)                            ]]--
--                  ---------------------------------------                  --

function UFVersion:_DynamicInsertChecks()
  if use_exp then error('INSERT unsupported on exp currently', 4) end

  local rel = self._insert_data.relation
  local unsafe_msg = rel:UnsafeToInsert(self._insert_data.record_type)
  if unsafe_msg then error(unsafe_msg, 4) end
end

function UFVersion:_bindInsertData()
  local insert_rel                    = self._insert_data.relation
  local center_size_logical           = self._relation:Size()
  local insert_size_concrete          = insert_rel:ConcreteSize()
  local insert_size_logical           = insert_rel:Size()
  --print('INSERT BIND',
  --  center_size_logical, insert_size_concrete, insert_size_logical)

  -- point the write index at the first entry after the end of the
  -- used portion of the data arrays
  self._insert_data.write_idx:set(insert_size_concrete)
  -- cache the old sizes
  self._insert_data.last_concrete_size = insert_size_concrete
  self._insert_data.last_logical_size  = insert_size_logical

  -- then make sure to reserve enough space to perform the insertion
  -- don't worry about updating logical size here
  insert_rel:_INTERNAL_Resize(insert_size_concrete + center_size_logical)
end

function UFVersion:_postprocessInsertions()
  local insert_rel        = self._insert_data.relation
  local old_concrete_size = self._insert_data.last_concrete_size
  local old_logical_size  = self._insert_data.last_logical_size

  local new_concrete_size = tonumber(self._insert_data.write_idx:get())
  local n_inserted        = new_concrete_size - old_concrete_size
  local new_logical_size  = old_logical_size + n_inserted
  --print("POST INSERT",
  --  old_concrete_size, old_logical_size, new_concrete_size,
  --  n_inserted, new_logical_size)

  -- shrink array back to fit how much we actually wrote
  insert_rel:_INTERNAL_Resize(new_concrete_size, new_logical_size)

  -- NOTE that this relation is now considered fragmented
  -- (change this?)
  insert_rel:_INTERNAL_MarkFragmented()
end

--                  ---------------------------------------                  --
--[[ Delete Processing ; all 4 stages (-launch)                            ]]--
--                  ---------------------------------------                  --

function UFVersion:_DynamicDeleteChecks()
  if use_exp then error('DELETE unsupported on exp currently', 4) end

  local unsafe_msg = self._delete_data.relation:UnsafeToDelete()
  if unsafe_msg then error(unsafe_msg, 4) end
end

function UFVersion:_bindDeleteData()
  local relsize = tonumber(self._delete_data.relation._logical_size)
  self._delete_data.n_deleted:set(0)
end

function UFVersion:_postprocessDeletions()
  -- WARNING UNSAFE CONVERSION FROM UINT64 TO DOUBLE
  local rel = self._delete_data.relation
  local n_deleted     = tonumber(self._delete_data.n_deleted:get())
  local updated_size  = rel:Size() - n_deleted
  local concrete_size = rel:ConcreteSize()
  rel:_INTERNAL_Resize(concrete_size, updated_size)
  rel:_INTERNAL_MarkFragmented()

  -- if we have too low an occupancy
  if updated_size < 0.5 * concrete_size then
    rel:Defrag()
  end
end


-------------------------------------------------------------------------------
-------------------------------------------------------------------------------
--[[ GPU Extensions     (Mainly Global Reductions)                         ]]--
-------------------------------------------------------------------------------
-------------------------------------------------------------------------------

-- The following are mainly support routines related to the GPU
-- A lot of them (identified in their names) are
-- strictly related to reductions, and may be used both
-- within the codegen compilation and the compilation of a secondary
-- CUDA Kernel (below)

function UFVersion:_numGPUBlocks(argptr)
  if self:overElasticRelation() then
    local size    = `argptr.bounds[0].hi - argptr.bounds[0].lo + 1
    local nblocks = `[uint64]( C.ceil( [double](size) /
                                       [double](self._blocksize) ))
    return nblocks
  else
    if self:isOverSubset() and self:isIndexSubset() then
      return math.ceil(self._subset._index:Size() / self._blocksize)
    else
      local size = `1
      for d = 1, #self._relation:Dims() do
          size = `((size) * (argptr.bounds[d-1].hi - argptr.bounds[d-1].lo + 1))
      end
      return `[uint64](C.ceil( [double](size) / [double](self._blocksize)))
    end
  end
end

function UFVersion:_nBytesSharedMem()
  return self._sharedmem_size or 0
end

function UFVersion:_getBlockSize()
  return self._blocksize
end

function UFVersion:_getTerraReduceGlobalMemPtr(args_sym, global)
  return self:_getTerraGreductionPtr(args_sym, global)
  --local data = self:_getReduceData(global)
  --return `[args_sym].[data.id]
end

function UFVersion:_getTerraReduceSharedMemPtr(global)
  local data = self:_getReduceData(global)

  if self._useTreeReduce then
    return data.sharedmem
  else
    return data.reduceobj:getSharedMemPtr()
  end
end

--                  ---------------------------------------                  --
--[[ GPU Reduction Compilation                                             ]]--
--                  ---------------------------------------                  --

function UFVersion:_CompileGPUReduction()
  self._useTreeReduce = true
  -- NOTE: because GPU memory is idiosyncratic, we need to handle
  --    GPU global memory and
  --    GPU shared memory differently.
  --  Specifically,
  --    * we handle the global memory in the same way we handle
  --      field and global data; by adding an entry into
  --      the argument structure, binding appropriate allocated data, etc.
  --    * we handle the shared memory via a mechanism that looks more
  --      like Terra globals.  As such, these "shared memory pointers"
  --      get inlined directly into the Terra code.  This is safe because
  --      the CUDA kernel launch, not the client CPU code, is responsible
  --      for allocating and deallocating shared memory on function launch/exit

  -- Find all the global variables in this function that are being reduced
  for globl, data in pairs(self._global_reductions) do
    local ttype             = globl._type:terratype()
    if self._useTreeReduce then
      data.sharedmem          = cudalib.sharedmemory(ttype, self._blocksize)
  
      self._sharedmem_size    = self._sharedmem_size +
                                  sizeof(ttype) * self._blocksize
    else
      local op      = data.phase.reduceop
      local ebbtype = globl._type
      local reduceobj = G.ReductionObj.New {
        ttype             = ttype,
        blocksize         = self._blocksize,
        reduce_ident      = codesupport.reduction_identity(ebbtype, op),
        reduce_binop      = function(lval, rhs)
          return codesupport.reduction_binop(ebbtype, op, lval, rhs)
        end,
        gpu_reduce_atomic = function(lval, rhs)
          return codesupport.gpu_atomic_exp(op, ebbtype, lval, rhs, ebbtype)
        end,
      }
      data.reduceobj = reduceobj
      self._sharedmem_size = self._sharedmem_size + reduceobj:sharedMemSize()
    end
  end

  if self._useTreeReduce then
    self:_CompileGlobalMemReductionKernel()
  end
end

-- The following routine is also used inside the primary compile CUDA kernel
function UFVersion:_GenerateSharedMemInitialization(tid_sym)
  local code = quote end
  for globl, data in pairs(self._global_reductions) do
    local op        = data.phase.reduceop
    local lz_type   = globl._type
    local sharedmem = data.sharedmem

    if self._useTreeReduce then
      code = quote
        [code]
        [sharedmem][tid_sym] = [codesupport.reduction_identity(lz_type, op)]
      end
    else
      code = quote
        [code]
        [data.reduceobj:sharedMemInitCode(tid_sym)]
      end
    end
  end
  return code
end

-- The following routine is also used inside the primary compile CUDA kernel
function UFVersion:_GenerateSharedMemReduceTree(
  args_sym, tid_sym, bid_sym, is_final
)
  is_final = is_final or false
  local code = quote end
  for globl, data in pairs(self._global_reductions) do
    local op          = data.phase.reduceop
    local lz_type     = globl._type
    local sharedmem   = data.sharedmem
    local finalptr    = self:_getTerraGlobalPtr(args_sym, globl)
    local globalmem   = self:_getTerraReduceGlobalMemPtr(args_sym, globl)

    -- Insert an unrolled reduction tree here
    if self._useTreeReduce then
      local step = self._blocksize
      while step > 1 do
        step = step/2
        code = quote
          [code]
          if tid_sym < step then
            var exp = [codesupport.reduction_binop(
                        lz_type, op, `[sharedmem][tid_sym],
                                     `[sharedmem][tid_sym + step])]
            terralib.attrstore(&[sharedmem][tid_sym], exp, {isvolatile=true})
          end
          G.barrier()
        end
      end

      -- Finally, reduce into the actual global value
      code = quote
        [code]
        if [tid_sym] == 0 then
          if is_final then
            @[finalptr] = [codesupport.reduction_binop(lz_type, op,
                                                       `@[finalptr],
                                                       `[sharedmem][0])]
          else
            [globalmem][bid_sym] = [sharedmem][0]
          end
        end
      end
    else
      code = quote
        [code]
        [data.reduceobj:sharedMemReductionCode(tid_sym, finalptr)]
      end
    end
  end
  return code
end

-- The full secondary CUDA kernel to reduce the contents of the
-- global mem array.  See comment inside function for sketch of algorithm
function UFVersion:_CompileGlobalMemReductionKernel()
  local ufv       = self
  local fn_name   = ufv._ufunc._name .. '_globalmem_reduction'

  -- Let N be the number of rows in the original relation
  -- and B be the block size for both the primary and this (the secondary)
  --          cuda kernels
  -- Let M = CEIL(N/B) be the number of blocks launched in the primary
  --          cuda kernel
  -- Then note that there are M entries in the globalmem array that
  --  need to be reduced.  We assume that the primary cuda kernel filled
  --  in a correct value for each of these.
  -- The secondary kernel will launch exactly one block with B threads.
  --  First we'll reduce all of the M entries in the globalmem array in
  --  chunks of B values into a sharedmem buffer.  Then we'll do a local
  --  tree reduction on those B values.
  -- NOTE EDGE CASE: What if M < B?  Then we'll initialize the shared
  --  buffer to an identity value and fail to execute the loop iteration
  --  for the trailing B-M threads of the block.  (This is memory safe)
  --  We will get the correct values b/c reducing identities has no effect.
  local args      = symbol(ufv:_argsType())
  local array_len = symbol(uint64)
  local tid       = symbol(uint32)
  local bid       = symbol(uint32)

  local cuda_kernel =
  terra([array_len], [args])
    var [tid]             = G.thread_id()
    var [bid]             = G.block_id()
    var n_blocks : uint32 = G.num_blocks()
    var gt                = tid + [ufv._blocksize] * bid
    
    -- INITIALIZE the shared memory
    [ufv:_GenerateSharedMemInitialization(tid)]
    
    -- REDUCE the global memory into the provided shared memory
    -- count from (gt) till (array_len) by step sizes of (blocksize)
    for gi = gt, array_len, n_blocks * [ufv._blocksize] do
      escape for globl, data in pairs(ufv._global_reductions) do
        local op          = data.phase.reduceop
        local lz_type     = globl._type
        local sharedmem   = data.sharedmem
        local globalmem   = ufv:_getTerraReduceGlobalMemPtr(args, globl)

        emit quote
          [sharedmem][tid]  = [codesupport.reduction_binop(lz_type, op,
                                                           `[sharedmem][tid],
                                                           `[globalmem][gi])]
        end
      end end
    end

    G.barrier()
  
    -- REDUCE the shared memory using a tree
    [ufv:_GenerateSharedMemReduceTree(args, tid, bid, true)]
  end
  cuda_kernel:setname(fn_name)
  cuda_kernel = G.kernelwrap(cuda_kernel, _INTERNAL_DEV_OUTPUT_PTX)

  -- the globalmem array has an entry for every block in the primary kernel
  local terra launcher( argptr : &(ufv:_argsType()) )
    var globalmem_array_len = [ ufv:_numGPUBlocks(argptr) ]
    var launch_params = terralib.CUDAParams {
      1,1,1, [ufv._blocksize],1,1, [ufv._sharedmem_size], nil
    }
    cuda_kernel(&launch_params, globalmem_array_len, @argptr )
  end
  launcher:setname(fn_name..'_launcher')

  ufv._global_reduction_pass = launcher
end

--                  ---------------------------------------                  --
--[[ GPU Reduction Dynamic Checks                                          ]]--
--                  ---------------------------------------                  --

function UFVersion:_DynamicGPUReductionChecks()
  if self._proc ~= GPU then
    error("INTERNAL ERROR: Should only try to run GPUReduction on the GPU...")
  end
end


--                  ---------------------------------------                  --
--[[ GPU Reduction Data Binding                                            ]]--
--                  ---------------------------------------                  --

function UFVersion:_generateGPUReductionPreProcess(argptrsym)
  if not self._useTreeReduce then return quote end end
  if not self:UsesGlobalReduce() then return quote end end

  -- allocate GPU global memory for the reduction
  return quote
    var n_blocks = [self:_numGPUBlocks(argptrsym)]
    escape for globl, _ in pairs(self._global_reductions) do
      local ttype     = globl._type:terratype()
      local reduceptr = self:_getTerraGreductionPtr(argptrsym, globl)
      emit quote
        [reduceptr] = [&ttype](G.malloc(sizeof(ttype) * n_blocks))
      end
    end end
  end
end


--                  ---------------------------------------                  --
--[[ GPU Reduction Postprocessing                                          ]]--
--                  ---------------------------------------                  --

function UFVersion:_generateGPUReductionPostProcess(argptrsym)
  if not self._useTreeReduce then return quote end end
  if not self:UsesGlobalReduce() then return quote end end
  
  -- perform inter-block reduction step (secondary kernel launch)
  local second_pass = self._global_reduction_pass
  local code = quote
    second_pass(argptrsym)
  end

  -- free GPU global memory allocated for the reduction
  for globl, _ in pairs(self._global_reductions) do
    local reduceptr = self:_getTerraGreductionPtr(argptrsym, globl)
    code = quote code
      G.free( [reduceptr] )
      [reduceptr] = nil -- just to be safe
    end
  end
  return code
end





-------------------------------------------------------------------------------
-------------------------------------------------------------------------------
--[[ Experimental Signature                                                ]]--
-------------------------------------------------------------------------------
-------------------------------------------------------------------------------

-- convert record phase to access privilege
local function phase_to_exp_privilege(phase)
  assert(not phase:iserror(), 'INTERNAL: phase should not be in error')
  if     phase:isReadOnly() then  return ewrap.READ_ONLY_PRIVILEGE
  elseif phase:isCentered() then  return ewrap.READ_WRITE_PRIVILEGE
  else                            return ewrap.REDUCE_PRIVILEGE     end
end

local get_exp_reduceop = Util.memoize(function(op_str, ebb_type)
  local tt = ebb_type:terratype()
  local terra reduction( accptr : &uint8, valptr : &uint8 )
    var acc     = [&tt](accptr)
    var val     = @[&tt](valptr)
    @acc = [codesupport.reduction_binop(ebb_type, op_str, `@acc, val)]
  end
  local e_reduce_op = ewrap.NewReduceOp {
    func = reduction,
    name = 'reduce_'..op_str..'_'..tostring(ebb_type),
  }
  return e_reduce_op
end)

--[[
{
  field_accesses
  field_use
  global_use
  relation
}
--]]
-- This signature helps establish a canonical argument ordering
local function BuildExpSignature(params)
  if not use_exp then
    error('INTERNAL: Should only call '..
          'BuildExpSignature() when running on experimental runtime')
  end


  -- get primary relation
  local prim_relation   = params.relation
  local prim_e_relation = prim_relation._ewrap_relation

  -- get field accesses
  local field_list      = newlist()
  local faccess_list    = newlist()
  local faccess_map     = {}
  for field, phase in pairs(params.field_use) do
    assert(params.field_accesses[field],
           "There is no recorded field access for field : " .. field:Name())
    local faccess = ewrap.NewFAccess{
      field     = field._ewrap_field,
      privilege = phase_to_exp_privilege(phase)
    }

    field_list:insert(field)
    faccess_map[field] = faccess
    faccess_list:insert(faccess)
  end
  -- sanity check
  for field,_ in pairs(params.field_accesses) do
    assert(params.field_use[field],
           "There is no recorded field use for field : " .. field:Name())
  end

  -- get global accesses
  local global_list     = newlist()
  local global_map      = {}
  local gaccess_list    = newlist()
  for globl, phase in pairs(params.global_use) do
    local priv      = phase_to_exp_privilege(phase)
    local reduceop  = nil
    if priv == ewrap.REDUCE_PRIVILEGE then
      reduceop = get_exp_reduceop(phase:reductionOp(), globl:Type())
    end
    local gaccess = ewrap.NewGAccess {
      global    = globl._ewrap_global,
      privilege = priv,
      reduceop  = reduceop,
    }

    global_list:insert(globl)
    global_map[globl] = gaccess
    gaccess_list:insert(gaccess)
  end


  local ExpSignature = {}

  function ExpSignature:GetFAccessList()
    return faccess_list
  end
  function ExpSignature:GetFieldList()
    return field_list
  end
  function ExpSignature:GetGAccessList()
    return gaccess_list
  end
  function ExpSignature:GetGlobalList()
    return global_list
  end

  return ExpSignature
end

function UFVersion:_CompileExpSignature()
  self._exp_signature = BuildExpSignature {
    field_use       = self._field_use,
    global_use      = self._global_use,
    relation        = self._relation,
    field_accesses  = self._field_accesses,
  }
end


-------------------------------------------------------------------------------
-------------------------------------------------------------------------------
--[[ Experimental Mode Extensions                                          ]]--
-------------------------------------------------------------------------------
-------------------------------------------------------------------------------

function UFVersion:_CompileExpAndGetLauncher(typed_ast)
  local task_function     = codegen.codegen(typed_ast, self)
  self._executable        = self:_CreateExpLauncher(task_function)
end


function UFVersion:_CreateExpWrappedTask(task_func)
  local ufv = self

  -- sanity checks
  --assert( not ufv._subset, 'EXP TODO: Support Subsets' )

  -- unpack things
  local n_dims  = #ufv._relation:Dims()
  local tSig    = ufv:_argsType()

  local terra wrapped_func( e_args : ewrap.TaskArgs )
    -- translate args to terra signature
    var t_args : tSig

    -- re-pack iteration bounds
    for d = 0, n_dims do
      t_args.bounds[d].lo   = e_args.bounds[d].lo
      t_args.bounds[d].hi   = e_args.bounds[d].hi
    end

    -- re-pack field data
    escape for i,f in ipairs(ufv._exp_signature:GetFieldList()) do
      local tf = ufv:_getTerraField(t_args, f)
      emit quote tf.ptr = e_args.fields[i-1].ptr end
      for d = 1,n_dims do
        emit quote tf.strides[d-1] = e_args.fields[i-1].strides[d-1] end
      end
    end end

    -- re-pack global data
    escape for i,g in ipairs(ufv._exp_signature:GetGlobalList()) do
      local phase = ufv._global_use[g]
      local g_ptr = self:_getTerraGlobalPtr(t_args, g)
      local tt    = g:Type():terratype()
      emit quote [g_ptr]  = [&tt](e_args.globals[i-1].ptr) end
      if phase:isUncenteredReduction() then
        -- initialize temp area if doing a reduction
        local op = assert(phase:reductionOp(), 'expecting reduce op')
        emit quote
          @[g_ptr] = [codesupport.reduction_identity(g:Type(), op)]
        end
      else assert(phase:isReadOnly()) end
    end end

    task_func(&t_args)

    -- reduced globals will be unpacked and shipped back by ewrap
  end
  wrapped_func:setname(ufv._name .. '_task')

  return wrapped_func
end


-- Launches task and returns.
function UFVersion:_CreateExpLauncher(task_func)
  local ufv = self

  local exp_task_func = ufv:_CreateExpWrappedTask(task_func)

  -- register a new task object across the system
  local exp_task = ewrap.RegisterNewTask {
    func            = exp_task_func,
    name            = ufv._name,
    relation        = ufv._relation._ewrap_relation,
    processor       = ufv._proc,
    field_accesses  = ufv._exp_signature:GetFAccessList(),
    global_accesses = ufv._exp_signature:GetGAccessList(),
  }

  return function()
    exp_task:exec()
  end
end


