local K   = {}
package.loaded["compiler.kernel_single"] = K

local use_legion = not not rawget(_G, '_legion_env')
local use_single = not use_legion

local Kc  = require "compiler.kernel_common"
local L   = require "compiler.lisztlib"
local C   = require "compiler.c"
local G   = require "compiler.gpu_util"

local codegen         = require "compiler.codegen_single"
local codesupport     = require "compiler.codegen_support"
local LE, legion_env, LW
if use_legion then
  LE = rawget(_G, '_legion_env')
  legion_env = LE.legion_env:get()
  LW = require 'compiler.legionwrap'
end
local DataArray       = require('compiler.rawdata').DataArray


local Bran = Kc.Bran
local seedbank_lookup = Kc.seedbank_lookup



-- Create a Lua Object that generates the needed Terra structure to pass
-- fields, globals and temporary allocated memory to the kernel as arguments
local ArgLayout = {}
ArgLayout.__index = ArgLayout


-------------------------------------------------------------------------------
-------------------------------------------------------------------------------
--[[ Kernels                                                               ]]--
-------------------------------------------------------------------------------
-------------------------------------------------------------------------------

-- THIS IS THE BEGINNING OF CODE FOR INVOCATION OF A FUNCTION
-- both the first time it's invoked and subsequent times

function L.LKernel.__call(kobj, relset, params)
  if not (relset and (L.is_relation(relset) or L.is_subset(relset)))
  then
      error("A kernel must be called on a relation or subset.", 2)
  end

  -- When a kernel is called, the first thing we do is to
  -- retreive an appropriate "Bran" specialization of the kernel.
  -- This call will also ensure that all one-time
  -- processing on the Kernel/Bran is completed and cached
  local bran = kobj:GetBran(relset, params)

  -- However, regardless of the caching, we still need to make sure
  -- that we perform any dynamic checks before starting to launch
  -- the Kernel.  These might include checks to make sure that
  -- all of the relevant data does in fact exist, and that any
  -- invariants we assumed when the Bran construction was cached
  -- are still true.
  bran:DynamicChecks()

  -- Next, we bind all the necessary data into the Bran.
  -- This involves looking up appropriate pointers, argument values,
  -- data location, Legion or CUDA parameters, and packing
  -- appropriate structures
  bran:BindData()

  -- Finally, once all the data is bound and marshalled, we
  -- can actually launch the computation.  Oddly enough, this
  -- may require some amount of further marshalling and binding
  -- of data depending on what runtime this Bran is being launched on.
  -- launch the kernel
  bran:Launch()

  -- Finally, some features may require some post-processing after the
  -- launch of the Bran, so we need to check and allow
  -- any such computations
  bran:PostLaunchCleanup()

end

function L.LKernel:GetBran(relset, params)
  -- Make sure that we have a typed ast available
  if not self.typed_ast then
    self:TypeCheck()
  end

  local proc = L.default_processor

  -- retreive the correct bran or create a new one
  local sig  = {
    kernel=self,
    relset=relset,
    proc=proc,
  }
  if proc == L.GPU then
    sig.blocksize = (params and params.blocksize) or 64
  end
  local bran = Bran.CompileOrFetch(sig)

  return bran
end



-------------------------------------------------------------------------------
-------------------------------------------------------------------------------
--[[ Brans                                                                 ]]--
-------------------------------------------------------------------------------
-------------------------------------------------------------------------------


-- Define ways of inspecting high-level Bran state/modes
function Bran:IsCompiled()
  return nil ~= self.executable
end
function Bran:UsesInsert()
  return nil ~= self.insert_data
end
function Bran:UsesDelete()
  return nil ~= self.delete_data
end
function Bran:UsesGlobalReduce()
  return next(self.global_reductions) ~= nil
end
function Bran:isOnGPU()
  return self.proc == L.GPU
end
function Bran:overElasticRelation()
  return self.is_elastic
end
function Bran:isOverSubset()
  return nil ~= self.subset
end
function Bran:isBoolMaskSubset()
  return nil ~= self.subset._boolmask
end
function Bran:isIndexSubset()
  return nil ~= self.subset._index
end

--                  ---------------------------------------                  --
--[[ Bran Compilation                                                      ]]--
--                  ---------------------------------------                  --

function Bran.CompileOrFetch(sig)
  -- expand the cache signature
  -- a bit based on dynamic state and convenience
  if L.is_relation(sig.relset) then
    sig.relation  = sig.relset
  else
    sig.relation  = sig.relset:Relation()
    sig.subset    = sig.relset
  end
  sig.relset = nil
  sig.is_elastic  = sig.relation:isElastic()

  -- This call will either retreive a cached table
  -- or construct a new table with Bran set as a prototype
  local bran = seedbank_lookup(sig)

  if not bran:IsCompiled() then
    -- copy over all data from the signature table
    -- into the bran itself and then compile it
    for k,v in pairs(sig) do bran[k] = v end
    bran:Compile()
  end

  return bran
end

function Bran:Compile()
  local kernel    = self.kernel
  local typed_ast = self.kernel.typed_ast


  -- type checking the kernel signature against the invocation
  if typed_ast.relation ~= self.relation then
      error('Kernels may only be called on a relation they were typed with', 3)
  end

  self.arg_layout = ArgLayout.New()
  self.arg_layout:setNDims(self.relation:nDims())

  -- compile various kinds of data into the arg layout
  self:CompileFieldsGlobalsSubsets()

  -- also compile insertion and/or deletion if used
  if kernel.inserts then    self:CompileInserts()   end
  if kernel.deletes then    self:CompileDeletes()   end

  -- handle GPU specific compilation
  if self:isOnGPU() then
    if use_legion then error("LEGION UNSUPPORTED TODO") end
    self.sharedmem_size = 0
    self:CompileGPUReduction()
  end

  if use_single then
    -- allocate memory for the arguments struct on the CPU.  It will be used
    -- to hold the parameter values that will be passed to the Liszt kernel.
    self.args = DataArray.New{
      size = 1,
      type = self.arg_layout:TerraStruct(),
      processor = L.CPU -- DON'T MOVE
    }
    
    -- compile an executable
    self.executable = codegen.codegen(typed_ast, self)

  elseif use_legion then
    self:CompileLegion()
  else
    error("INTERNAL: IMPOSSIBLE BRANCH")
  end
end

function Bran:CompileFieldsGlobalsSubsets()
  -- initialize id structures
  self.field_ids    = {}
  self.n_field_ids  = 0

  self.global_ids   = {}
  self.n_global_ids = 0

  self.global_reductions = {}

  if use_legion then
    self.region_data        = {}
    self.sorted_region_data = {}
    self.n_regions          = 0

    self.future_nums  = {}
    self.n_futures    = 0

    self:getPrimaryRegionData()
  end

  -- reserve ids
  for field, _ in pairs(self.kernel.field_use) do
    self:getFieldId(field)
  end
  if self:overElasticRelation() then
    if use_legion then error("LEGION UNSUPPORTED TODO") end
    self:getFieldId(self.relation._is_live_mask)
  end
  if self:isOverSubset() then
    if self.subset._boolmask then
      self:getFieldId(self.subset._boolmask)
    end
  end
  for globl, phase in pairs(self.kernel.global_use) do
    local gid = self:getGlobalId(globl)

    -- record reductions
    if phase.reduceop then
      self.uses_global_reduce = true
      local ttype             = globl.type:terraType()

      local reduce_data       = self:getReduceData(globl)
      reduce_data.phase       = phase
    end
  end

  -- compile subsets in if appropriate
  if self.subset then
    self.arg_layout:turnSubsetOn()
  end
end

--                  ---------------------------------------                  --
--[[ Bran Interface for Codegen / Compilation                              ]]--
--                  ---------------------------------------                  --

function Bran:argsType ()
  return self.arg_layout:TerraStruct()
end

local function get_region_data(bran, relation, field)
  if not use_legion then
    error('INTERNAL: Should only try to record Regions '..
          'when running on the Legion Runtime')
  end
  -- NOTE WE create a new region data for each region/field pair
  local sig = tostring(relation:_INTERNAL_UID())
  if field then sig = sig ..'_'..tostring(field.fid) end
  local reg_data    = bran.region_data[sig]
  if reg_data then return reg_data
  else
    if bran.arg_layout:isCompiled() then
      error('INTERNAL ERROR: cannot add region after compiling \n'..
            '  argument layout.  (debug data follows)\n'..
            '      violating relation: '..relation:Name())
    end
    
    local reg_data = {
      wrapper   = relation._logical_region_wrapper,
      num       = bran.n_regions,
      --relation  = relation,
    }
    bran.n_regions = bran.n_regions + 1

    bran.region_data[sig]                 = reg_data
    bran.sorted_region_data[reg_data.num] = reg_data
    return reg_data
  end
end

function Bran:getPrimaryRegionData()
  if use_single then error("INTERNAL: Cannot use regions w/o Legion") end
  return get_region_data(self, self.relation)
end

function Bran:getRegionData(field)
  if use_single then error("INTERNAL: Cannot use regions w/o Legion") end
  local rel         = field:Relation()
  return get_region_data(self, rel, field)
end

function Bran:getFutureNum(globl)
  if use_single then error("INTERNAL: Cannot use futures w/o Legion") end
  local fut_num     = self.future_nums[globl]
  if fut_num then return fut_num
  else
    if self.arg_layout:isCompiled() then
      error('INTERNAL ERROR: cannot add future after compiling '..
            'argument layout.')
    end

    fut_num         = self.n_futures
    self.n_futures  = self.n_futures + 1

    self.future_nums[globl] = fut_num
    return fut_num
  end
end

function Bran:getFieldId(field)
  local id = self.field_ids[field]
  if id then return id
  else
    id = 'field_'..tostring(self.n_field_ids)..'_'..field:Name()
    self.n_field_ids = self.n_field_ids+1

    if use_legion then self:getRegionData(field) end

    self.field_ids[field] = id
    self.arg_layout:addField(id, field)
    return id
  end
end

function Bran:getGlobalId(global)
  local id = self.global_ids[global]
  if id then return id
  else
    id = 'global_'..tostring(self.n_global_ids) -- no global names
    self.n_global_ids = self.n_global_ids+1

    if use_legion then self:getFutureNum(global) end

    self.global_ids[global] = id
    self.arg_layout:addGlobal(id, global)
    return id
  end
end

function Bran:getReduceData(global)
  local data = self.global_reductions[global]
  if not data then
    local gid = self:getGlobalId(global)
    local id  = 'reduce_globalmem_'..gid:sub(#'global_' + 1)
         data = { id = id }

    self.global_reductions[global] = data
    if self:isOnGPU() then
      self.arg_layout:addReduce(id, global.type:terraType())
    end
  end
  return data
end

function Bran:setFieldPtr(field)
  if use_legion then
    error('INTERNAL: Do not call setFieldPtr() when using Legion') end
  local id = self:getFieldId(field)
  local dataptr = field:DataPtr()
  self.args:ptr()[id] = dataptr
end
function Bran:setGlobalPtr(global)
  if use_legion then
    error('INTERNAL: Do not call setGlobalPtr() when using Legion') end
  local id = self:getGlobalId(global)
  local dataptr = global:DataPtr()
  self.args:ptr()[id] = dataptr
end

function Bran:getTerraGlobalPtr(args_sym, global)
  local id = self:getGlobalId(global)
  return `[args_sym].[id]
end


--                  ---------------------------------------                  --
--[[ Bran Dynamic Checks                                                   ]]--
--                  ---------------------------------------                  --

function Bran:DynamicChecks()
  if use_single then
    -- Check that the fields are resident on the correct processor
    local underscore_field_fail = nil
    for field, _ in pairs(self.field_ids) do
      if field.array:location() ~= self.proc then
        if field:Name():sub(1,1) == '_' then
          underscore_field_fail = field
        else
          error("cannot execute kernel because field "..field:FullName()..
                " is not currently located on "..tostring(self.proc), 3)
        end
      end
    end
    if underscore_field_fail then
      error("cannot execute kernel because hidden field "..
            underscore_field_fail:FullName()..
            " is not currently located on "..tostring(self.proc), 3)
    end
  end

  if self:UsesInsert()  then  self:DynamicInsertChecks()  end
  if self:UsesDelete()  then  self:DynamicDeleteChecks()  end
end

--                  ---------------------------------------                  --
--[[ Bran Data Binding                                                     ]]--
--                  ---------------------------------------                  --

function Bran:BindData()
  -- Bind inserts and deletions before anything else, because
  -- the binding may trigger computations to re-size/re-allocate
  -- data in some cases, invalidating previous data pointers
  if self:UsesInsert()  then  self:bindInsertData()       end
  if self:UsesDelete()  then  self:bindDeleteData()       end

  -- Bind the rest of the data
  self:bindFieldGlobalSubsetArgs()

  -- Bind/Initialize any reduction data as needed
  if self:isOnGPU() then
    if self:UsesGlobalReduce()  then  self:bindGPUReductionData()   end
  end
end

function Bran:bindFieldGlobalSubsetArgs()
  -- Don't worry about binding on Legion, since we need
  -- to handle that a different way anyways
  if use_legion then return end

  local argptr    = self.args:ptr()
  argptr.n_rows   = self.relation:ConcreteSize()

  if self.subset and self.subset._index then
    argptr.index        = self.subset._index:DataPtr()
    -- Spoof the number of entries in the index, which is what
    -- we actually want to iterate over
    argptr.n_rows       = self.subset._index:Size()
  end

  for field, _ in pairs(self.field_ids) do
    self:setFieldPtr(field)
  end
  for globl, _ in pairs(self.global_ids) do
    self:setGlobalPtr(globl)
  end
end

--                  ---------------------------------------                  --
--[[ Bran Launch                                                           ]]--
--                  ---------------------------------------                  --

function Bran:Launch()
  if self:isOnGPU() then
    self.executable(self:numGPUBlocks(), self.args:ptr())
  else
    if use_legion then
      self.executable({ ctx = legion_env.ctx, runtime = legion_env.runtime })
    else
      self.executable(self.args:ptr())
    end
  end
end

--                  ---------------------------------------                  --
--[[ Bran Postprocess / Cleanup                                            ]]--
--                  ---------------------------------------                  --

function Bran:PostLaunchCleanup()
  -- GPU Reduction finishing and cleanup
  if self:isOnGPU() then
    if self:UsesGlobalReduce() then  self:postprocessGPUReduction()  end
  end

  -- Handle post execution Insertion and Deletion Behaviors
  if self:UsesInsert()         then   self:postprocessInsertions()    end
  if self:UsesDelete()         then   self:postprocessDeletions()     end
end



-------------------------------------------------------------------------------
-------------------------------------------------------------------------------
--[[ Insert / Delete Extensions                                            ]]--
-------------------------------------------------------------------------------
-------------------------------------------------------------------------------

--                  ---------------------------------------                  --
--[[ Insert Processing ; all 4 stages (-launch)                            ]]--
--                  ---------------------------------------                  --

function Bran:CompileInserts()
  local bran = self
  assert(bran.proc == L.CPU)

  local rel, ast_nodes = next(bran.kernel.inserts)
  bran.insert_data = {
    relation = rel,
    record_type = ast_nodes[1].record_type,
    n_inserted  = L.Global(L.uint64, 0),
  }
  -- register the global variable
  bran:getGlobalId(bran.insert_data.n_inserted)

  -- prep all the fields we want to be able to write to.
  for _,field in ipairs(rel._fields) do
    bran:getFieldId(field)
  end
  bran:getFieldId(rel._is_live_mask)
  bran.arg_layout:addInsertion()
end

function Bran:DynamicInsertChecks()
  if self.proc ~= L.CPU then
    error("insert statement is currently only supported in CPU-mode.", 4)
  end
  local rel = self.insert_data.relation
  local unsafe_msg = rel:UnsafeToInsert(self.insert_data.record_type)
  if unsafe_msg then error(unsafe_msg, 4) end
end

function Bran:bindInsertData()
  local insert_rel                    = self.insert_data.relation
  local center_size_logical           = self.relation:Size()
  local insert_size_concrete          = insert_rel:ConcreteSize()

  self.insert_data.n_inserted:set(0)
  -- cache the old size
  self.insert_data.last_concrete_size = insert_size_concrete
  -- set the write head to point to the end of array
  self.args:ptr().insert_write        = insert_size_concrete
  -- resize to create more space at the end of the array
  insert_rel:ResizeConcrete(insert_size_concrete +
                            center_size_logical)
end

function Bran:postprocessInsertions()
  local insert_rel        = self.insert_data.relation
  local old_concrete_size = self.insert_data.last_concrete_size
  local old_logical_size  = insert_rel._logical_size
  -- WARNING UNSAFE CONVERSION FROM UINT64 to DOUBLE
  local n_inserted        = tonumber(self.insert_data.n_inserted:get())

  -- shrink array back down to where we actually ended up writing
  local new_concrete_size = old_concrete_size + n_inserted
  insert_rel:ResizeConcrete(new_concrete_size)
  -- update the logical view of the size
  insert_rel._logical_size = old_logical_size + n_inserted

  -- NOTE that this relation is definitely fragmented now
  self.insert_data.relation:_INTERNAL_MarkFragmented()
end

--                  ---------------------------------------                  --
--[[ Delete Processing ; all 4 stages (-launch)                            ]]--
--                  ---------------------------------------                  --

function Bran:CompileDeletes()
  local bran = self
  assert(bran.proc == L.CPU)

  local rel = next(bran.kernel.deletes)
  bran.delete_data = {
    relation = rel,
    updated_size = L.Global(L.uint64, 0)
  }
  -- register global variable
  bran:getGlobalId(bran.delete_data.updated_size)
end

function Bran:DynamicDeleteChecks()
  if self.proc ~= L.CPU then
    error("delete statement is currently only supported in CPU-mode.", 4)
  end
  local unsafe_msg = self.delete_data.relation:UnsafeToDelete()
  if unsafe_msg then error(unsafe_msg, 4) end
end

function Bran:bindDeleteData()
  local relsize = tonumber(self.delete_data.relation._logical_size)
  self.delete_data.updated_size:set(relsize)
end

function Bran:postprocessDeletions()
  -- WARNING UNSAFE CONVERSION FROM UINT64 TO DOUBLE
  local rel = self.delete_data.relation
  local updated_size = tonumber(self.delete_data.updated_size:get())
  rel._logical_size = updated_size
  rel:_INTERNAL_MarkFragmented()

  -- if we have too low an occupancy
  if rel:Size() < 0.5 * rel:ConcreteSize() then
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

function Bran:numGPUBlocks()
  if self:isOverSubset() then
    if self.subset._boolmask then
      return math.ceil(self.relation:ConcreteSize() / self.blocksize)
    elseif self.subset._index then
      return math.ceil(self.subset._index:Size() / self.blocksize)
    end
  else
    return math.ceil(self.relation:ConcreteSize() / self.blocksize)
  end
end

function Bran:nBytesSharedMem()
  return self.sharedmem_size
end

function Bran:getBlockSize()
  return self.blocksize
end

function Bran:setReduceGlobalMemPtr(global, dataptr)
  local data = self:getReduceData(global)
  self.args:ptr()[data.id] = dataptr
end

function Bran:getTerraReduceGlobalMemPtr(args_sym, global)
  local data = self:getReduceData(global)
  return `[args_sym].[data.id]
end

function Bran:freeReduceGlobalMemPtr(global)
  local data = self:getReduceData(global)
  local aptr = self.args:ptr()
  G.free( aptr[data.id] )
  aptr[data.id] = nil
end

function Bran:getTerraReduceSharedMemPtr(global)
  local data = self:getReduceData(global)
  return data.sharedmem
end

--                  ---------------------------------------                  --
--[[ GPU Reduction Compilation                                             ]]--
--                  ---------------------------------------                  --

function Bran:CompileGPUReduction()
  self.sharedmem_size         = self.sharedmem_size or 0

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
  --      for allocating and deallocating shared memory on kernel launch/exit

  -- Find all the global variables in this kernel that are being reduced
  for globl, data in pairs(self.global_reductions) do
    local ttype             = globl.type:terraType()
    data.sharedmem          = cudalib.sharedmemory(ttype, self.blocksize)

    self.sharedmem_size     = self.sharedmem_size +
                                sizeof(ttype) * self.blocksize
  end
--  for globl, phase in pairs(self.kernel.global_use) do
--    if phase.reduceop then
--      self.uses_gpu_reduce  = true
--      local ttype           = globl.type:terraType()
--
--      local reduce_data     = self:getReduceData(globl)
--      reduce_data.phase     = phase
--      reduce_data.sharedmem = cudalib.sharedmemory(ttype, self.blocksize)
--
--      self.sharedmem_size   = self.sharedmem_size +
--                                sizeof(ttype) * self.blocksize
--    end
--  end

  self:CompileGlobalMemReductionKernel()
end

-- The following routine is also used inside the primary compile CUDA kernel
function Bran:GenerateSharedMemInitialization(tid_sym)
  local code = quote end
  for globl, data in pairs(self.global_reductions) do
    local op        = data.phase.reduceop
    local lz_type   = globl.type
    local sharedmem = data.sharedmem

    code = quote
      [code]
      [sharedmem][tid_sym] = [codesupport.reduction_identity(lz_type, op)]
    end
  end
  return code
end

-- The following routine is also used inside the primary compile CUDA kernel
function Bran:GenerateSharedMemReduceTree(args_sym, tid_sym, bid_sym, is_final)
  is_final = is_final or false
  local code = quote end
  for globl, data in pairs(self.global_reductions) do
    local op          = data.phase.reduceop
    local lz_type     = globl.type
    local sharedmem   = data.sharedmem
    local finalptr    = self:getTerraGlobalPtr(args_sym, globl)
    local globalmem   = self:getTerraReduceGlobalMemPtr(args_sym, globl)

    -- Insert an unrolled reduction tree here
    local step = self.blocksize
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
  end
  return code
end

-- The full secondary CUDA kernel to reduce the contents of the
-- global mem array.  See comment inside function for sketch of algorithm
function Bran:CompileGlobalMemReductionKernel()
  local bran      = self
  local fn_name   = bran.kernel.typed_ast.id .. '_globalmem_reduction'

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
  local args      = symbol(bran:argsType())
  local array_len = symbol(uint64)
  local tid       = symbol(uint32)
  local bid       = symbol(uint32)

  local cuda_kernel =
  terra([array_len], [args])
    var [tid]    : uint32 = G.thread_id()
    var [bid]    : uint32 = G.block_id()
    var n_blocks : uint32 = G.num_blocks()
    var gt                = tid + [bran.blocksize] * bid
    
    -- INITIALIZE the shared memory
    [bran:GenerateSharedMemInitialization(tid)]
    
    -- REDUCE the global memory into the provided shared memory
    -- count from (gt) till (array_len) by step sizes of (blocksize)
    for gi = gt, array_len, n_blocks * [bran.blocksize] do
      escape for globl, data in pairs(bran.global_reductions) do
        local op          = data.phase.reduceop
        local lz_type     = globl.type
        local sharedmem   = data.sharedmem
        local globalmem   = bran:getTerraReduceGlobalMemPtr(args, globl)

        emit quote
          [sharedmem][tid]  = [codesupport.reduction_binop(lz_type, op,
                                                           `[sharedmem][tid],
                                                           `[globalmem][gi])]
        end
      end end
    end

    G.barrier()
  
    -- REDUCE the shared memory using a tree
    [bran:GenerateSharedMemReduceTree(args, tid, bid, true)]
  end
  cuda_kernel:setname(fn_name)
  cuda_kernel = G.kernelwrap(cuda_kernel, L._INTERNAL_DEV_OUTPUT_PTX)

  -- the globalmem array has an entry for every block in the primary kernel
  local globalmem_array_len = bran:numGPUBlocks() 
  local terra launcher( argptr : &(bran:argsType()) )
    var launch_params = terralib.CUDAParams {
      1,1,1, [bran.blocksize],1,1, [bran.sharedmem_size], nil
    }
    cuda_kernel(&launch_params, globalmem_array_len, @argptr )
  end
  launcher:setname(fn_name..'_launcher')

  bran.global_reduction_pass = launcher
end

--                  ---------------------------------------                  --
--[[ GPU Reduction Dynamic Checks                                          ]]--
--                  ---------------------------------------                  --

function Bran:DynamicGPUReductionChecks()
  if self.proc ~= L.GPU then
    error("INTERNAL ERROR: Should only try to run GPUReduction on the GPU...")
  end
end

--                  ---------------------------------------                  --
--[[ GPU Reduction Data Binding                                            ]]--
--                  ---------------------------------------                  --

function Bran:bindGPUReductionData()
  local n_blocks = self:numGPUBlocks()

  -- allocate GPU global memory for the reduction
  for globl, _ in pairs(self.global_reductions) do
    local ttype = globl.type:terraType()
    self:setReduceGlobalMemPtr(globl, G.malloc(ttype, n_blocks))
  end
end

--                  ---------------------------------------                  --
--[[ GPU Reduction Postprocessing                                          ]]--
--                  ---------------------------------------                  --

function Bran:postprocessGPUReduction()
  -- perform inter-block reduction step (secondary kernel launch)
  self.global_reduction_pass(self.args:ptr())

  -- free GPU global memory allocated for the reduction
  for globl, _ in pairs(self.global_reductions) do
    self:freeReduceGlobalMemPtr(globl)
  end
end



-------------------------------------------------------------------------------
-------------------------------------------------------------------------------
--[[ Legion Extensions                                                     ]]--
-------------------------------------------------------------------------------
-------------------------------------------------------------------------------




local function pairs_val_sorted(tbl)
  local list = {}
  for k,v in pairs(tbl) do table.insert(list, {k,v}) end
  table.sort(list, function(p1,p2)
    return p1[2] < p2[2]
  end)

  local i = 0
  return function() -- iterator
    i = i+1
    if list[i] == nil then return nil
                      else return list[i][1], list[i][2] end
  end
end



-- Creates a task launcher with task region requirements.
function Bran:CreateLegionTaskLauncher(task_func)

  -- TODO: Cannot pass a terra function to another terra function.
  -- Doing so throws error "cannot convert 'table' to 'bool (*)()'".
  -- Should fix this, and remove the wrapper terra function defined below.
  local task_func_wrapper
  local task_TID
  --local task_func_ptr
  if self:UsesGlobalReduce() then
    task_func_wrapper = terra()
      return LW.NewFutureKernelLauncher(task_func)
    end
    task_TID = LW.TID_FUTURE
  else
    task_func_wrapper = terra()
      return LW.NewSimpleKernelLauncher(task_func)
    end
    task_TID = LW.TID_SIMPLE
  end
  local task_as_arg = task_func_wrapper():PackToTaskArg()


  local task_launcher = LW.legion_task_launcher_create(
                            task_TID,
                            task_as_arg,
                            LW.legion_predicate_true(), 0, 0
                        )

  -- ADD EACH REGION to the launcher as a requirement
  -- WITH THE appropriate permissions set
  -- NOTE: Need to make sure to do this in the right order
  for ri, datum in pairs(self.sorted_region_data) do
    local reg_req = 
      LW.legion_task_launcher_add_region_requirement_logical_region(
        task_launcher,
        datum.wrapper.handle,
        LW.READ_WRITE,
        LW.EXCLUSIVE,
        datum.wrapper.handle, -- why is this repeated?
        0,
        false
      )
    assert(reg_req == ri)
  end

  -- ADD EACH FIELD to the launcher as a requirement
  -- as part of the correct, corresponding region
  for field, _ in pairs(self.field_ids) do
    LW.legion_task_launcher_add_field(
      task_launcher,
      self:getRegionData(field).num,
      field.fid,
      true
    )
  end

  -- ADD EACH GLOBAL to the launcher as a future being passed to the task
  -- NOTE: Need to make sure to do this in the right order
  for globl, gi in pairs_val_sorted(self.future_nums) do
    LW.legion_task_launcher_add_future(task_launcher, globl.data)
  end

  return task_launcher
end

-- Launches Legion task and returns.
function Bran:CreateLegionLauncher(task_func)
  local bran = self
  local task_launcher = bran:CreateLegionTaskLauncher(task_func)
  if bran:UsesGlobalReduce() then
    return function(leg_args)
      local globl   = next(bran.global_reductions)
      local future  = LW.legion_task_launcher_execute(leg_args.runtime,
                                                      leg_args.ctx,
                                                      task_launcher)
      local res = LW.legion_future_get_result(future)
      -- Wait till value is available. We can remove this once apply and
      -- fold operations are implemented using legion API, and we figure out
      -- how to safely delete old future : Is it safe to call DestroyFuture
      -- immediately after launching the tasks that use the future?
      -- TODO: We must apply this return value to old value - necessary for
      -- multiple partitions. Work around right now applies reduction in the
      -- task (Liszt kernel) itself, so we can simply replace the old future.
      globl.data = LW.legion_future_from_buffer(leg_args.runtime,
                                                res.value, res.value_size)
      LW.legion_task_result_destroy(res)
    end
  else
    return function(leg_args)
      LW.legion_task_launcher_execute(leg_args.runtime, leg_args.ctx,
                                      task_launcher)
    end
  end
end

-- Here we translate the Legion task arguments into our
-- custom argument layout structure.  This allows us to write
-- the body of generated code in a way that's agnostic to whether
-- the code is being executed in a Legion task or not.
function Bran:GenerateUnpackLegionTaskArgs(argsym, task_args)
  local bran = self
  
  local LegionRect = {}
  local LegionGetRectFromDom = {}
  local LegionRawPtrFromAcc = {}

  LegionRect[1] = LW.legion_rect_1d_t
  LegionRect[2] = LW.legion_rect_2d_t
  LegionRect[3] = LW.legion_rect_3d_t

  LegionGetRectFromDom[1] = LW.legion_domain_get_rect_1d
  LegionGetRectFromDom[2] = LW.legion_domain_get_rect_2d
  LegionGetRectFromDom[3] = LW.legion_domain_get_rect_3d

  LegionRawPtrFromAcc[1] = LW.legion_accessor_generic_raw_rect_ptr_1d
  LegionRawPtrFromAcc[2] = LW.legion_accessor_generic_raw_rect_ptr_2d
  LegionRawPtrFromAcc[3] = LW.legion_accessor_generic_raw_rect_ptr_3d

  -- temporary collection of symbols from unpacking the regions
  local region_temporaries = {}

  local code = quote
    do -- close after unpacking the fields
    -- UNPACK REGIONS
    escape for ri, datum in pairs(bran.sorted_region_data) do
      local reg_dim       = datum.wrapper.dimensions
      -- KLUDGE cause of WRAPPER
      if not reg_dim then reg_dim = 1 end
      local physical_reg  = symbol(LW.legion_physical_region_t)
      local rect          = symbol(LegionRect[reg_dim])
      local rectFromDom   = LegionGetRectFromDom[reg_dim]

      region_temporaries[ri] = {
        physical_reg  = physical_reg,
        reg_dim       = reg_dim,
        rect          = rect
      }

      emit quote
        var [physical_reg]  = [task_args].regions[ri]
        var index_space     =
          LW.legion_physical_region_get_logical_region(
                                           physical_reg).index_space
        var domain          =
          LW.legion_index_space_get_domain([task_args].lg_runtime,
                                           [task_args].lg_ctx,
                                           index_space)
        var [rect]          = rectFromDom(domain)
      end
    end end

    -- UNPACK PRIMARY REGION BOUNDS RECTANGLE
    escape
      local ri    = bran:getPrimaryRegionData().num
      local rect  = region_temporaries[ri].rect
      local ndims = region_temporaries[ri].reg_dim
      for i=1,ndims do emit quote
        [argsym].bounds[i-1].lo = rect.lo.x[i-1]
        [argsym].bounds[i-1].hi = rect.hi.x[i-1]
      end end
    end
    
    -- UNPACK FIELDS
    escape for field, farg_name in pairs(bran.field_ids) do
      local rtemp         = region_temporaries[bran:getRegionData(field).num]
      local physical_reg  = rtemp.physical_reg
      local reg_dim       = rtemp.reg_dim
      local rect          = rtemp.rect


      emit quote
        var field_accessor =
          LW.legion_physical_region_get_field_accessor_generic(
                                              physical_reg, [field.fid])
        var subrect : LegionRect[reg_dim]
        var strides : LW.legion_byte_offset_t[reg_dim]
        var base = [&uint8](
          [ LegionRawPtrFromAcc[reg_dim] ](
                              field_accessor, rect, &subrect, strides))
        [argsym].[farg_name] = [ LW.FieldAccessor[reg_dim] ] { base, strides }
      end
    end end
    end -- closing do started before unpacking the regions

    -- UNPACK FUTURES
    -- DO NOT WRAP THIS IN A LOCAL SCOPE or IN A DO BLOCK (SEE BELOW)
    escape for globl, garg_name in pairs(bran.global_ids) do
      -- position in the Legion task arguments
      local fut_i   = bran:getFutureNum(globl) 
      local gtyp    = globl.type:terraType()

      emit quote
        var fut     = LW.legion_task_get_future([task_args].task, fut_i)
        var result  = LW.legion_future_get_result(fut)
        var datum   = @[&gtyp](result.value)
        -- note that we're going to rely on this variable
        -- being stably allocated on the stack
        -- for the remainder of this function scope
        [argsym].[garg_name] = &datum
        LW.legion_task_result_destroy(result)
      end
    end end
  end -- end quote

  return code
end



function Bran:CompileLegion()
  local task_function     = codegen.codegen(self.kernel.typed_ast, self)
  -- we attach the task function to the Bran in order to
  -- prevent it from being garbage collected prematurely
  self.remember_task_func = task_function
  self.executable         = self:CreateLegionLauncher(task_function)
end




--                  ---------------------------------------                  --
--[[ Legion Dynamic Checks                                                 ]]--
--                  ---------------------------------------                  --

function Bran:DynamicLegionChecks()
end

--                  ---------------------------------------                  --
--[[ Legion Data Binding                                                   ]]--
--                  ---------------------------------------                  --

function Bran:bindLegionData()
  -- meh
end

--                  ---------------------------------------                  --
--[[ Legion Postprocessing                                                 ]]--
--                  ---------------------------------------                  --

function Bran:postprocessLegion()
  -- meh for now
end










-------------------------------------------------------------------------------
-------------------------------------------------------------------------------
--[[ ArgLayout                                                             ]]--
-------------------------------------------------------------------------------
-------------------------------------------------------------------------------


function ArgLayout.New()
  return setmetatable({
    fields            = terralib.newlist(),
    globals           = terralib.newlist(),
    reduce            = terralib.newlist()
  }, ArgLayout)
end

function ArgLayout:setNDims(n)
  self.n_dims = n
end

function ArgLayout:addField(name, field)
  if self:isCompiled() then
    error('INTERNAL ERROR: cannot add new fields to compiled layout')
  end
  if use_single then
    local typ = field:Type():terraType()
    table.insert(self.fields, { field=name, type=&typ })
  elseif use_legion then
    local ndims = field:Relation():nDims()
    table.insert(self.fields, { field=name, type=LW.FieldAccessor[ndims] })
  end
end

function ArgLayout:addGlobal(name, global)
  if self:isCompiled() then
    error('INTERNAL ERROR: cannot add new globals to compiled layout')
  end
  local typ = global.type:terraType()
  table.insert(self.globals, { field=name, type=&typ })
end

function ArgLayout:addReduce(name, typ)
  if self:isCompiled() then
    error('INTERNAL ERROR: cannot add new reductions to compiled layout')
  end
  table.insert(self.reduce, { field=name, type=&typ})
end

function ArgLayout:turnSubsetOn()
  if self:isCompiled() then
    error('INTERNAL ERROR: cannot add a subset to compiled layout')
  end
  self.subset_on = true
end

function ArgLayout:addInsertion()
  if self:isCompiled() then
    error('INTERNAL ERROR: cannot add insertions to compiled layout')
  end
  self.insert_on = true
end

function ArgLayout:TerraStruct()
  if not self:isCompiled() then self:Compile() end
  return self.terrastruct
end

local struct bounds_struct { lo : uint64, hi : uint64 }

function ArgLayout:Compile()
  local terrastruct = terralib.types.newstruct(self.name)

  -- add counter
  if use_legion then
    table.insert(terrastruct.entries,
                 {field='bounds', type=(bounds_struct[self.n_dims])})
  else
    table.insert(terrastruct.entries, {field='n_rows', type=uint64})
  end
  -- add subset data
  local taddr = L.addr_terra_types[self.n_dims]
  if self.subset_on then
    table.insert(terrastruct.entries, {field='index',        type=&taddr})
    table.insert(terrastruct.entries, {field='index_size',   type=uint64})
  end
  if self.insert_on then
    table.insert(terrastruct.entries, {field='insert_write', type=uint64})
  end
  -- add fields
  for _,v in ipairs(self.fields) do table.insert(terrastruct.entries, v) end
  -- add globals
  for _,v in ipairs(self.globals) do table.insert(terrastruct.entries, v) end
  -- add global reduction space
  for _,v in ipairs(self.reduce) do table.insert(terrastruct.entries, v) end

  self.terrastruct = terrastruct
end

function ArgLayout:isCompiled()
  return self.terrastruct ~= nil
end






