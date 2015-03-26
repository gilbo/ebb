local K = {}
package.loaded["compiler.kernel_single"] = K
local Kc = require "compiler.kernel_common"
local L = require "compiler.lisztlib"

local codegen         = require "compiler.codegen_single"
local DataArray       = require('compiler.rawdata').DataArray


local Bran = Kc.Bran
local seedbank_lookup = Kc.seedbank_lookup



-- Create a Lua Object that generates the needed Terra structure to pass
-- fields, globals and temporary allocated memory to the kernel as arguments
local ArgLayout = {}
ArgLayout.__index = ArgLayout


-------------------------------------------------------------------------------
--[[ Kernels                                                               ]]--
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
--[[ Brans                                                                 ]]--
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
  local bran      = self
  local kernel    = bran.kernel
  local typed_ast = bran.kernel.typed_ast


  -- type checking the kernel signature against the invocation
  if typed_ast.relation ~= bran.relation then
      error('Kernels may only be called on a relation they were typed with', 3)
  end

  bran.arg_layout = ArgLayout.New()

  -- compile various kinds of data into the arg layout
  bran:CompileFieldsGlobalsSubsets()
  -- also compile insertion and/or deletion if used
  if kernel.inserts then    bran:CompileInserts()   end
  if kernel.deletes then    bran:CompileDeletes()   end

  -- allocate memory for the arguments struct on the CPU.  It will be used
  -- to hold the parameter values that will be passed to the Liszt kernel.
  bran.args = DataArray.New{
    size = 1,
    type = bran.arg_layout:TerraStruct(),
    processor = L.CPU -- DON'T MOVE
  }

  -- compile an executable
  bran.executable = codegen.codegen(typed_ast, bran)
end

function Bran:CompileFieldsGlobalsSubsets()
  -- initialize id structures
  self.field_ids    = {}
  self.n_field_ids  = 0

  self.global_ids   = {}
  self.n_global_ids = 0

  -- reserve ids
  for field, _ in pairs(self.kernel.field_use) do
    self:getFieldId(field)
  end
  self:getFieldId(self.relation._is_live_mask)
  for globl, _ in pairs(self.kernel.global_use) do
    self:getGlobalId(globl)
  end

  -- compile subsets in if appropriate
  if self.subset then
    self.arg_layout:turnSubsetOn()
  end
end

--                  ---------------------------------------                  --

function Bran:getFieldId(field)
  local id = self.field_ids[field]
  if not id then
    id = 'field_'..tostring(self.n_field_ids)..'_'..field:Name()
    self.n_field_ids = self.n_field_ids+1

    self.field_ids[field] = id
    self.arg_layout:addField(id, field:Type():terraType())
  end
  return id
end

function Bran:getGlobalId(global)
  local id = self.global_ids[global]
  if not id then
    id = 'global_'..tostring(self.n_global_ids) -- no global names
    self.n_global_ids = self.n_global_ids+1

    self.global_ids[global] = id
    self.arg_layout:addGlobal(id, global.type:terraType())
  end
  return id
end

function Bran:setFieldPtr(field)
  local id = self:getFieldId(field)
  local dataptr = field:DataPtr()
  self.args:ptr()[id] = dataptr
end
function Bran:setGlobalPtr(global)
  local id = self:getGlobalId(global)
  local dataptr = global:DataPtr()
  self.args:ptr()[id] = dataptr
end
function Bran:argsType ()
  return self.arg_layout:TerraStruct()
end
function Bran:getFieldPtr(signature_ptr, field)
  local id = self:getFieldId(field)
  return `[signature_ptr].[id]
end
function Bran:getGlobalPtr(signature_ptr, global)
  local id = self:getGlobalId(global)
  return `[signature_ptr].[id]
end
function Bran:getGlobalScratchPtr(signature_ptr, global)
    local id = self:getGlobalId(global)
    id = "reduce_" .. id
    return `[signature_ptr].[id]
end

--                  ---------------------------------------                  --

function Bran:DynamicChecks()
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

  if self:UsesInsert() then   self:DynamicInsertChecks()   end
  if self:UsesDelete() then   self:DynamicDeleteChecks()   end
end

--                  ---------------------------------------                  --

function Bran:BindData()
  -- Bind inserts and deletions before anything else, because
  -- the binding may trigger computations to re-size/re-allocate
  -- data in some cases, invalidating previous data pointers
  if self:UsesInsert() then   self:bindInsertData()   end
  if self:UsesDelete() then   self:bindDeleteData()   end

  -- Bind the rest of the data
  self:bindFieldGlobalSubsetArgs()
end

function Bran:bindFieldGlobalSubsetArgs()
  local argptr    = self.args:ptr()
  argptr.n_rows   = self.relation:ConcreteSize()

  if self.subset then
    argptr.use_boolmask   = false
    if self.subset._boolmask then
      argptr.use_boolmask = true
      argptr.boolmask     = self.subset._boolmask:DataPtr()
    elseif self.subset._index then
      argptr.index        = self.subset._index:DataPtr()
      argptr.index_size   = self.subset._index:Size()
    else
      error('INTERNAL ERROR: trying to bind subset, '..
            'must have boolmask or index')
    end
  end

  for field, _ in pairs(self.field_ids) do
    self:setFieldPtr(field)
  end
  for globl, _ in pairs(self.global_ids) do
    self:setGlobalPtr(globl)
  end
end

--                  ---------------------------------------                  --

function Bran:Launch()
  self.executable(self.args:ptr())
end

--                  ---------------------------------------                  --

function Bran:PostLaunchCleanup()
  -- Handle post execution Insertion and Deletion Behaviors
  if self:UsesInsert() then   self:postprocessInsertions()  end
  if self:UsesDelete() then   self:postprocessDeletions()   end
end



-------------------------------------------------------------------------------
--[[ Insert / Delete Extensions                                            ]]--
-------------------------------------------------------------------------------

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
--[[ ArgLayout                                                            ]]--
-------------------------------------------------------------------------------


function ArgLayout.New()
  return setmetatable({
    fields    = terralib.newlist(),
    globals   = terralib.newlist(),
    reduce    = terralib.newlist()
  }, ArgLayout)
end

function ArgLayout:addField(name, typ)
  if self:isCompiled() then
    error('INTERNAL ERROR: cannot add new fields to compiled layout')
  end
  table.insert(self.fields, { field=name, type=&typ })
end

function ArgLayout:addGlobal(name, typ)
  if self:isCompiled() then
    error('INTERNAL ERROR: cannot add new globals to compiled layout')
  end
  table.insert(self.globals, { field=name, type=&typ })
  table.insert(self.globals, { field='reduce_'..name,type=&typ})
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

function ArgLayout:Compile()
  local terrastruct = terralib.types.newstruct(self.name)

  -- add counter
  table.insert(terrastruct.entries, {field='n_rows', type=uint64})
  -- add subset data
  local taddr = L.addr_terra_types[1]
  if self.subset_on then
    table.insert(terrastruct.entries, {field='use_boolmask', type=bool})
    table.insert(terrastruct.entries, {field='boolmask',     type=&bool})
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






