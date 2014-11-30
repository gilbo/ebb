local K = {}
package.loaded["compiler.kernel_single"] = K
local Kc = terralib.require "compiler.kernel_common"
local L = terralib.require "compiler.lisztlib"

local codegen        = terralib.require "compiler.codegen_single"
local DataArray = terralib.require('compiler.rawdata').DataArray


-------------------------------------------------------------------------------
--[[ Kernels, Brans, Signatures                                            ]]--
-------------------------------------------------------------------------------

local Bran = Kc.Bran
local Seedbank = Kc.Seedbank
local seedbank_lookup = Kc.seedbank_lookup


-------------------------------------------------------------------------------
--[[ Signatures                                                            ]]--
-------------------------------------------------------------------------------

-- Create a Lua Object that generates the needed Terra structure to pass
-- fields, globals and temporary allocated memory to the kernel as arguments
local SignatureTemplate = {}
SignatureTemplate.__index = SignatureTemplate

function SignatureTemplate.New()
  return setmetatable({
    fields    = terralib.newlist(),
    globals   = terralib.newlist(),
    reduce    = terralib.newlist()
  }, SignatureTemplate)
end

function SignatureTemplate:addField(name, typ)
  table.insert(self.fields, { field=name, type=&typ })
end

function SignatureTemplate:addGlobal(name, typ)
  table.insert(self.globals, { field=name, type=&typ })
  table.insert(self.globals, { field='reduce_'..name,type=&typ})
end

function SignatureTemplate:turnSubsetOn()
  self.subset_on = true
end

function SignatureTemplate:addInsertion()
  self.insert_on = true
end

local taddr = uint64 --L.addr:terraType() -- weird dependency error
function SignatureTemplate:TerraStruct()
  if self.terrastruct then return self.terrastruct end
  local terrastruct = terralib.types.newstruct(self.name)

  -- add counter
  table.insert(terrastruct.entries, {field='n_rows', type=uint64})
  -- add subset data
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
  return terrastruct
end

function SignatureTemplate:isGenerated()
  return self.terrastruct ~= nil
end

K.SignatureTemplate = SignatureTemplate


-------------------------------------------------------------------------------
--[[ Kernels                                                               ]]--
-------------------------------------------------------------------------------

L.LKernel.__call  = function (kobj, relset)
    if not (relset and (L.is_relation(relset) or L.is_subset(relset)))
    then
        error("A kernel must be called on a relation or subset.", 2)
    end

    local proc = L.default_processor

    -- retreive the correct bran or create a new one
    local bran = seedbank_lookup({
        kernel=kobj,
        relset=relset,
        proc=proc,
    })
    if not bran.executable then
      bran.relset = relset
      bran.kernel = kobj
      bran.location = proc
      bran:generate()
    end

    -- determine whether or not this kernel invocation is
    -- safe to run or not.
    bran:dynamicChecks()


    -- set execution parameters in the signature
    local signature    = bran.signature:ptr()
    signature.n_rows   = bran.relation:ConcreteSize()
    -- bind the subset data
    if bran.subset then
      signature.use_boolmask     = false
      if bran.subset then
        if bran.subset._boolmask then
          signature.use_boolmask = true
          signature.boolmask     = bran.subset._boolmask:DataPtr()
        elseif bran.subset._index then
          signature.index        = bran.subset._index:DataPtr()
          signature.index_size   = bran.subset._index:Size()
        else
          assert(false)
        end
      end
    end
    -- bind insert data
    if bran.insert_data then
      local insert_rel            = bran.insert_data.relation
      local center_size_logical   = bran.relation:Size()
      local insert_size_concrete  = insert_rel:ConcreteSize()

      bran.insert_data.n_inserted:set(0)
      -- cache the old size
      bran.insert_data.last_concrete_size = insert_size_concrete
      -- set the write head to point to the end of array
      signature.insert_write = insert_size_concrete
      -- resize to create more space at the end of the array
      insert_rel:ResizeConcrete(insert_size_concrete +
                                center_size_logical)
    end
    -- bind delete data (just a global here)
    if bran.delete_data then
      -- FORCE CONVERSION OUT OF UINT64; NOTE DANGEROUS
      local relsize = tonumber(bran.delete_data.relation._logical_size)
      bran.delete_data.updated_size:set(relsize)
    end
    -- bind the field data (MUST COME LAST)
    for field, _ in pairs(bran.field_ids) do
      bran:setField(field)
    end
    for globl, _ in pairs(bran.global_ids) do
      bran:setGlobal(globl)
    end

    -- launch the kernel
    bran.executable(bran.signature:ptr())

    -- adjust sizes based on extracted information
    if bran.insert_data then
      local insert_rel        = bran.insert_data.relation
      local old_concrete_size = bran.insert_data.last_concrete_size
      local old_logical_size  = insert_rel._logical_size
      -- WARNING UNSAFE CONVERSION FROM UINT64 to DOUBLE
      local n_inserted        = tonumber(bran.insert_data.n_inserted:get())

      -- shrink array back down to where we actually ended up writing
      local new_concrete_size = old_concrete_size + n_inserted
      insert_rel:ResizeConcrete(new_concrete_size)
      -- update the logical view of the size
      insert_rel._logical_size = old_logical_size + n_inserted

      -- NOTE that this relation is definitely fragmented now
      bran.insert_data.relation._typestate.fragmented = true
    end
    if bran.delete_data then
      -- WARNING UNSAFE CONVERSION FROM UINT64 TO DOUBLE
      local rel = bran.delete_data.relation
      local updated_size = tonumber(bran.delete_data.updated_size:get())
      rel._logical_size = updated_size
      rel._typestate.fragmented = true

      -- if we have too low an occupancy
      if rel:Size() < 0.5 * rel:ConcreteSize() then
        rel:Defrag()
      end
    end
end


-------------------------------------------------------------------------------
--[[ Brans                                                                 ]]--
-------------------------------------------------------------------------------

function Bran:generate()
  local bran      = self
  local kernel    = bran.kernel
  local typed_ast = bran.kernel.typed_ast

  -- break out the arguments
  if L.is_relation(bran.relset) then
    bran.relation = bran.relset
  else
    bran.relation = bran.relset:Relation()
    bran.subset   = bran.relset
  end

  -- type checking the kernel signature against the invocation
  if typed_ast.relation ~= bran.relation then
      error('Kernels may only be called on a relation they were typed with', 3)
  end

  bran.signature_template = SignatureTemplate.New()

  -- fix the mapping for the fields before compiling the executable
  bran.field_ids    = {}
  bran.n_field_ids  = 0
  for field, _ in pairs(kernel.field_use) do
    bran:getFieldId(field)
  end
  bran:getFieldId(bran.relation._is_live_mask)
  -- fix the mapping for the globals before compiling the executable
  bran.global_ids   = {}
  bran.n_global_ids = 0
  for globl, _ in pairs(kernel.global_use) do
    bran:getGlobalId(globl)
  end
  -- setup subsets?
  if bran.subset then bran.signature_template:turnSubsetOn() end

  -- setup insert and delete
  if kernel.inserts then
    bran:generateInserts()
  end
  if kernel.deletes then
    bran:generateDeletes()
  end

  -- allocate memory for the signature on the CPU.  It will be used
  -- to hold the parameter values that will be passed to the liszt kernel.
  bran.signature = DataArray.New{
    size = 1,
    type = bran.signature_template:TerraStruct(),
    processor = L.CPU -- DON'T MOVE
  }

  -- compile an executable
  bran.executable = codegen.codegen(typed_ast, bran)
end

function Bran:signatureType()
  return self.signature_template:TerraStruct()
end

function Bran:getFieldId(field)
  local id = self.field_ids[field]
  if not id then
    if self.signature_template:isGenerated() then
      error('INTERNAL ERROR: cannot add new fields after struct gen')
    end
    id = 'field_'..tostring(self.n_field_ids)..'_'..field:Name()
    self.n_field_ids = self.n_field_ids+1

    self.field_ids[field] = id
    self.signature_template:addField(id, field:Type():terraType())
  end
  return id
end

function Bran:getGlobalId(global)
  local id = self.global_ids[global]
  if not id then
    if self.signature_template:isGenerated() then
      error('INTERNAL ERROR: cannot add new globals after struct gen')
    end
    id = 'global_'..tostring(self.n_global_ids) -- no global names
    self.n_global_ids = self.n_global_ids+1

    self.global_ids[global] = id
    self.signature_template:addGlobal(id, global.type:terraType())
  end
  return id
end

function Bran:setField(field)
  local id = self:getFieldId(field)
  local dataptr = field:DataPtr()
  self.signature:ptr()[id] = dataptr
end
function Bran:setGlobal(global)
  local id = self:getGlobalId(global)
  local dataptr = global:DataPtr()
  self.signature:ptr()[id] = dataptr
end
function Bran:signatureType ()
  return self.signature_template:TerraStruct()
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

function Bran:dynamicChecks()
  -- Check that the fields are resident on the correct processor
  -- TODO(crystal)  - error message here can be confusing.  For example, the
  -- dynamic check may report an error on the location of a field generated by a
  -- liszt library.  Since the user is not aware of how/when the field was
  -- generated, this makes it hard to determine how to fix the error.  Perhaps we
  -- should report *all* incorrectly located fields? Or at least prefer printing
  -- fields that are not prefaced with an underscore?
  for field, _ in pairs(self.field_ids) do
    if field.array:location() ~= self.location then
      error("cannot execute kernel because field "..field:FullName()..
            " is not currently located on "..tostring(self.location), 3)
    end
  end

  if self.insert_data then 
    if self.location ~= L.CPU then
      error("insert statement is currently only supported in CPU-mode.", 3)
    end
    local rel = self.insert_data.relation
    local unsafe_msg = rel:UnsafeToInsert(self.insert_data.record_type)
    if unsafe_msg then error(unsafe_msg, 3) end
  end
  if self.delete_data then
    if self.location ~= L.CPU then
      error("delete statement is currently only supported in CPU-mode.", 3)
    end
    local unsafe_msg = self.delete_data.relation:UnsafeToDelete()
    if unsafe_msg then error(unsafe_msg, 3) end
  end
end


function Bran:generateInserts()
  local bran = self
  assert(bran.location == L.CPU)

  local rel, ast_nodes = next(bran.kernel.inserts)
  bran.insert_data = {
    relation = rel,
    record_type = ast_nodes[1].record_type,
    n_inserted  = L.NewGlobal(L.addr, 0),
  }
  -- register the global variable
  bran:getGlobalId(bran.insert_data.n_inserted)

  -- prep all the fields we want to be able to write to.
  for _,field in ipairs(rel._fields) do
    bran:getFieldId(field)
  end
  bran:getFieldId(rel._is_live_mask)
  bran.signature_template:addInsertion()
end

function Bran:generateDeletes()
  local bran = self
  assert(bran.location == L.CPU)

  local rel = next(bran.kernel.deletes)
  bran.delete_data = {
    relation = rel,
    updated_size = L.NewGlobal(L.addr, 0)
  }
  -- register global variable
  bran:getGlobalId(bran.delete_data.updated_size)
end
