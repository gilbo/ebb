local K = {}
package.loaded["compiler.kernel"] = K

-- Use the following to produce
-- deterministic order of table entries
-- From the Lua Documentation
function pairs_sorted(tbl, compare)
  local arr = {}
  for k in pairs(tbl) do table.insert(arr, k) end
  table.sort(arr, compare)

  local i = 0
  local iter = function() -- iterator
    i = i + 1
    if arr[i] == nil then return nil
    else return arr[i], tbl[arr[i]] end
  end
  return iter
end


local L = terralib.require "compiler.lisztlib"
local specialization = terralib.require "compiler.specialization"
local semant  = terralib.require "compiler.semant"
local phase   = terralib.require "compiler.phase"
local codegen = terralib.require "compiler.codegen"


-------------------------------------------------------------------------------
--[[ Kernels, Brans, Germs                                                 ]]--
-------------------------------------------------------------------------------
--[[

We use a Kernel as the primary unit of computation.
  For internal use, we define the related concepts of Germ and Bran

((
etymology:
  a Germ is the plant embryo within a kernel,
  a Bran is the outer part of a kernel, encasing the germ and endosperm
))

A Germ -- a Terra struct.
          It provides a dynamic context at execution time.
          Example entries:
            - number of rows in the relation
            - subset masks
            - field data pointers

A Bran -- a Lua table
          It provides metadata about a particular kernel specialization.
          e.g. one bran for each (kernel, runtime, subset) tuple
          Examples entries:
            - signature params: (relation, subset)
            - a germ
            - executable function
            - field/phase signature

Each Kernel may have many Brans, each a compile-time specialization
Each Bran may have a different assignment of Germ values for each execution

]]--

local Bran = {}
Bran.__index = Bran

function Bran.New()
  return setmetatable({}, Bran)
end

-- Seedbank is a cache of brans
local Seedbank = {}
local function seedbank_lookup(sig)
  local str_sig = ''
  for k,v in pairs_sorted(sig) do
    str_sig = str_sig .. k .. '=' .. tostring(v) .. ';'
  end
  local bran = Seedbank[str_sig]
  if not bran then
    bran = Bran.New()
    for k,v in pairs(sig) do bran[k] = v end
    Seedbank[str_sig] = bran
  end
  return bran
end

local MAX_FIELDS = 32
-- Germ Terra Structs
local struct GermField {
    data : &opaque;
}
local taddr = uint64 --L.addr:terraType() -- weird dependency error
local struct GermSubset {
    index : &taddr;
    boolmask : &bool;
    use_boolmask : bool;
    use_index : bool;
    index_size : uint64;
}
local struct Germ {
    n_rows : uint64;
    subset : GermSubset;
    n_fields : int; -- # of fields referred to, i.e. below
    fields : GermField[MAX_FIELDS];
}

-------------------------------------------------------------------------------
--[[ Kernels                                                               ]]--
-------------------------------------------------------------------------------


function L.NewKernel(kernel_ast, env)
    local new_kernel = setmetatable({}, L.LKernel)

    -- All declaration time processing here
    local specialized    = specialization.specialize(env, kernel_ast)
    new_kernel.typed_ast = semant.check(env, specialized)
    new_kernel.field_use = phase.phasePass(new_kernel.typed_ast)

    return new_kernel
end


L.LKernel.__call  = function (kobj, relset)
    if not (relset and (L.is_relation(relset) or L.is_subset(relset)))
    then
        error("A kernel must be called on a relation or subset.", 2)
    end


    -- retreive the correct bran or create a new one
    local bran = seedbank_lookup({
        kernel=kobj,
        relset=relset
    })
    if not bran.executable then
      bran.relset = relset
      bran.kernel = kobj
      bran:generate()
    end


    -- set execution parameters in the germ
    local germ = bran.germ:get()
    germ.n_rows   = bran.relation:Size()
    germ.n_fields = bran.n_field_ids
    -- set up the subset
      -- defaults
    bran:getSubsetGerm().use_boolmask = false
    bran:getSubsetGerm().use_index    = false
    if bran.subset then
      if bran.subset._boolmask then
        bran:getSubsetGerm().use_boolmask = true
        bran:getSubsetGerm().boolmask = bran.subset._boolmask.data
      elseif bran.subset._index then
        bran:getSubsetGerm().use_index = true
        bran:getSubsetGerm().index = bran.subset._index._data
        bran:getSubsetGerm().index_size = bran.subset._index._size
      end
    end
    -- bind the field data
    for field, _ in pairs(bran.field_ids) do
      bran:getFieldGerm(field).data = field.data
    end

    -- launch the kernel
    bran.executable()
end

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
      error('Kernels may only be called on a relation they were typed with')
  end

  -- initialize the Germ and germ related metadata
  bran.germ         = global(Germ)
  bran.field_ids    = {}
  bran.n_field_ids  = 0 -- for safety

  -- fix the mapping for the fields before compiling the executable
  for field, _ in pairs(kernel.field_use) do
    bran:getFieldGerm(field)
  end

  -- compile an executable
  bran.executable = codegen.codegen(typed_ast, bran)
end

function Bran:getFieldGerm(field)
  -- get the id for the bran
  local id = self.field_ids[field]
  if not self.field_ids[field] then
    if self.executable then
      error('INTERNAL ERROR: cannot add new fields after compilation')
    end
    id = self.n_field_ids
    self.field_ids[field] = id
    self.n_field_ids = id + 1
  end

  return self.germ:get().fields[id]
end

function Bran:getSubsetGerm()
  return self.germ:get().subset
end

--function Bran:setFieldData(field_name, value)
--  --local field_name = field:Name()
--  local id = self.field_id[field_name]
--  local germ = self.germ:get()
--  germ.fields[id].data = value
--end
--
--function Bran:fieldCode(field_name)
--  local id = self.field_id[field_name]
--  local germ = self.germ
--  return `germ.fields[id].data
--end




