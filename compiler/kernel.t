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

local MAX_FIELDS = 16
-- Germ Terra Structs
local struct GermField {
    data : &opaque;
}
local struct Germ {
    n_rows : uint64;
    fields : GermField[MAX_FIELDS];
}

-------------------------------------------------------------------------------
--[[ Kernels                                                               ]]--
-------------------------------------------------------------------------------


function L.NewKernel(kernel_ast, env)
    local new_kernel = setmetatable({}, L.LKernel)

    -- All declaration time processing here
    new_kernel.typed_ast = semant.check(env, kernel_ast)
    phase.phasePass(new_kernel.typed_ast)

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

    if L.is_relation(relset) then
        germ.n_rows = relset:Size()
    elseif L.is_subset(relset) then
        germ.n_rows = relset:Relation():Size()
        -- bind in the boolmask for the subset
        bran:setFieldData('_boolmask', relset._boolmask.data)
    end


    -- launch the kernel
    bran.executable()
end

function Bran:generate()
  local bran      = self
  local kernel    = bran.kernel
  local typed_ast = bran.kernel.typed_ast

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


  -- initialize the Germ
  bran.germ = global(Germ)
  bran.field_id = {
    _boolmask = 0
  }

  -- compile an executable
  bran.executable = codegen.codegen(typed_ast, bran)
end

function Bran:setFieldData(field_name, value)
  local id = self.field_id[field_name]
  local germ = self.germ:get()
  germ.fields[id].data = value
end

function Bran:fieldCode(field_name)
  local id = self.field_id[field_name]
  local germ = self.germ
  return `germ.fields[id].data
end




