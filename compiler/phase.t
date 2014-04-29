local Phase = {}
package.loaded['compiler.phase'] = Phase


local ast = require "compiler.ast"


------------------------------------------------------------------------------
--[[ Phase Types                                                          ]]--
------------------------------------------------------------------------------

local PhaseType = {}
PhaseType.__index = PhaseType
local PT = PhaseType

-- phase kind constants
PT.READ        = {}
PT.REDUCE      = {}
PT.EXCLUSIVE   = {} -- write puts us here immediately
PT.ERROR       = {}
-- Ordering of lattice constants
-- READ < EXCLUSIVE < ERROR
-- REDUCE < EXCLUSIVE
-- READ | REDUCE (incomparable)

function PhaseType:__tostring()
  local center = ''
  if self.is_centered then center = '_or_EXCLUSIVE' end

  if self.kind == PT.READ then
    return 'READ' .. center
  elseif self.kind == PT.REDUCE then
    return 'REDUCE('..self.reduceop..')' .. center
  elseif self.kind == PT.EXCLUSIVE then
    return 'EXCLUSIVE'
  else
    return 'ERROR'
  end
end

function PhaseType.New(kind, opts)
  opts = opts or {}
  local pt = setmetatable({
    kind = kind,
    reduceop = opts.reduceop,
    is_centered = opts.is_centered
  }, PhaseType)

  -- promote uncentered exclusive access to error
  if pt.kind == PT.EXCLUSIVE and not pt.is_centered then
    pt.kind = PT.ERROR
  end

  return pt
end

function PhaseType:isError()
  return self.kind == PT.ERROR
end

function PhaseType:join(rhs)
  local kind
  if self.kind == PT.ERROR or rhs.kind == PT.ERROR then
    kind = PT.ERROR
  elseif self.kind == rhs.kind then
    kind = self.kind
  else
    kind = PT.EXCLUSIVE
  end

  -- promote if there's a reduce conflict
  if kind == PT.REDUCE and self.reduceop ~= rhs.reduceop then
    kind = PT.EXCLUSIVE
  end

  return PhaseType.New(kind, {
    reduceop      = self.reduceop,
    is_centered   = self.is_centered and rhs.is_centered
  })
end

-- Global variables can't be used in exclusive mode
function PhaseType:globals_join(rhs)
  if self.kind == rhs.kind and
     (self.kind ~= PT.REDUCE or self.reduceop == rhs.reduceop)
  then
    return PhaseType.New(self.kind, {reduceop=self.reduceop})
  end

  -- otherwise
  return PhaseType.New(PT.ERROR)
end 

------------------------------------------------------------------------------
--[[ Context:                                                             ]]--
------------------------------------------------------------------------------

local Context = {}
Context.__index = Context

function Context.new(env, diag)
  local ctxt = setmetatable({
    --env     = env,
    diag    = diag,
    fields  = {},
    globals = {}
  }, Context)
  return ctxt
end
function Context:error(ast, ...)
  self.diag:reporterror(ast, ...)
end

function Context:logfield(field, phase_type, node)
  -- Create an entry for the field
  local lookup = self.fields[field]

  -- if this access was an error and is the first error
  if phase_type:isError() then
    if not (lookup and lookup.phase_type:isError()) then
      self:error(node, 'Non-Exclusive WRITE')
    end
  end

  -- first access
  if not lookup then
    lookup = {
      phase_type = phase_type,
      last_access = node,
    }
    self.fields[field] = lookup
  -- later accesses
  else
    local join_type = lookup.phase_type:join(phase_type)
    -- if first error, then...
    if join_type:isError() and
       not (phase_type:isError() or lookup.phase_type:isError())
    then
      local lastfile = lookup.last_access.filename
      local lastline = lookup.last_access.linenumber
      self:error(node, tostring(phase_type)..' Phase is'..
                                             ' incompatible with\n'..
                       lastfile..':'..lastline..': '..
                       tostring(lookup.phase_type)..' Phase\n')
    end
    lookup.phase_type  = join_type
    lookup.last_access = node
  end
end

function Context:logglobal(global, phase_type, node)
  local lookup = self.globals[global]

  -- first access
  if not lookup then
    lookup = {
      phase_type = phase_type,
      last_access = node,
    }
    self.globals[global] = lookup
  else
    local join_type = lookup.phase_type:globals_join(phase_type)
    if join_type:isError() and
      not (phase_type:isError() or lookup.phase_type:isError())
    then
      local lastfile = lookup.last_access.filename
      local lastline = lookup.last_access.linenumber
      self:error(node, tostring(phase_type)..' Phase for Global is'..
                                             ' incompatible with\n'..
                       lastfile..':'..lastline..': '..
                       tostring(lookup.phase_type)..' Phase for Global\n')
    end
    lookup.phase_type = join_type
    lookup.last_access = node
  end
end

function Context:dumpFieldTypes()
  local res = {}
  for k,record in pairs(self.fields) do
    res[k] = record.phase_type
  end
  return res
end


------------------------------------------------------------------------------
--[[ Phase Pass:                                                          ]]--
------------------------------------------------------------------------------

function Phase.phasePass(kernel_ast)
  local env  = terralib.newenvironment(nil)
  local diag = terralib.newdiagnostics()
  local ctxt = Context.new(env, diag)

  diag:begin()
    kernel_ast:phasePass(ctxt)
  diag:finishandabortiferrors("Errors during phasechecking Liszt", 1)

  local field_use = ctxt:dumpFieldTypes()

  return field_use
end


------------------------------------------------------------------------------
--[[ AST Nodes:                                                           ]]--
------------------------------------------------------------------------------

function ast.AST:phasePass(ctxt)
  self:callthrough('phasePass', ctxt)
end




function ast.FieldWrite:phasePass (ctxt)
  -- We intentionally skip over the Field Access here...
  self.fieldaccess.row:phasePass(ctxt)

  local reduceop = self.reduceop
  local centered = self.fieldaccess.row.is_centered
  local kind     = (self.reduceop and PT.REDUCE) or PT.EXCLUSIVE
  local ptype    = PT.New(kind, {reduceop=reduceop, is_centered=centered})

  local field = self.fieldaccess.field
  ctxt:logfield(field, ptype, self)

  self.exp:phasePass(ctxt)
end

function ast.FieldAccess:phasePass (ctxt)
  self.row:phasePass(ctxt)

  local centered = self.row.is_centered
  local ptype    = PT.New(PT.READ, {is_centered=centered})
  ctxt:logfield(self.field, ptype, self)
end


function ast.Call:phasePass (ctxt)
  for i,p in ipairs(self.params) do
    -- Terra Funcs may write or do other nasty things...
    if self.func.is_a_terra_func and p:is(ast.FieldAccess) then
      p.row:phasePass()

      local centered = p.row.is_centered
      local ptype    = PT.New(PT.EXCLUSIVE, {is_centered=centered})
      ctxt:logfield(p.field, ptype, p)
    elseif self.func.is_a_terra_func and p:is(ast.Global) then
      self:error(p, 'Unable to verify that global field will not be '..
                    'written by external function call.')
    else
      p:phasePass(ctxt)
    end
  end
end


function ast.GlobalReduce:phasePass(ctxt)
  local ptype = PT.New(PT.REDUCE,{reduceop=self.reduceop})
  local global = self.global.global
  ctxt:logglobal(global, ptype, self)

  self.exp:phasePass(ctxt)
end

function ast.Global:phasePass (ctxt)
  -- if we got here, it wasn't through a write or reduce use
  ctxt:logglobal(self.global, PT.New(PT.READ), self)
end


function ast.Where:phasePass(ctxt)
  -- Would like to log that there was a use of the index...?

  -- Which field is the index effectively having us read?
  local ptype = PT.New(PT.READ)
  local field = self.relation._grouping.key_field
  ctxt:logfield(field, ptype, self)

  self.key:phasePass(ctxt)
end

function ast.GenericFor:phasePass(ctxt)
  self.set:phasePass(ctxt)
  -- assert(self.set.node_type:isQuery())

  -- deal with any field accesses implied by projection
  local rel = self.set.node_type.relation
  for i,p in ipairs(self.set.node_type.projections) do
    local field = rel[p]
    ctxt:logfield(field, PT.New(PT.READ), self)

    rel = field.type.relation
  end

  self.body:phasePass(ctxt)
end



