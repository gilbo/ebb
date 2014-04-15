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
  }, Context)
  return ctxt
end
--function Context:localenv()
--  --return self.env:localenv()
--end
--function Context:enterblock()
--  --self.env:enterblock()
--end
--function Context:leaveblock()
--  --self.env:leaveblock()
--end
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
    else
      p:phasePass(ctxt)
    end
  end
end


function ast.Global:phasePass (ctxt)
  --local d = self.global.data
  --local s = symbol(&self.global.type:terraType())
  --return `@d
end


function ast.Where:phasePass(ctxt)
  -- Would like to log that there was a use of the index...?

  -- Which field is the index effectively having us read?
  local ptype = PT.New(PT.READ)
  local field = self.relation._index
  ctxt:logfield(field, ptype, self)

  self.key:phasePass(ctxt)
end

function ast.GenericFor:phasePass(ctxt)
  self.set:phasePass(ctxt)
  -- assert(self.set.node_type:isQuery())

  -- TODO: Make Projections Count as Field-READs

  -- deal with any field accesses implied by projection
  --local rel = r.set.node_type.relation
  --for i,p in ipairs(r.set.node_type.projections) do
  --    rel = rel[p].type.relation
  --    assert(rel)
  --end

  self.body:phasePass(ctxt)
end



