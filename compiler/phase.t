local Phase = {}
package.loaded['compiler.phase'] = Phase


local ast = require "compiler.ast"


------------------------------------------------------------------------------
--[[ Phase Types                                                          ]]--
------------------------------------------------------------------------------

local PhaseType = {}
PhaseType.__index = PhaseType
local PT = PhaseType

function PhaseType.New(params)
  local pt = setmetatable({
    read        = params.read,
    reduceop    = params.reduceop,
    write       = params.write,
    centered    = params.centered, -- i.e. centered
  }, PhaseType)
  return pt
end

-- does not pay attention to whether or not we're centered
function PhaseType:requiresExclusive()
  if self.write then return true end
  if self.read and self.reduceop then return true end
  if self.reduceop == 'multiop' then return true end
  return false
end

function PhaseType:isCentered()
  return self.centered
end

function PhaseType:isError()
  return not self.centered and self:requiresExclusive()
end

function PhaseType:__tostring()
  if self:isError() then return 'ERROR' end
  if self:requiresExclusive() then return 'EXCLUSIVE' end

  local centered = ''
  if self.is_centered then centered = '_or_EXCLUSIVE' end

  if self.read then
    return 'READ' .. centered
  elseif self.reduceop then
    return 'REDUCE('..self.reduceop..')' .. centered
  end

  -- should never reach here
  return 'ERROR'
end

function PhaseType:join(rhs)
  local lhs = self

  local args = {
    read  = lhs.read  or rhs.read,
    write = lhs.write or rhs.write,
  }
  if lhs.centered and rhs.centered then args.centered = true end
  if lhs.reduceop or  rhs.reduceop then
    if lhs.reduceop and rhs.reduceop and lhs.reduceop ~= rhs.reduceop then
      args.reduceop = 'multiop'
    else
      args.reduceop = lhs.reduceop or rhs.reduceop
    end
  end

  return PhaseType.New(args)
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
    globals = {},
    inserts = {},
    deletes = {},
  }, Context)
  return ctxt
end
function Context:error(ast, ...)
  self.diag:reporterror(ast, ...)
end



local function log_helper(ctxt, is_field, f_or_g, phase_type, node)
  -- Create an entry for the field or global
  local cache = ctxt.globals
  if is_field then cache = ctxt.fields end
  local lookup = cache[f_or_g]

  -- if this access was an error and is the first error
  if phase_type:isError() then
    if not (lookup and lookup.phase_type:isError()) then
      ctxt:error(node, 'Non-Exclusive WRITE')
    end
  end

  -- check if this field access conflicts with insertion
  if is_field then
    local insert = ctxt.inserts[f_or_g:Relation()]
    if insert then
      local insertfile = insert.last_access.filename
      local insertline = insert.last_access.linenumber
      ctxt:error(node,
        'Cannot access field '..f_or_g:FullName()..' while inserting\n('..
        insertfile..':'..insertline..') into relation '..
        f_or_g:Relation():Name())
    end
  end

  -- first access
  if not lookup then
    lookup = {
      phase_type = phase_type,
      last_access = node,
    }
    cache[f_or_g] = lookup
  -- later accesses
  else
    local join_type = lookup.phase_type:join(phase_type)
    -- if first error, then...
    if join_type:isError() and
       not (phase_type:isError() or lookup.phase_type:isError())
    then
      local lastfile = lookup.last_access.filename
      local lastline = lookup.last_access.linenumber
      local g_opt = ' for Global'
      if is_field then g_opt = '' end
      ctxt:error(node, tostring(phase_type)..' Phase'..g_opt..' is'..
                                             ' incompatible with\n'..
                       lastfile..':'..lastline..': '..
                       tostring(lookup.phase_type)..' Phase'..g_opt..'\n')
    end
    lookup.phase_type  = join_type
    lookup.last_access = node
  end
end

function Context:logfield(field, phase_type, node)
  log_helper(self, true, field, phase_type, node)
end

function Context:logglobal(global, phase_type, node)
  log_helper(self, false, global, phase_type, node)
end

function Context:loginsert(relation, node)

  -- check that none of the relation's fields have been accessed
  for field,record in pairs(self.fields) do
    if relation == field:Relation() then
      local insertfile = node.filename
      local insertline = node.linenumber
      self:error(record.last_access,
        'Cannot access field '..field:FullName()..' while inserting\n('..
        insertfile..':'..insertline..') into relation '..relation:Name())
      return
    end
  end

  -- check that the relation being mapped over isn't being inserted into
  if self.relation == relation then
    self:error(node, 'Cannot insert into relation '..relation:Name()..
               ' while mapping over it')
  end

  -- check that this is the only insert for this relation
  if self.inserts[relation] then
    self:error(node, 'Cannot insert into relation '..relation:Name()..' twice')
  end

  -- register insertion
  self.inserts[relation] = {
    last_access = node
  }
end

function Context:logdelete(relation, node)
  -- check that the row is centered happens in type-checking pass

  -- log exclusive write accesses for all the relation's fields
  for _,f in ipairs(relation._fields) do
    self:logfield(f, PhaseType.New {
      write = true,
      centered = true
    }, node)
  end

  -- check that this is the only delete for this kernel
  -- since only the relation mapped over can possibly be deleted from
  -- this check suffices
  if self.deletes[relation] then
    self:error(node,
      'Temporary: can only have one delete statement per kernel')
  end

  -- register the deletion
  self.deletes[relation] = {
    last_access = node
  }
end

function Context:dumpFieldTypes()
  local res = {}
  for k,record in pairs(self.fields) do
    res[k] = record.phase_type
  end
  return res
end

function Context:dumpGlobalTypes()
  local res = {}
  for k, record in pairs(self.globals) do
    res[k] = record.phase_type
  end
  return res
end

function Context:dumpInserts()
  local ret = {}
  for relation,record in pairs(self.inserts) do
    -- ASSUME THERE IS ONLY ONE INSERT
    ret[relation] = {record.last_access} -- list of AST nodes for inserts
  end
  if next(ret) == nil then return nil end -- return nil if nothing
  return ret
end

function Context:dumpDeletes()
  local ret = {}
  for relation,record in pairs(self.deletes) do
    -- ASSUME UNIQUE INSERT PER KERNEL
    ret[relation] = {record.last_access} -- list of AST nodes for deletes
  end
  if next(ret) == nil then return nil end -- return nil if nothing
  return ret
end

------------------------------------------------------------------------------
--[[ Phase Pass:                                                          ]]--
------------------------------------------------------------------------------

function Phase.phasePass(kernel_ast)
  local env  = terralib.newenvironment(nil)
  local diag = terralib.newdiagnostics()
  local ctxt = Context.new(env, diag)

  -- record the relation being mapped over
  ctxt.relation = kernel_ast.relation

  diag:begin()
    kernel_ast:phasePass(ctxt)
  diag:finishandabortiferrors("Errors during phasechecking Liszt", 1)

  local field_use   = ctxt:dumpFieldTypes()
  local global_use  = ctxt:dumpGlobalTypes()
  local inserts     = ctxt:dumpInserts()
  local deletes     = ctxt:dumpDeletes()

  return {
    field_use   = field_use,
    global_use  = global_use,
    inserts     = inserts,
    deletes     = deletes,
  }
end


------------------------------------------------------------------------------
--[[ AST Nodes:                                                           ]]--
------------------------------------------------------------------------------

ast.NewInertPass('phasePass')




function ast.FieldWrite:phasePass (ctxt)
  -- We intentionally skip over the Field Access here...
  self.fieldaccess.row:phasePass(ctxt)

  local pargs    = { centered = self.fieldaccess.row.is_centered }
  if self.reduceop then
    pargs.reduceop = self.reduceop
  else
    pargs.write    = true
  end
  local ptype    = PhaseType.New(pargs)

  local field    = self.fieldaccess.field
  ctxt:logfield(field, ptype, self)

  self.exp:phasePass(ctxt)
end

function ast.FieldAccess:phasePass (ctxt)
  self.row:phasePass(ctxt)

  local ptype = PhaseType.New {
    centered = self.row.is_centered,
    read = true
  }
  ctxt:logfield(self.field, ptype, self)
end


function ast.Call:phasePass (ctxt)
  for i,p in ipairs(self.params) do
    -- Terra Funcs may write or do other nasty things...
    if self.func.is_a_terra_func and p:is(ast.FieldAccess) then
      p.row:phasePass()

      local ptype = PhaseType.New {
        write = true, read = true, -- since we can't tell for calls!
        centered = p.row.is_centered,
      }
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
  local ptype = PhaseType.New { reduceop = self.reduceop }
  local global = self.global.global
  ctxt:logglobal(global, ptype, self)

  self.exp:phasePass(ctxt)
end

function ast.Global:phasePass (ctxt)
  -- if we got here, it wasn't through a write or reduce use
  ctxt:logglobal(self.global, PhaseType.New { read = true } , self)
end


function ast.Where:phasePass(ctxt)
  -- Would like to log that there was a use of the index...?

  -- Which field is the index effectively having us read?
  local ptype = PhaseType.New{ read = true }
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
    ctxt:logfield(field, PhaseType.New { read = true }, self)

    rel = field.type.relation
  end

  self.body:phasePass(ctxt)
end

--------------------------------
-- handle inserts and deletes

function ast.InsertStatement:phasePass(ctxt)
  self.record:phasePass(ctxt)
  self.relation:phasePass(ctxt)

  local relation = self.relation.node_type.value
  local unsafe_msg = relation:UnsafeToInsert(self.record_type)
  if unsafe_msg then
      ctxt:error(self,unsafe_msg)
  end

  -- log the insertion
  ctxt:loginsert(relation, self)
end

function ast.DeleteStatement:phasePass(ctxt)
  self.row:phasePass(ctxt)

  local relation = self.row.node_type.relation
  local unsafe_msg = relation:UnsafeToDelete()
  if unsafe_msg then
    ctxt:error(self, unsafe_msg)
  end

  -- log the delete
  ctxt:logdelete(relation, self)
end










