
local S = {}
package.loaded["compiler.specialization"] = S

local ast = require "compiler.ast"
local B   = require "compiler.builtins"
local T   = require "compiler.types"




------------------------------------------------------------------------------
--[[ Context Definition                                                   ]]--
------------------------------------------------------------------------------
local Context = {}
Context.__index = Context

function Context.new(env, diag)
  local ctxt = setmetatable({
    env         = env,
    diag        = diag
  }, Context)
  return ctxt
end
function Context:liszt()
  return self.env:localenv()
end
function Context:lua()
  return self.env:luaenv()
end
function Context:enterblock()
  self.env:enterblock()
end
function Context:leaveblock()
  self.env:leaveblock()
end
function Context:error(ast, ...)
  self.diag:reporterror(ast, ...)
end





------------------------------------------------------------------------------
--[[ specialization:                                                      ]]--
------------------------------------------------------------------------------
function S.specialize(luaenv, kernel_ast)
  local env  = terralib.newenvironment(luaenv)
  local diag = terralib.newdiagnostics()
  local ctxt = Context.new(env, diag)

  diag:begin()
  env:enterblock()
  local new_ast = kernel_ast:specialize(ctxt)
  env:leaveblock()
  diag:finishandabortiferrors("Errors during specializing Liszt", 1)

  return new_ast
end




local function exec_external(exp, ctxt, default)
  local status, v = pcall(function()
    return exp(ctxt:lua())
  end)
  if not status then
    ctxt:error(exp, "Error evaluating lua expression")
    v = default
  end
  return v
end

local function exec_type_annotation(typexp, ast_node, ctxt)
  local typ = exec_external(typexp, ctxt, L.error)
  if not T.isLisztType(typ) then
    ctxt:error(ast_node, "Expected Liszt type annotation but found " ..
                         type(typ))
    typ = L.error
  end
  return typ
end





------------------------------------------------------------------------------
--[[ AST Structural Junk:                                                 ]]--
------------------------------------------------------------------------------
--function ast.AST:specialize(ctxt)
--  return self:passthrough('specialize', ctxt)
--
--  --error("Specialization not implemented for AST node " .. self.kind)
--end

ast.NewCopyPass('specialize')

-- Passthrough
--  ast.Block
--  ast.ExprStatement

function ast.IfStatement:specialize(ctxt)
  local ifstmt = self:clone()

  ifstmt.if_blocks = {}
  for id, node in ipairs(self.if_blocks) do
    ctxt:enterblock()
    ifstmt.if_blocks[id] = node:specialize(ctxt)
    ctxt:leaveblock()
  end
  if self.else_block then
    ctxt:enterblock()
    ifstmt.else_block = self.else_block:specialize(ctxt)
    ctxt:leaveblock()
  end

  return ifstmt
end

function ast.WhileStatement:specialize(ctxt)
  local whilestmt = self:clone()

  whilestmt.cond = self.cond:specialize(ctxt)

  ctxt:enterblock()
  whilestmt.body = self.body:specialize(ctxt)
  ctxt:leaveblock()

  return whilestmt
end

function ast.DoStatement:specialize(ctxt)
  local dostmt = self:clone()

  ctxt:enterblock()
  dostmt.body = self.body:specialize(ctxt)
  ctxt:leaveblock()

  return dostmt
end

function ast.RepeatStatement:specialize(ctxt)
  local repeatstmt = self:clone()

  ctxt:enterblock()
    repeatstmt.body = self.body:specialize(ctxt)
    repeatstmt.cond = self.cond:specialize(ctxt)
  ctxt:leaveblock()

  return repeatstmt
end

function ast.CondBlock:specialize(ctxt)
  local condblk   = self:clone()
  condblk.cond    = self.cond:specialize(ctxt)

  ctxt:enterblock()
  condblk.body = self.body:specialize(ctxt)
  ctxt:leaveblock()

  return condblk
end

function ast.Quote:specialize(ctxt)
  -- Once quotes are typed, they should be fully specialized
  -- Don't repeat the specialization at that point, since it
  -- could cause some sort of hiccup? (dunno)
  if self.node_type then
    return self
  else
    local q = self:clone()
    q.code  = self.code:specialize(ctxt)
    return q
  end
end

function ast.LetExpr:specialize(ctxt)
  local let = self:clone()

  ctxt:enterblock()
  let.block = self.block:specialize(ctxt)
  let.exp   = self.exp:specialize(ctxt)
  ctxt:leaveblock()

  return let
end





------------------------------------------------------------------------------
--[[ AST Name Related:                                                    ]]--
------------------------------------------------------------------------------
-- We must re-write names with symbols in this pass...

function ast.DeclStatement:specialize(ctxt)
  local decl = self:clone()
  -- assert(self.typeexpression or self.initializer)

  -- SPECIALIZATION
  -- process any explicit type annotation
  if self.typeexpression then
    decl.node_type = exec_type_annotation(self.typeexpression, self, ctxt)
  end

  if self.initializer then
    decl.initializer = self.initializer:specialize(ctxt)
  end

  -- REGISTER SYMBOL
  local name_str = decl.name
  decl.name = ast.GenSymbol(name_str)
  ctxt:liszt()[name_str] = decl.name

  return decl
end

function ast.NumericFor:specialize(ctxt)
  local numfor     = self:clone()
  numfor.lower     = self.lower:specialize(ctxt)
  numfor.upper     = self.upper:specialize(ctxt)

  if self.step then
    numfor.step = self.step:specialize(ctxt)
  end

  ctxt:enterblock()
  -- REGISTER SYMBOL
  local name_str = numfor.name
  numfor.name = ast.GenSymbol(name_str)
  ctxt:liszt()[name_str] = numfor.name
  numfor.body = self.body:specialize(ctxt)
  ctxt:leaveblock()

  return numfor
end

function ast.GenericFor:specialize(ctxt)
  local r = self:clone()
  r.set   = self.set:specialize(ctxt)

  -- NOTE: UNHANDLED.
  -- The projection chain in the set along with the referred
  -- relation should be baked in here.
  -- However, that seems tricky/difficult to me right now...
  --local rel = r.set.node_type.relation
  --for i,p in ipairs(r.set.node_type.projections) do
  --    if not rel[p] then
  --        ctxt:error(self,"Could not find field '"..p.."'")
  --        return r
  --    end
  --    rel = rel[p].type.relation
  --    assert(rel)
  --end

  ctxt:enterblock()
  -- REGISTER SYMBOL
  local name_str = r.name
  r.name = ast.GenSymbol(name_str)
  ctxt:liszt()[name_str] = r.name
  r.body = self.body:specialize(ctxt)
  ctxt:leaveblock()

  return r
end

function ast.LisztKernel:specialize(ctxt)
  local kernel                = self:clone()

  kernel.set                  = self.set:specialize(ctxt)

  -- REGISTER SYMBOL
  local name_str              = kernel.name
  kernel.name                 = ast.GenSymbol(name_str)
  ctxt:liszt()[name_str]      = kernel.name
  kernel.body                 = self.body:specialize(ctxt)

  return kernel
end




------------------------------------------------------------------------------
--[[ AST NAME:                                                            ]]--
------------------------------------------------------------------------------

local function NewLuaObject(anchor, obj)
  local lo     = ast.LuaObject:DeriveFrom(anchor)
  lo.node_type = L.internal(obj)
  return lo
end

-- This function attempts to produce an AST node which looks as if
-- the resulting AST subtree has just been emitted from the Parser
local function luav_to_ast(luav, src_node)
  -- try to construct an ast node to return...
  local node

  -- Global objects are replaced with special Global nodes
  if L.is_global(luav) then
    node        = ast.Global:DeriveFrom(src_node)
    node.global = luav

  -- Vector objects are expanded into literal AST trees
  elseif L.is_vector(luav) then
    node            = ast.VectorLiteral:DeriveFrom(src_node)
    node.elems      = {}
    -- We have to copy the type here b/c the values
    -- may not imply the right type
    node.node_type  = luav.type
    for i,v in ipairs(luav.data) do
      node.elems[i] = luav_to_ast(v, src_node)
      node.elems[i].node_type = luav.type:baseType()
    end

  elseif B.isBuiltin(luav) then
    node = NewLuaObject(src_node,luav)
  elseif L.is_relation(luav) then
    node = NewLuaObject(src_node,luav)
  elseif L.is_user_func(luav) then
    node = NewLuaObject(src_node,luav)
  elseif L.is_macro(luav) then
    node = NewLuaObject(src_node,luav)
  elseif terralib.isfunction(luav) then
    node = NewLuaObject(src_node,B.terra_to_func(luav))
  elseif type(luav) == 'table' then
    -- Determine whether this is an AST node
    if ast.is_ast(luav) and luav:is(ast.Quote) then
      node = luav
    else
      node = NewLuaObject(src_node, luav)
    end

  elseif type(luav) == 'number' then
    node       = ast.Number:DeriveFrom(src_node)
    node.value = luav
  elseif type(luav) == 'boolean' then
    node       = ast.Bool:DeriveFrom(src_node)
    node.value = luav

  else
    return nil
  end

  -- return the constructed node if we made it here
  return node
end

function ast.Name:specialize(ctxt)
  -- try to find the name in the local Liszt scope
  local symbol = ctxt:liszt()[self.name]
  if symbol then
    local name_clone = self:clone()
    name_clone.name  = symbol
    return name_clone
  end

  -- Otherwise, does the name exist in the lua scope?
  local luav = ctxt:lua()[self.name]
  if luav ~= nil then
    -- convert the lua value into an ast node
    local ast = luav_to_ast(luav, self)
    if ast then
      return ast
    else
      ctxt:error(self, "could not convert Lua value of '"..self.name.."' "..
                       "to a Liszt value")
      return self:clone()
    end
  end

  -- Otherwise, failed to find this name anywhere
  ctxt:error(self, "variable '" .. self.name .. "' is not defined")
  return self:clone()
end


function ast.TableLookup:specialize(ctxt)
  local tab = self.table:specialize(ctxt)
  local member = self.member

  -- use of the .[lua_expression] syntax
  -- the lua expression must be evaluated into a string
  if type(member) == "function" then
    member = exec_external(member, ctxt, "<error>")
    if type(member) ~= "string" then
      ctxt:error(self,"expected escape to evaluate to a string but found ",
                      type(member))
      member = "<error>"
    end
  end

  -- internal type node
  if ast.is_ast(tab) and tab:is(ast.LuaObject) then
    local thetable = tab.node_type.value

    -- lookup
    local luaval = thetable[member]
    if luaval == nil then
      ctxt:error(self, "lua table does not have member '" .. member .. "'")
      return self
    end

    -- convert the value
    local ast = luav_to_ast(luaval, self)
    if ast then
      return ast
    else
      ctxt:error(self, "The table member '"..member.."' could not be "..
                       "resolved to a Liszt value")
      return self
    end
  else
    -- assume we're ok for now
    local lookup = self:clone()
    lookup.table = tab
    lookup.member = member
    return lookup
  end
end



------------------------------------------------------------------------------
--[[ AST USER FUNCTIONS:                                                  ]]--
------------------------------------------------------------------------------

function ast.UserFunction:specialize(ctxt)
  local func = self:clone()

  ctxt:enterblock()
  -- register symbols for params
  local params = {}
  for i=1,#(self.params) do
    local name_str = self.params[i]
    params[i] = ast.GenSymbol(name_str)
    ctxt:liszt()[name_str] = params[i]
  end

  local body = self.body:specialize(ctxt)
  local exp = nil
  if self.exp then
    exp  = self.exp:specialize(ctxt)
  end
  ctxt:leaveblock()

  func.params, func.body, func.exp = params, body, exp
  return func
end










