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
local S = {}
package.loaded["ebb.src.semant"] = S

local Pre = require "ebb.src.prelude"
local ast = require "ebb.src.ast"
local B   = require "ebb.src.builtins"
local T   = require "ebb.src.types"
local R   = require 'ebb.src.relations'
local F   = require 'ebb.src.functions'
--[[
  AST:check(ctxt) type checking routines
    These methods define type checking.
    Each check() method is designed to accept a typing context (see below).
    Once called, check() will construct and return a new type-checked
    version of its subtree (new AST nodes)
--]]

local errorT    = T.error
local floatT    = T.float
local doubleT   = T.double
local intT      = T.int
local uint64T   = T.uint64
local boolT     = T.bool

local keyT      = T.key
local internalT = T.internal
local recordT   = T.record
local queryT    = T.query
local vectorT   = T.vector
local matrixT   = T.matrix

--local is_relation   = R.is_relation
--local is_builtin    = B.is_builtin
local is_macro      = Pre.is_macro
--local is_field      = R.is_field
--local is_function   = F.is_function

--local isfielddispatcher = R.isfielddispatcher


local MAX_INT_32, MIN_INT_32 = math.pow(2,31)-1, -math.pow(2,31)
local function is_int32(x)
  return type(x) == 'number' and x % 1 == 0 and
         x <= MAX_INT_32 and x >= MIN_INT_32
end

------------------------------------------------------------------------------
--[[ Context Definition                                                   ]]--
------------------------------------------------------------------------------
--[[
  A Context is passed through type-checking, keeping track of any kind of
    store or gadget we want to use, such as the environment object and
    error diagnostic object. It can be used to quickly thread new stores
    through the entire typechecker.

--]]

local Context = {}
Context.__index = Context

function Context.new(env, diag)
  local ctxt = setmetatable({
    env         = env,
    diag        = diag,
    loop_count  = 0,
    centers     = {}, -- variable symbols bound to the centered key
  }, Context)
  return ctxt
end
function Context:ebb()
  return self.env:localenv()
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
function Context:in_loop()
  return self.loop_count > 0
end
function Context:enterloop()
  self.loop_count = self.loop_count + 1
end
function Context:leaveloop()
  self.loop_count = self.loop_count - 1
  if self.loop_count < 0 then self.loop_count = 0 end
end
function Context:recordcenter(sym)
  self.centers[sym] = true
end
function Context:iscenter(sym)
  return self.centers[sym]
end

-- Terrible hacks to get info about immediate parent use context
function Context:setWriteHint(node)
  self._write_hint = node
end
function Context:checkWriteHint(node)
  return self._write_hint == node
end
function Context:setReduceHint(node, op)
  self._reduce_hint_node = node
  self._reduce_hint_op   = op
end
function Context:checkReduceHint(node)
  if self._reduce_hint_node == node then return self._reduce_hint_op end
end

local function try_coerce(target_typ, node, ctxt)
  if node.node_type == target_typ then
    return node -- simple-case, avoid superfluous casts
  elseif node.node_type:isCoercableTo(target_typ) then
    local cast = ast.Cast:DeriveFrom(node)
    cast.node_type  = target_typ
    cast.value      = node
    return cast
  else
    if ctxt then
      ctxt:error(node, "Could not coerce expression of type '"..
                       tostring(node.node_type) .. "' into type '"..
                       tostring(target_typ) .. "'")
    end
    return nil
  end
end


------------------------------------------------------------------------------
--[[ Substitution Functions:                                              ]]--
------------------------------------------------------------------------------

local function QuoteParams(all_params_asts)
  local quoted_params = {}
  for i,param_ast in ipairs(all_params_asts) do
    assert(ast.is_ast(param_ast), 'INTERNAL: parameters not ASTs?')
    if param_ast:is(ast.Quote) then
      quoted_params[i] = param_ast
    else
      local q = ast.Quote:DeriveFrom(param_ast)
      q.code = param_ast
      if param_ast.is_centered then q.is_centered = true end
      q.node_type = param_ast.node_type -- halt type-checking here
      quoted_params[i] = q
    end
  end
  return quoted_params
end

local function RunMacro(ctxt,src_node,the_macro,params)
  local result = the_macro.genfunc(unpack(QuoteParams(params)))

  if ast.is_ast(result) and result:is(ast.Quote) then
    return result
  else
    ctxt:error(src_node, 'Macros must return quoted code')
    local errnode     = src_node:clone()
    errnode.node_type = errorT
    return errnode
  end
end

local function InlineUserFunc(ctxt, src_node, the_func, param_asts)
  -- note that this alpha rename is safe because the
  -- param_asts haven't been attached yet.  We know from
  -- specialization that there are no free variables in the function.
  local f = the_func._decl_ast:alpha_rename()

  -- check that the correct number of arguments are provided
  if #(f.params) ~= #param_asts then
    ctxt:error(src_node,
        'Expected '..tostring(#(f.params))..' arguments, '..
        'but was supplied '..tostring(#param_asts)..' arguments')
    local errnode     = src_node:clone()
    errnode.node_type = errorT
    return errnode
  end

  -- bind params to variables (use a quote to fake it?)
  local statements = {}
  local string_literals = {}
  for i,p_ast in ipairs(QuoteParams(param_asts)) do
    -- exception for strings
    local ptype     = param_asts[i].node_type
    local argname   = f.params[i]
    if ptype:isinternal() and type(ptype.value) == 'string' then
      string_literals[argname] = ptype
    else
      local decl = ast.DeclStatement:DeriveFrom(src_node)
      decl.name = argname
      decl.initializer = p_ast
      table.insert(statements, decl)
    end
  end

  -- now add the function body statements to the list
  for _,stmt in ipairs(f.body.statements) do
    table.insert(statements, stmt)
  end
  local block = ast.Block:DeriveFrom(src_node)
  block.statements = statements

  -- ultimately, the user func call is translated into a let expression
  -- which looks like
  -- LET
  --   var param_1 = ...
  --   var param_2 = ...
  --   ...
  --   <body>
  -- IN
  --   <exp>
  -- unless we're missing an expression to return, in which
  -- case we expand into a do-statement instead
  local expansion = ast.LetExpr:DeriveFrom(src_node)
  expansion.block = block
  expansion.exp   = f.exp

  -- Lacking an expression, use a DO block instead
  if not f.exp then
    expansion = ast.DoStatement:DeriveFrom(src_node)
    expansion.body = block
    -- hack to make checker barf when this is used as an expression
    expansion.node_type = internalT('no-return function')
  end

  -- wrap up in a quote and typecheck
  local q = ast.Quote:DeriveFrom(src_node)
  q.code = expansion
  ctxt:enterblock()
    for name,strtype in pairs(string_literals) do
        ctxt:ebb()[name] = strtype
    end
    local qchecked = q:check(ctxt)
  ctxt:leaveblock()

  return qchecked
end



------------------------------------------------------------------------------
--[[ AST semantic checking methods:                                       ]]--
------------------------------------------------------------------------------
function ast.AST:check(ctxt)
  error("Typechecking not implemented for AST node " .. self.kind)
end

function ast.Block:check(ctxt)
  local copy = self:clone()

  copy.statements = {}
  for i,node in ipairs(self.statements) do
    copy.statements[i] = self.statements[i]:check(ctxt)
  end

  return copy
end

function ast.IfStatement:check(ctxt)
  local ifstmt = self:clone()

  ifstmt.if_blocks = {}
  for id, node in ipairs(self.if_blocks) do
    ctxt:enterblock()
    ifstmt.if_blocks[id] = node:check(ctxt)
    ctxt:leaveblock()
  end
  if self.else_block then
    ctxt:enterblock()
    ifstmt.else_block = self.else_block:check(ctxt)
    ctxt:leaveblock()
  end

  return ifstmt
end

function ast.WhileStatement:check(ctxt)
  local whilestmt = self:clone()

  whilestmt.cond = self.cond:check(ctxt)
  local condtype = whilestmt.cond.node_type
  if condtype ~= errorT and condtype ~= boolT then
    ctxt:error(self, "expected bool expression but found " ..
                     tostring(condtype) )
  end

  ctxt:enterblock()
  ctxt:enterloop()
  whilestmt.body = self.body:check(ctxt)
  ctxt:leaveloop()
  ctxt:leaveblock()

  return whilestmt
end

function ast.DoStatement:check(ctxt)
  local dostmt = self:clone()

  ctxt:enterblock()
  dostmt.body = self.body:check(ctxt)
  ctxt:leaveblock()

  return dostmt
end

function ast.RepeatStatement:check(ctxt)
  local repeatstmt = self:clone()

  ctxt:enterblock()
  ctxt:enterloop()
  repeatstmt.body = self.body:check(ctxt)
  repeatstmt.cond = self.cond:check(ctxt)
  local condtype  = repeatstmt.cond.node_type

  if condtype ~= errorT and condtype ~= boolT then
    ctxt:error(self, "expected bool expression but found " ..
                     tostring(condtype) )
  end
  ctxt:leaveloop()
  ctxt:leaveblock()

  return repeatstmt
end

function ast.ExprStatement:check(ctxt)
  local copy = self:clone()
  copy.exp   = self.exp:check(ctxt)
  return copy
end

-- Do not allow subtractions and divsions, b/c
-- 1) simplification
-- 2) divisions of integers are not even approximately associative
local reductions_by_type = {
  [floatT] = {
    ['+']   = true,
    --['-']   = true,
    ['*']   = true,
    --['/']   = true,
    ['min'] = true,
    ['max'] = true,
  },
  [doubleT] = {
    ['+']   = true,
    --['-']   = true,
    ['*']   = true,
    --['/']   = true,
    ['min'] = true,
    ['max'] = true,
  },
  [intT] = {
    ['min'] = true,
    ['max'] = true,
    ['+']   = true,
    --['-']   = true,
    ['*']   = true,
    --['/']   = true,
  },
  [uint64T] = {
    ['min'] = true,
    ['max'] = true,
    ['+']   = true,
    --['-']   = true,
    ['*']   = true,
    --['/']   = true,
  },
  [boolT] = {
    ['and'] = true,
    ['or']  = true,
  },
}

function check_reduce(node, ctxt)
  local op        = node.reduceop
  local lvalue    = node.lvalue
  local ltype     = lvalue.node_type

  -- Centered reductions don't need to be atomic!
  if (node.lvalue:is(ast.FieldAccess) or
      node.lvalue:is(ast.FieldAccessIndex))
    and node.lvalue.key.is_centered
  then return end

  if op == nil then return end

  local reductions = reductions_by_type[ltype:basetype()]
  if not reductions or not reductions[op] then
    ctxt:error(node, 'Reduce operator "'..op..'" for type '..
                     '"'..tostring(ltype)..'" is not currently supported.')
  end
end

function ast.Assignment:check(ctxt)
  local node = self:clone()

  -- LHS tracked for phase-checking
  if self.reduceop then -- signal context to child
    ctxt:setReduceHint(self.lvalue, self.reduceop)
  else
    ctxt:setWriteHint(self.lvalue)
  end
  node.lvalue = self.lvalue:check(ctxt)

  local ltype  = node.lvalue.node_type
  if ltype == errorT then return node end

  node.exp     = self.exp:check(ctxt)
  local rtype  = node.exp.node_type
  if rtype == errorT then return node end

  -- check for write/reduce functions...
  if ltype:isinternal() then
    local func, key = ltype.value[1], ltype.value[2]
    if not node.lvalue.from_dispatch then
      ctxt:error(self.lvalue, "Illegal assignment: left hand side cannot "..
                              "be assigned")
      return node
    end
    assert(F.is_function(func))
    return InlineUserFunc(ctxt, self, func, { key, node.exp })
  end

  -- Promote global lhs to lvalue if there was a reduction
  if (node.lvalue:is(ast.Global) or node.lvalue:is(ast.GlobalIndex)) and
    not self.reduceop then
    ctxt:error(self.lvalue, "Cannot write to globals in functions")
    node.lvalue.is_lvalue = true
  end

  -- enforce that the lhs is an lvalue
  if not node.lvalue.is_lvalue then
    ctxt:error(self.lvalue, "Illegal assignment: left hand side cannot "..
                            "be assigned")
    return node
  -- How should we restrict assignments to keys?
  elseif node.lvalue.node_type:iskey() and
         ( not ( node.lvalue:is(ast.FieldAccess) or
                 ( node.lvalue:is(ast.FieldAccessIndex) and
                   node.lvalue.base and
                   node.lvalue.base:is(ast.FieldAccess)
                 )
               )
         )
  then
    ctxt:error(self.lvalue, "Illegal assignment: variables of key type "..
                            "cannot be re-assigned")
    return node
  end

  -- handle any coercions
  if ltype ~= rtype then
    node.exp = try_coerce(ltype, node.exp, ctxt)
    if not node.exp then return node end
  end

  -- replace assignment with a global reduce if we see a
  -- global on the left hand side
  if node.lvalue:is(ast.Global)  or node.lvalue:is(ast.GlobalIndex) then
    check_reduce(node, ctxt)
    local gr = ast.GlobalReduce:DeriveFrom(node)
    gr.global      = node.lvalue
    gr.exp         = node.exp
    gr.reduceop    = node.reduceop
    return gr
  -- replace assignment with a field write if we see a
  -- field access on the left hand side
  elseif node.lvalue:is(ast.FieldAccess) or
         node.lvalue:is(ast.FieldAccessIndex)
  then
    check_reduce(node, ctxt)
    local fw = ast.FieldWrite:DeriveFrom(node)
    fw.fieldaccess = node.lvalue
    fw.exp         = node.exp
    fw.reduceop    = node.reduceop
    return fw
  end

  return node
end


function ast.DeclStatement:check(ctxt)
  local decl = self:clone()
  --assert(self.node_type or self.initializer)
  --assert(not self.typeexpression)

  -- process any explicit type annotation
  if self.node_type then
    decl.node_type = self.node_type
  end
  -- check the initialization against the annotation or infer type.
  if self.initializer then
    decl.initializer = self.initializer:check(ctxt)
    local exptyp     = decl.initializer.node_type
    -- if the type was annotated handle any coercions
    if decl.node_type then
      decl.initializer = try_coerce(decl.node_type, decl.initializer, ctxt)
    else
      decl.node_type = exptyp
    end
    -- if the rhs is a centered key, try to propagate that information
    -- NOTE: this pseudo-constant propagation is sound b/c
    -- we don't allow re-assignment of key-type variables
    if exptyp:isscalarkey() and decl.initializer.is_centered then
      ctxt:recordcenter(decl.name)
    end
  end

  if decl.node_type ~= errorT and not decl.node_type:isfieldvalue() then
    ctxt:error(self,"can only assign numbers, bools, "..
                    "or keys to local temporaries")
  end

  ctxt:ebb()[decl.name] = decl.node_type

  return decl
end

function ast.NumericFor:check(ctxt)
  local node = self
  local function check_num_type(tp)
    if tp ~= errorT and not tp:isintegral() then
      ctxt:error(node, "expected an integer-type expression to define the "..
                       "iterator bounds/step (found "..
                       tostring(tp)..")")
    end
  end

  local numfor     = self:clone()
  numfor.lower     = self.lower:check(ctxt)
  numfor.upper     = self.upper:check(ctxt)
  local lower_type = numfor.lower.node_type
  local upper_type = numfor.upper.node_type
  check_num_type(lower_type)
  check_num_type(upper_type)
  if lower_type ~= upper_type then -- sanity check!
    ctxt:error(node, 'iterator bound types must match!')
  end

  local step, step_type
  if self.step then
    numfor.step = self.step:check(ctxt)
    step_type   = numfor.step.node_type
    check_num_type(step_type)
    if lower_type ~= step_type then -- sanity check!
        ctxt:error(node, 'iterator bound types must match!')
    end
  end

  -- infer iterator type
  numfor.name           = self.name
  numfor.node_type      = lower_type

  ctxt:enterblock()
  ctxt:enterloop()
  ctxt:ebb()[numfor.name] = numfor.node_type
  numfor.body = self.body:check(ctxt)
  ctxt:leaveloop()
  ctxt:leaveblock()
  return numfor
end

function ast.GenericFor:check(ctxt)
  local r = self:clone()
  r.name  = self.name
  r.set   = self.set:check(ctxt)
  if not r.set.node_type:isquery() then
    ctxt:error(self,"for statement expects a query but found type "..
                    tostring(r.set.node_type))
    return r
  end
  local rel = r.set.node_type.relation
  for i,p in ipairs(r.set.node_type.projections) do
    if not rel[p] then
      ctxt:error(self,"Could not find field '"..p.."'")
      return r
    end
    assert(R.is_field(rel[p]))
    rel = rel[p]:Type().relation
    assert(rel)
  end
  local keyType = keyT(rel)
  ctxt:enterblock()
  ctxt:enterloop()
  ctxt:ebb()[r.name] = keyType
  r.body = self.body:check(ctxt)
  r.node_type = keyType
  ctxt:leaveloop()
  ctxt:leaveblock()
  return r
end

function ast.Break:check(ctxt)
  if not ctxt:in_loop() then
    ctxt:error(self, "cannot have a break statement outside a loop")
  end
  return self:clone()
end

function ast.InsertStatement:check(ctxt)
  local insert        = self:clone()

  -- check relation
  insert.relation     = self.relation:check(ctxt)
  local reltyp        = insert.relation.node_type
  local rel           = reltyp:isinternal() and reltyp.value
  if not rel or not R.is_relation(rel) then
    ctxt:error(self,"Expected a relation to insert into")
    return insert
  end
  -- check record child
  local record        = self.record:clone()
  insert.record       = record
  record.names        = self.record.names
  record.exprs        = {}

  local rectyp_proto  = {}
  insert.fieldindex   = {}
  for i,name in ipairs(self.record.names) do
    local exp           = self.record.exprs[i]:check(ctxt)
    local field         = rel[name]

    if not field then
      ctxt:error(self, 'cannot insert a value into field '..
                       rel:Name()..'.'..name..' because it is undefined')
      return insert
    end
    -- coercion?
    if field:Type() ~= exp.node_type then
      exp = try_coerce(field:Type(), exp, ctxt)
      if not exp then return insert end
    end
    if not exp or exp.node_type == errorT then return insert end

    record.exprs[i]     = exp
    rectyp_proto[name]  = exp.node_type
    insert.fieldindex[field] = i
  end

  -- save record type
  record.node_type    = recordT(rectyp_proto)
  insert.record_type  = record.node_type
  -- type compatibility with the relation will be checked
  -- in the phase pass

  return insert
end

function ast.DeleteStatement:check(ctxt)
  local delete = self:clone()

  delete.key   = self.key:check(ctxt)
  local keytyp = delete.key.node_type

  if not keytyp:isscalarkey() or not delete.key.is_centered then
    ctxt:error(self,"Only centered keys may be deleted")
    return delete
  end

  return delete
end

function ast.CondBlock:check(ctxt)
  local new_node  = self:clone()
  new_node.cond   = self.cond:check(ctxt)
  local condtype  = new_node.cond.node_type
  if condtype ~= errorT and condtype ~= boolT then
    ctxt:error(self, "conditional expression type should be "..
                     "boolean (found " .. tostring(condtype) .. ")")
  end

  ctxt:enterblock()
  new_node.body = self.body:check(ctxt)
  ctxt:leaveblock()

  return new_node
end


------------------------------------------------------------------------------
--[[                         Expression Checking:                         ]]--
------------------------------------------------------------------------------
function ast.Expression:check(ctxt)
  error("Semantic checking has not been implemented for "..
        "expression type " .. self.kind, 2)
end



local is_ord_op = {
  ['<='] = true,
  ['>='] = true,
  ['>']  = true,
  ['<']  = true
}

local function matching_type_dims(t1, t2)
  if t1:isscalar() and t2:isscalar() then return true end
  if t1:isvector() and t2:isvector() then
    return t1.N == t2.N
  elseif t1:ismatrix() and t2:ismatrix() then
    return t1.Nrow == t2.Nrow and t1.Ncol == t2.Ncol
  end
  return false
end

local function err (node, ctx, msg)
  node.node_type = errorT
  if msg then ctx:error(node, msg) end
  return node
end


-- NOTE THIS IS UNSAFE.  Caller must check
-- whether or not the coercion is type-safe
local function coerce_base(btyp, node)
  local ntyp = node.node_type
  if btyp == ntyp:basetype() then return node end

  local cast = ast.Cast:DeriveFrom(node)
  if ntyp:isscalar()   then  cast.node_type = btyp end
  if ntyp:isvector()      then  cast.node_type = vectorT(btyp, ntyp.N) end
  if ntyp:ismatrix() then
    cast.node_type = matrixT(btyp, ntyp.Nrow, ntyp.Ncol)
  end
  cast.value = node
  return cast
end
-- Will coerce base type but not the underlying 
local function try_bin_coerce(binop, errf)
  local join = T.type_join(binop.lhs.node_type:basetype(),
                           binop.rhs.node_type:basetype())
  if join == errorT then return errf() end

  binop.lhs = coerce_base(join, binop.lhs)
  binop.rhs = coerce_base(join, binop.rhs)
  binop.node_type = T.type_join(binop.lhs.node_type, binop.rhs.node_type)

  return binop
end
local function try_bin_coerce_bool(binop, errf)
  local node = try_bin_coerce(binop, errf)
  if node.node_type == errorT then return node end
  node.node_type = boolT
  return node
end
local function try_mat_prod_coerce(binop, errf, N, M)
  local join = T.type_join(binop.lhs.node_type:basetype(),
                           binop.rhs.node_type:basetype())
  if join == errorT then return errf() end
  binop.lhs = coerce_base(join, binop.lhs)
  binop.rhs = coerce_base(join, binop.rhs)
  if M == nil then binop.node_type = vectorT(join, N)
              else binop.node_type = matrixT(join, N, M) end
  return binop
end

-- binary expressions
function ast.BinaryOp:check(ctxt)
  local binop         = self:clone()
  binop.op            = self.op
  binop.lhs           = self.lhs:check(ctxt)
  binop.rhs           = self.rhs:check(ctxt)

  local lt, rt  = binop.lhs.node_type, binop.rhs.node_type
  local op            = binop.op
  -- Silently ignore/propagate errors
  if lt == errorT or rt == errorT then return err(self, ctxt) end

  -- error messages
  local function type_err()
    return err(binop, ctxt, 'incompatible types: ' .. tostring(lt) ..
                            ' and ' .. tostring(rt))
  end
  local function op_err()
    return err(binop, ctxt, 'invalid types for operator \'' .. binop.op ..
                '\': ' .. tostring(lt) .. ' and ' .. tostring(rt) )
  end

  -- special case for key types
  if lt:iskey() and rt:iskey() and (op == '==' or op == '~=') then
    if lt ~= rt then return type_err() end
    binop.node_type = boolT
    return binop
  end

  -- prohibit min/max outside of reductions
  if op == 'min' or op == 'max' then
    return err(binop, ctxt, op .. ' is unsupported as a binary operator')
  end

  -- Now make sure we have value types
  if not lt:isvalue() or not rt:isvalue() then
    return op_err()
  end

  -- Given that we're working with value types, there are three properties
  -- that we need to consider in deciding whether or not this binary op
  -- should type check:
  --    1. OP:       (below)
  --    2. BASETYPE: bool, int, float, double, uint64
  --    3. DIM:      scalar, vector(n), matrix(n,m)
  -- OPS:
  --    logical(and,or),
  --    Eq(==,~=),
  --    Ord(<=,>=,>,<)
  --    + -
  --    *
  --    /
  --    Mod(%)


  -- OP: Logical
    -- DIM: L = R
      -- BASETYPE L/R: Logical
        -- OK, return Copy type
  if op == 'and' or op == 'or' then
    if not lt:islogical() or not rt:islogical() then return op_err() end
    if not matching_type_dims(lt,rt) then return type_err() end
    binop.node_type = lt
    return binop
  end

  -- OP: Ord
    -- DIM L/R: scalar
      -- BASETYPE L/R: Numeric
        -- COERCE, return BOOL
  if is_ord_op[op] then
    if not (lt:isnumeric() and lt:isprimitive() and
            rt:isnumeric() and rt:isprimitive()) then return op_err() end
    return try_bin_coerce_bool(binop, type_err)
  end

  -- OP: Mod
    -- DIM L/R: scalar
      -- BASETYPE L/R: Integral
        -- COERCE, return coercion
  if op == '%' then
    if not (lt:isintegral() and lt:isprimitive() and
            rt:isintegral() and rt:isprimitive()) then return op_err() end
    return try_bin_coerce(binop, type_err)
  end

  -- OP: Eq
    -- DIM: L = R
      -- BASETYPE: all
        -- COERCE, return BOOL
  if op == '==' or op == '~=' then
    if not matching_type_dims(lt,rt) then return type_err() end
    return try_bin_coerce_bool(binop, type_err)
  end

  -- OP: + -
    -- DIM: L = R
      -- BASETYPE: Numeric
        -- COERCE, return coerced type
  if op == '+' or op == '-' then
    if not lt:isnumeric() or not rt:isnumeric() then return op_err() end
    if not matching_type_dims(lt,rt) then return type_err() end
    return try_bin_coerce(binop, type_err)
  end

  -- OP: *
    -- BASETYPE: Numeric
      -- DIM: Scalar _
      -- DIM: _ Scalar
        -- COERCE, return coerced type
      -- DIM: Vector(n) Matrix(n,_)
      -- DIM: Matrix(_,m) Vector(m)
      -- DIM: Matrix(_,m) Matrix(m,_)
        -- COERCE, BUT return correctly dimensioned type
  if op == '*' then
    if not lt:isnumeric() or not rt:isnumeric() then return op_err() end
    if lt:isprimitive() or rt:isprimitive() then
      return try_bin_coerce(binop, type_err)

--    elseif lt:isvector() and rt:ismatrix() and lt.N == rt.Nrow then
--      return try_mat_prod_coerce(binop, type_err, rt.Ncol, nil)
--
--    elseif lt:ismatrix() and rt:isvector() and lt.Ncol == rt.N then
--      return try_mat_prod_coerce(binop, type_err, lt.Nrow, nil)
--
--    elseif lt:ismatrix() and rt:ismatrix() and lt.Ncol == rt.Nrow
--    then
--      return try_mat_prod_coerce(binop, type_err, lt.Nrow, rt.Ncol)

    else
      return op_err()
    end
  end

  -- OP: /
    -- BASETYPE: Numeric
      -- DIM: _ Scalar
        -- COERCE, return coerced type
  if op == '/' then
    if not lt:isnumeric() or not rt:isnumeric() then return op_err() end
    if not rt:isprimitive()                     then return op_err() end

    return try_bin_coerce(binop, type_err)
  end

  -- unrecognized op error
  return err(binop, ctxt, "Failed to recognize operator '"..binop.op.."'")
end

function ast.UnaryOp:check(ctxt)
  local unop     = self:clone()
  unop.op        = self.op
  unop.exp       = self.exp:check(ctxt)
  local exptype  = unop.exp.node_type
  unop.node_type = errorT -- default

  if unop.op == 'not' then
    if exptype:islogical() then
        unop.node_type = exptype
    else
        ctxt:error(self, "unary \"not\" expects a boolean operand")
    end
  elseif unop.op == '-' then
    if exptype:isnumeric() then
        unop.node_type = exptype
    else
        ctxt:error(self, "unary minus expects a numeric operand")
    end
  else
    ctxt:error(self, "unknown unary operator \'".. self.op .."\'")
  end

  return unop
end

local function NewLuaObject(anchor, obj)
  local lo     = ast.LuaObject:DeriveFrom(anchor)
  lo.node_type = internalT(obj)
  return lo
end

------------------------------------------------------------------------------
--[[                               Variables                              ]]--
------------------------------------------------------------------------------

function ast.Name:check(ctxt)
  -- try to find the name in the local Ebb scope
  local typ = ctxt:ebb()[self.name]
  -- if the name is in the local scope, then it must have been declared
  -- somewhere in the ebb function.  Thus, it has to be a primitive, a
  -- bool, or a topological element.
  if typ then
    local node     = ast.Name:DeriveFrom(self)
    node.node_type = typ
    node.name      = self.name
    if ctxt:iscenter(self.name) then
      node.is_centered = true
    end
    return node
  end

  -- Lua environment variables should already have been handled
  -- during specialization

  -- failed to find this name anywhere
  ctxt:error(self, "variable '" .. tostring(self.name) .. "' is not defined")
  local err_node     = self:clone()
  err_node.name      = self.name
  err_node.node_type = errorT
  return err_node
end

function ast.Number:check(ctxt)
  local number = self:clone()
  number.value = self.value
  if self.node_type then
    number.node_type = self.node_type
  elseif self.valuetype then
    if     self.valuetype == int then
        number.node_type = intT
    elseif self.valuetype == double then
        number.node_type = doubleT
    elseif self.valuetype == float then
        number.node_type = floatT
    elseif self.valuetype == uint64 then
        number.node_type = uint64T
    else
        ctxt:error(self, "numeric literal type unsupported by Ebb")
        number.node_type = errorT
    end
  elseif is_int32(self.value) then
    number.node_type = intT
  else
    number.node_type = doubleT
  end
  return number
end

function ast.Bool:check(ctxt)
  local boolnode      = self:clone()
  boolnode.node_type  = boolT
  return boolnode
end

function ast.String:check(ctxt)
  local strnode       = self:clone()
  strnode.node_type   = internalT(self.value)
  return strnode
end

function convert_to_matrix_literal(literal, ctxt)
  local matlit        = ast.MatrixLiteral:DeriveFrom(literal)
  matlit.elems        = {}

  local tp_error  = "matrix literals can only contain vectors"
  local mt_error  = "matrix entries must be of the same type"
  local dim_error = "matrix literals must contain vectors of the same size"

  local vecs = literal.elems
  matlit.n   = #vecs
  matlit.m   = #(vecs[1].elems)
  local max_type = vecs[1].node_type:basetype()

  -- take max type
  for i=2,matlit.n do
    local tp = vecs[i].node_type
    if not tp:isvector() then return err(literal, ctxt, tp_error) end
    if tp.N ~= matlit.m  then return err(literal, ctxt, dim_error) end

    if max_type:isCoercableTo(tp:basetype()) then
      max_type = tp:basetype()
    elseif not tp:basetype():isCoercableTo(max_type) then
      return err(literal, ctxt, mt_error)
    end
  end

  -- if the type was explicitly provided...
  if literal.node_type then
    max_type = literal.node_type:basetype()
  end

  -- coerce and re-marshall the entries
  for i = 1, matlit.n do
    for j = 1, matlit.m do
      local idx = (i-1)*matlit.m + (j-1) + 1
      matlit.elems[idx] = try_coerce(max_type, vecs[i].elems[j], ctxt)
    end
  end

  matlit.node_type = matrixT(max_type, matlit.n, matlit.m)
  return matlit
end

function ast.VectorLiteral:check(ctxt)
  local veclit      = self:clone()
  veclit.elems      = {}

  local tp_error = "vector literals can only contain scalar values"
  local mt_error = "vector entries must be of the same type"

  -- recursively process children
  for i=1,#self.elems do
    veclit.elems[i] = self.elems[i]:check(ctxt)
    local tp        = veclit.elems[i].node_type
    if tp == errorT then return err(self, ctxt) end
  end

  -- check if matrix applies...
  local max_type  = veclit.elems[1].node_type
  if max_type:isvector() then
    if self.node_type then veclit.node_type = self.node_type end
    -- SPECIAL CASE: a vector of vectors is a matrix and
    --               needs to be treated specially
    return convert_to_matrix_literal(veclit, ctxt)
  elseif not max_type:isscalar() then
    return err(self, ctxt, tp_error)
  end

  -- compute a max type
  for i = 2, #self.elems do
    local tp = veclit.elems[i].node_type
    if not tp:isscalar() then
      return err(self, ctxt, tp_error)
    end

    if max_type:isCoercableTo(tp) then
      max_type = tp
    elseif not tp:isCoercableTo(max_type) then
      return err(self, ctxt, mt_error)
    end
  end

  -- If the type was explicitly provided...
  if self.node_type then
    max_type = self.node_type:basetype()
  end

  -- now coerce all of the entries into the max type
  for i = 1, #veclit.elems do
    veclit.elems[i] = try_coerce(max_type, veclit.elems[i], ctxt)
  end

  veclit.node_type = vectorT(max_type, #veclit.elems)
  return veclit
end

--function ast.RecordLiteral:check(ctxt)
--    local record      = self:clone()
--    record.names      = self.names
--    record.exprs      = {}
--
--    local rectyp_proto = {}
--    local found_error = false
--    for i,n in ipairs(self.names) do
--        local e = self.exprs[i]
--
--        local exp           = e:check(ctxt)
--        record.exprs[i]     = exp
--        rectyp_proto[n]     = exp.node_type
--        if exp.node_type == errorT then found_error = true end
--    end
--
--    if found_error then
--        record.node_type = errorT
--    else
--        record.node_type = recordT(rectyp_proto)
--    end
--    return record
--end



----------------------------
--[[ AST Alpha-Renaming ]]--
----------------------------
  
  ast.NewCopyPass('alpha_rename')
  
  function ast.UserFunction:alpha_rename()
    local symbol_remap = {}
    local func = self:clone()

    func.params = {}
    func.ptypes = {}
    for i=1,#(self.params) do
      func.params[i] = self.params[i]:UniqueCopy()
      func.ptypes[i] = self.ptypes[i]
      symbol_remap[self.params[i]] = func.params[i]
    end

    func.body = self.body:alpha_rename(symbol_remap)
    if self.exp then
      func.exp = self.exp:alpha_rename(symbol_remap)
    end

    return func
  end

  function ast.DeclStatement:alpha_rename(remap)
    local decl = self:clone()
    if self.initializer then
      decl.initializer = self.initializer:alpha_rename(remap)
    end

    decl.name = self.name:UniqueCopy()
    remap[self.name] = decl.name
    return decl
  end

  function ast.NumericFor:alpha_rename(remap)
    local numfor        = self:clone()
    numfor.lower        = self.lower:alpha_rename(remap)
    numfor.upper        = self.upper:alpha_rename(remap)

    if self.step then numfor.step = self.step:alpha_rename(remap) end

    numfor.name = self.name:UniqueCopy()
    remap[self.name] = numfor.name

    numfor.body = self.body:alpha_rename(remap)

    return numfor
  end

  function ast.GenericFor:alpha_rename(remap)
    local r = self:clone()
    r.set   = self.set:alpha_rename(remap)

    r.name  = self.name:UniqueCopy()
    remap[self.name] = r.name

    r.body  = self.body:alpha_rename(remap)

    return r
  end

  function ast.RepeatStatement:alpha_rename(remap)
    local r = self:clone()
    r.body = self.body:alpha_rename(remap)
    r.cond = self.cond:alpha_rename(remap)
    return r
  end

  function ast.Name:alpha_rename(remap)
    local n = self:clone()
    n.name = remap[self.name] -- remapped symbol
    return n
  end

------------------------------------------------------------------------------
--[[                         Miscellaneous nodes:                         ]]--
------------------------------------------------------------------------------



function ast.TableLookup:check(ctxt)
  local tab = self.table:check(ctxt)
  local member = self.member
  local ttype = tab.node_type

  if ttype == errorT then
    return err(self, ctxt)
  end

  if ttype:isscalarkey() then
    local luaval = ttype.relation[member]

    -- create a field access normally
    if R.is_field(luaval) then
      local field         = luaval
      local ast_node      = ast.FieldAccess:DeriveFrom(tab)
      ast_node.name       = member
      ast_node.key        = tab
      local name = ast_node.key.name
      if name and ctxt:iscenter(name) then
          ast_node.key.is_centered = true
      end
      ast_node.field      = field
      ast_node.node_type  = field:Type()
      return ast_node

    -- desugar macro-fields from key.macro to macro(key)
    elseif is_macro(luaval) then
      return RunMacro(ctxt,self,luaval,{tab})
    -- desugar function-fields from key.func to func(key)
    elseif R.isfielddispatcher(luaval) then
      if ctxt:checkWriteHint(self) then
        if not luaval._writer then
          return err(self, ctxt, "relation "..ttype.relation:Name()..
            " does not have a field write function '"..member.."'")
        end
        local obj = NewLuaObject(self, { luaval._writer, tab })
        obj.from_dispatch = true
        return obj
      elseif ctxt:checkReduceHint(self) then
        local op = ctxt:checkReduceHint(self)
        if not luaval._reducers[op] then
          return err(self, ctxt, "relation "..ttype.relation:Name()..
            " does not have a field write function '"..member.."' "..
            "for reduction '"..op.."'")
        end
        local obj = NewLuaObject(self, { luaval._reducers[op], tab })
        obj.from_dispatch = true
        return obj
      else
        if not luaval._reader then
          return err(self, ctxt, "relation "..ttype.relation:Name()..
            " does not have a field read function '"..member.."'")
        end
        return InlineUserFunc(ctxt,self,luaval._reader,{tab})
      end
    elseif F.is_function(luaval) then
      return InlineUserFunc(ctxt,self,luaval,{tab})
    else
      return err(self, ctxt, "Key from "..ttype.relation:Name()..
                             " does not have field or macro-field "..
                             "'"..member.."'")
    end
  elseif ttype:isquery() then
    local rel = ttype.relation
    local ct  = tab:clone()
    for k,v in pairs(tab) do ct[k] = v end
    local projs = {}
    for i,p in ipairs(ttype.projections) do
      table.insert(projs,p)
      rel = rel[p]:Type().relation 
      assert(rel)
    end
    local field = rel[member]
    if not R.is_field(field) then
      ctxt:error(self, "Relation "..rel:Name()..
                       " does not have field "..member)
    else 
      table.insert(projs,member)
    end
    ct.node_type = queryT(ttype.relation,projs)
    return ct
  else
    return err(self, ctxt, "select operator not "..
                           "supported for "..
                           tostring(ttype))
  end
end

local function SqIdxVecMat(self, base, ctxt)
  -- mutate squareindex into a node of type base

  local sqidx = nil
  if base:is(ast.FieldAccess) then
    sqidx = ast.FieldAccessIndex:DeriveFrom(base)
    sqidx.name  = base.member
    sqidx.key   = base.key
    sqidx.field = base.field
    sqidx.node_type = base.node_type
  elseif base:is(ast.Global) then
    sqidx = ast.GlobalIndex:DeriveFrom(base)
    sqidx.node_type = base.node_type
    sqidx.global    = base.global
  else
    sqidx = self:clone()
  end

  local idx  = self.index:check(ctxt)
  sqidx.base, sqidx.index = base, idx
  local btyp, ityp = base.node_type, idx.node_type

  if btyp == errorT or ityp == errorT then
    sqidx.node_type = errorT
    return sqidx
  end

  if not ityp:isintegral() then return err(idx, ctxt, 'expected an '..
      'integer index, but found '..tostring(ityp)) end

  -- matrix case
  if self.index2 then
    local idx2      = self.index2:check(ctxt)
    sqidx.index2     = idx2
    local i2typ     = idx2.node_type

    if i2typ == errorT then
      sqidx.node_type = errorT
      return sqidx
    end

    if not ityp:isintegral() then return err(idx2, ctxt, 'expected an '..
        'integer index, but found '..tostring(i2typ)) end
    if not btyp:ismatrix() then return err(base, ctxt, 'expected '..
        'small matrix to index into, not '.. tostring(btyp)) end
  -- vector case
  else
    if not btyp:isvector() then return err(base, ctxt, 'expected '..
        'vector to index into, not '..tostring(btyp)) end
  end

  -- is an lvalue only when the base is
  if base.is_lvalue then sqidx.is_lvalue = true end
  sqidx.node_type = btyp:basetype()
  return sqidx
end

local function SqIdxKey(self, base, ctxt)
  local lookup    = ast.TableLookup:DeriveFrom(self)
  lookup.table    = self.base -- assign unchecked version...

  -- make sure we have a string we're indexing with
  local stringobj = self.index:check(ctxt)
  local stype     = stringobj.node_type
  if not stype:isinternal() or type(stype.value) ~= 'string' then
    ctxt:error(self.index, 'Expecting string literal to index key')
    lookup.node_type = errorT
    return lookup
  end
  local str = stype.value

  -- re-direct to the full table-lookup logic
  lookup.member   = str
  return lookup:check(ctxt)
end

function ast.SquareIndex:check(ctxt)
  -- Square indices come up in two cases:
  --  1. vector or matrix entries being extracted
  --  2. accessing fields from a key using a string literal
  
  local base  = self.base:check(ctxt)
  if base.node_type:ismatrix() or base.node_type:isvector() then
    return SqIdxVecMat(self, base, ctxt)
  elseif base.node_type:isscalarkey() then
    return SqIdxKey(self, base, ctxt)
  else
    ctxt:error(base, 'type '..tostring(base.node_type)..
                ' does not support indexing with square brackets []')
    local errnode = ast.SquareIndex:DeriveFrom(self)
    errnode.node_type = errorT
    return errnode
  end
end

function ast.Call:check(ctxt)
  local call = self:clone()
  
  call.node_type = errorT -- default
  local func     = self.func:check(ctxt)
  call.params    = {}

  for i,p in ipairs(self.params) do
    call.params[i] = p:check(ctxt)
  end

  local v = func.node_type:isinternal() and func.node_type.value
  if v and B.is_builtin(v) then
    -- check the built-in.  If an ast is returned,
    -- then we assume the built-in is functioning as an internal macro
    -- Otherwise, assume standard built-in behavior
    local check_result = v.check(call, ctxt)
    if ast.is_ast(check_result) then
      return check_result
    else
      call.func      = v
      call.node_type = check_result
    end
  elseif v and is_macro(v) then
    -- replace the call node with the inlined AST
    call = RunMacro(ctxt, self, v, call.params)
  elseif v and F.is_function(v) then
    call = InlineUserFunc(ctxt, self, v, call.params)
  elseif v and T.istype(v) and v:isvalue() then
    local params = call.params
    if #params ~= 1 then
      ctxt:error(self, "Cast to " .. tostring(v) ..
              " expects exactly 1 argument (instead got " .. #params ..
              ")")
    else
      -- TODO: We should have more aggresive casting protection.
      -- i.e. we should allow anything reasonable in Terra/C but
      -- no more.
      local pretype = params[1].node_type
      local casttype = v
      if not matching_type_dims(pretype, casttype) then
        ctxt:error(self, "Cannot cast between primitives, vectors, "..
                         "matrices of different dimensions")
      else
        call = ast.Cast:DeriveFrom(self)
        call.value      = params[1]
        call.node_type  = v
      end
    end
  -- __apply_macro  i.e.  c(1,0)  for offsetting in a grid
  elseif func.node_type:isscalarkey() then
    local apply_macro = func.node_type.relation.__apply_macro
    if is_macro(apply_macro) then
      local params = {func}
      for _,v in ipairs(call.params) do table.insert(params, v) end
      call = RunMacro(ctxt, self, apply_macro, params)
    else
      ctxt:error(self, "Relation "..tostring(func.node_type.relation)..
                       " does not have an __apply_macro"..
                       " macro defined; cannot call key.")
    end
  elseif func.node_type:iserror() then
    -- fall through
    -- (do not print error messages for errors already reported)
  else
    ctxt:error(self, "This call was neither a function nor macro.")
  end

  return call
end

function ast.Global:check(ctxt)
  local n     = self:clone()
  n.node_type = self.global._type
  return n
end

function ast.Quote:check(ctxt)
  -- Ensure quotes are only typed once
  -- By typing the quote at declaration, we make it safe
  -- to included it in other code as is.

  -- We also strip out the quote wrapper here since it's no longer needed

  if self.node_type then
    return self.code
  else
    return self.code:check(ctxt)
  end
end

function ast.LetExpr:check(ctxt)
  local let = self:clone()

  ctxt:enterblock()
  let.block = self.block:check(ctxt)
  let.exp   = self.exp:check(ctxt)
  ctxt:leaveblock()

  let.node_type = let.exp.node_type
  return let
end


function ast.FieldAccess:check(ctxt)
  local fa = self:clone()
  fa.key = self.key:check(ctxt)
  return fa
end

function ast.LuaObject:check(ctxt)
  assert(self.node_type and self.node_type:isinternal())
  return self
end
function ast.Where:check(ctxt)
  --note: where is generated in an internal macro,
  --      so its fields are already type-checked
  local fieldobj = self.field.node_type
  local keytype  = self.key.node_type
  if not fieldobj:isinternal() or not R.is_field(fieldobj.value) then
    ctxt:error(self,"Expected a field as the first argument but found "..
                    tostring(fieldobj))
  end
  local field = fieldobj.value
  if keytype ~= field:Type() then
    ctxt:error(self,"Key of where is type "..tostring(keytype)..
                    " but expected type "..tostring(field:Type()))
  end
  local rel = field:Relation()
  if not rel:isGrouped() or rel:GroupedKeyField() ~= field then
    ctxt:error(self,"Relation '"..rel:Name().."' is not "..
                    "grouped by Field '"..field:Name().."'")
  end
  local w     = self:clone()
  w.relation  = rel
  w.field     = self.field -- for safety/completeness
  w.key       = self.key
  w.node_type = queryT(w.relation,{})
  return w
end

------------------------------------------------------------------------------
--[[ Semantic checking called from here:                                  ]]--
------------------------------------------------------------------------------

-- only makes sense to type-check as a top-level execution right now...
function ast.UserFunction:check(ctxt)
  local ufunc                 = self:clone()
  local keyparam              = self.params[1]
  local keytype               = self.ptypes[1]
  if not keytype:isscalarkey() then
    ctxt:error(self, 'First argument to function must have key type')
    return self
  end
  -- record the first parameter in the context
  ctxt:recordcenter(keyparam)
  ctxt:ebb()[keyparam]      = keytype

  for i=2,#self.params do
    local pname = self.params[i]
    local ptype = self.ptypes[i]
    if not ptype:isinternal() or type(ptype.value) ~= 'string' then
      ctxt:error(self, 'Expected secondary arguments to be strings')
      ptype = errorT
    end
    ctxt:ebb()[pname]     = ptype
  end

  -- double-check that there's no return value; redundant
  if self.exp then
    ctxt:error(self, 'A mapped function may not return a value')
    return self
  end

  -- discard the string arguments, because type-checking
  -- will substitute them into the AST directly
  ufunc.params                = { keyparam }
  ufunc.ptypes                = { keytype }
  
  ufunc.relation              = keytype.relation
  ufunc.body                  = self.body:check(ctxt)

  return ufunc
end

function S.check(some_ast)
  -- environment for checking variables and scopes
  local env  = terralib.newenvironment(nil)
  local diag = terralib.newdiagnostics()
  local ctxt = Context.new(env, diag)

  --------------------------------------------------------------------------

  diag:begin()
  env:enterblock()
  local typed_ast  = some_ast:check(ctxt)
  env:leaveblock()
  diag:finishandabortiferrors("Errors during typechecking ebb", 1)
  return typed_ast
end

function S.check_quote(quote_ast)
  local checked = S.check(quote_ast)

  -- now re-wrap in a quote
  if checked:is(ast.Quote) then
    return checked
  else
    local q       = ast.Quote:DeriveFrom(checked)
    q.code        = checked
    if checked.is_centered then q.is_centered = true end
    q.node_type   = checked.node_type -- halt type-checking here
    return q
  end
end
