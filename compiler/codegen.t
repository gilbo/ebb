local C = {}
package.loaded["compiler.codegen"] = C

local ast = require "compiler.ast"

----------------------------------------------------------------------------

local Context = {}
Context.__index = Context

function Context.new(env, runtime)
    local ctxt = setmetatable({
        env     = env,
        runtime = runtime
    }, Context)
    return ctxt
end
function Context:localenv()
  return self.env:localenv()
end
function Context:enterblock()
  self.env:enterblock()
end
function Context:leaveblock()
  self.env:leaveblock()
end
function Context:runtime_codegen_kernel_body (kernel_node, relation)
  return self.runtime:codegen_kernel_body(self, kernel_node, relation)
end
function Context:runtime_codegen_field_write (fw_node)
  return self.runtime:codegen_field_write(self, fw_node)
end
function Context:runtime_codegen_field_read (fa_node)
  return self.runtime:codegen_field_read(self, fa_node)
end


function C.codegen (runtime, luaenv, kernel_ast, relation)
  local env = terralib.newenvironment(luaenv)
  local ctxt = Context.new(env, runtime)

  ctxt:enterblock()
    local parameter = symbol(L.row(relation):terraType())
    ctxt:localenv()[kernel_ast.name] = parameter
    local kernel_body =
      ctxt:runtime_codegen_kernel_body(kernel_ast, relation)
  ctxt:leaveblock()

  local r = terra ()
    [kernel_body]
  end
  return r
end

----------------------------------------------------------------------------

function ast.AST:codegen (ctxt)
  print(debug.traceback())
  error("Codegen not implemented for AST node " .. self.kind)
end

function ast.ExprStatement:codegen (ctxt)
  return self.exp:codegen(ctxt)
end

function ast.QuoteExpr:codegen (ctxt)
  if self.block then
    assert(self.exp)
    ctxt:enterblock()
    local block = self.block:codegen(ctxt)
    local exp   = self.exp:codegen(ctxt)
    ctxt:leaveblock()

    return quote [block] in [exp] end
  else
    assert(self.exp)
    return self.exp:codegen(ctxt)
  end
end

-- DON'T CODEGEN THE KERNEL THIS WAY; HANDLE IN C.codegen()
--function ast.LisztKernel:codegen (ctxt)
--end

function ast.Block:codegen (ctxt)
  -- start with an empty ast node, or we'll get an error when appending new quotes below
  local code = quote end
  for i = 1, #self.statements do
    local stmt = self.statements[i]:codegen(ctxt)
    code = quote code stmt end
  end
  return code
end

function ast.CondBlock:codegen(ctxt, cond_blocks, else_block, index)
  index = index or 1

  local cond  = self.cond:codegen(ctxt)
  ctxt:enterblock()
  local body = self.body:codegen(ctxt)
  ctxt:leaveblock()

  if index == #cond_blocks then
    if else_block then
      return quote if [cond] then [body] else [else_block:codegen(ctxt)] end end
    else
      return quote if [cond] then [body] end end
    end
  else
    ctxt:enterblock()
    local nested = cond_blocks[index + 1]:codegen(ctxt, cond_blocks, else_block, index + 1)
    ctxt:leaveblock()
    return quote if [cond] then [body] else [nested] end end
  end
end

function ast.IfStatement:codegen (ctxt)
  return self.if_blocks[1]:codegen(ctxt, self.if_blocks, self.else_block)
end

function ast.WhileStatement:codegen (ctxt)
  local cond = self.cond:codegen(ctxt)
  ctxt:enterblock()
  local body = self.body:codegen(ctxt)
  ctxt:leaveblock()
  return quote while [cond] do [body] end end
end

function ast.DoStatement:codegen (ctxt)
  ctxt:enterblock()
  local body = self.body:codegen(ctxt)
  ctxt:leaveblock()
  return quote do [body] end end
end

function ast.RepeatStatement:codegen (ctxt)
  ctxt:enterblock()
  local body = self.body:codegen(ctxt)
  local cond = self.cond:codegen(ctxt)
  ctxt:leaveblock()

  return quote repeat [body] until [cond] end
end

function ast.NumericFor:codegen (ctxt)
  -- min and max expression should be evaluated in current scope,
  -- iter expression should be in a nested scope, and for block
  -- should be nested again -- that way the loop var is reset every
  -- time the loop runs.
  local minexp  = self.lower:codegen(ctxt)
  local maxexp  = self.upper:codegen(ctxt)
  local stepexp = self.step and self.step:codegen(ctxt) or nil

  ctxt:enterblock()
  local iterstr = self.name
  local itersym = symbol()
  ctxt:localenv()[iterstr] = itersym

  ctxt:enterblock()
  local body = self.body:codegen(ctxt)
  ctxt:leaveblock()
  ctxt:leaveblock()

  if stepexp then
    return quote for [itersym] = [minexp], [maxexp], [stepexp] do [body] end end
  end

  return quote for [itersym] = [minexp], [maxexp] do [body] end end
end

function ast.Break:codegen(ctxt)
  return quote break end
end

function ast.Name:codegen(ctxt)
  local s = ctxt:localenv()[self.name]
  assert(terralib.issymbol(s))
  return `[s]
end

local function bin_exp (op, lhe, rhe)
  if     op == '+'   then return `[lhe] +   [rhe]
  elseif op == '-'   then return `[lhe] -   [rhe]
  elseif op == '/'   then return `[lhe] /   [rhe]
  elseif op == '*'   then return `[lhe] *   [rhe]
  elseif op == '%'   then return `[lhe] %   [rhe]
  elseif op == '^'   then return `[lhe] ^   [rhe]
  elseif op == 'or'  then return `[lhe] or  [rhe]
  elseif op == 'and' then return `[lhe] and [rhe]
  elseif op == '<'   then return `[lhe] <   [rhe]
  elseif op == '>'   then return `[lhe] >   [rhe]
  elseif op == '<='  then return `[lhe] <=  [rhe]
  elseif op == '>='  then return `[lhe] >=  [rhe]
  elseif op == '=='  then return `[lhe] ==  [rhe]
  elseif op == '~='  then return `[lhe] ~=  [rhe]
  end
end

function let_vec_binding(typ, N, exp)
  local val = symbol(typ:terraType())
  local let_binding = quote var [val] = [exp] end

  local coords = {}
  if typ:isVector() then
    for i=1, N do coords[i] = `val._0[i-1] end
  else
    for i=1, N do coords[i] = `val end
  end

  return let_binding, coords
end

function vec_bin_exp(op, result_typ, lhe, rhe, lhtyp, rhtyp)
  if not result_typ:isVector() then return bin_exp(op, lhe, rhe) end

  local N = result_typ.N
  local res_typ = result_typ:terraBaseType()

  -- otherwise, bind the expression results into a let-clause
  -- to ensure each sub-expression is evaluated exactly once.
  -- Then, construct a vector expression result using these bound
  -- values.
  local lhbind, lhcoords = let_vec_binding(lhtyp, N, lhe)
  local rhbind, rhcoords = let_vec_binding(rhtyp, N, rhe)

  -- Now, assemble the vector by mashing up the coords
  local result_coords = {}
  for i=1, N do
    local exp = bin_exp(op, lhcoords[i], rhcoords[i])
    result_coords[i] = `[res_typ](exp)
  end

  local q = quote
    [lhbind]
    [rhbind]
  in
    { arrayof(res_typ, [result_coords]) }
  end

  return q
end

--[[ Factored logic for generating binary expressions in BinaryOp, Assignments
   that contain reductions and FieldWrites with reductions.
    Args:
    - op  : lua string encoding operator (e.g. '+', etc.)
    - lhs : terra quote representing lhs expression
    - rhs : terra quote representing rhs expression
    - typ : liszt type of binop result 
--]]
--local function codegen_binary_op (op, lhs, rhs, typ)
--  if not typ:isVector() then return bin_exp(op, lhs, rhs) end
--
--  if lhs
--
--  local entries = {}
--  for i = 1, typ.N do
--    entries[i] = bin_exp(op, `lhs._0[i-1], `rhs._0[i-1])
--  end
--
--  local q = `{ [entries] }
--  return q

  --local result = symbol(typ:terraType())
  --local bt  = typ:terraBaseType()
  --local stmts = { quote var[res] }
  --local q = quote var [res] end

  --for i = 1, typ.N do
  --  local exp = bin_exp(op, `lhs._0[i-1], `rhs._0[i-1])
  --  q = quote [q] res._0[i-1] = exp end
  --end
--
  --return quote [q] in [res] end
--end

--[[ Factored logic for generating assignments in Assignments, DeclStatements,
   and FieldWrites:
      Args:
    - lval      : terra quote representing the lvalue
    - lval_type : liszt type of lval
    - rval      : terra quote representing the rhs expression
    - rval_type : liszt type of rval
--]]
--local function codegen_assignment (lval, lval_type, rval, rval_type)
--  if not lval_type:isVector() then
--    return quote [lval] = rval end
--  end
--
--  local lbt = lval_type:terraBaseType()
--  local rtt = rval_type:terraType()
--  local rbt = rval_type:terraBaseType()
--  local len = lval_type.N
--  return quote
--    var r : rtt = rval
--    var vt = [&lbt](&lval)
--    var rt = [&rbt](&r)
--    for i = 0, len do
--      vt[i] = rt[i]
--    end
--  end
--end

function ast.Assignment:codegen (ctxt)
  local lhs   = self.lvalue:codegen(ctxt)
  local rhs   = self.exp:codegen(ctxt)

  local ltype, rtype = self.lvalue.node_type, self.exp.node_type

  if self.reduceop then
    rhs = vec_bin_exp(self.reduceop, ltype, lhs, rhs, ltype, rtype)
    --rhs = codegen_binary_op(self.reduceop, lhs, rhs, ltype)
    --return codegen_assignment(lhs, ltype, rhs, ltype)
  end
  return quote [lhs] = rhs end
  --return codegen_assignment(lhs, ltype, rhs, rtype)
end

function ast.FieldWrite:codegen (ctxt)
  return ctxt:runtime_codegen_field_write(self)
end

function ast.FieldAccess:codegen (ctxt)
  return ctxt:runtime_codegen_field_read(self)
end

function ast.Cast:codegen(ctxt)
  local typ = self.node_type
  local valuecode = self.value:codegen(ctxt)

  if not typ:isVector() then
    return `[typ:terraType()](valuecode)
  else
    local vec = symbol(self.value.node_type:terraType())
    local bt  = typ:terraBaseType()

    local coords = {}
    for i= 1, typ.N do coords[i] = `[bt](vec._0[i-1]) end

    return quote
      var [vec] = valuecode
    in
      { arrayof(bt, [coords]) }
    end
  end
end

-- By the time we make it to codegen, Call nodes are only used to represent builtin function calls.
function ast.Call:codegen (ctxt)
    return self.func.codegen(self, ctxt)
end


function ast.DeclStatement:codegen (ctxt)
  local varname = self.name
  local tp      = self.node_type:terraType()
  local varsym  = symbol(tp)
  ctxt:localenv()[varname] = varsym

  if self.initializer then
    local exp = self.initializer:codegen(ctxt)
    return quote 
      var [varsym] = [exp]
      --var [varsym]
      --[codegen_assignment(varsym, self.node_type, exp, self.initializer.node_type)]
    end
  else
    return quote var [varsym] end
  end
end

function ast.VectorLiteral:codegen (ctxt)
  local typ = self.node_type
  local btt = typ:terraBaseType()

  -- type everything explicitly
  local elems = {}
  for i = 1, #self.elems do
    local code = self.elems[i]:codegen(ctxt)
    elems[i] = `[btt](code)
  end

  -- we allocate vectors as a struct with a single array member
  return `{ arrayof(btt, [elems]) }
end

function ast.Scalar:codegen (ctxt)
  local d = self.scalar.data
  local s = symbol(&self.scalar.type:terraType())
  return `@d
end

function ast.VectorIndex:codegen (ctxt)
  local vector = self.vector:codegen(ctxt)
  local index  = self.index:codegen(ctxt)

  return `vector._0[index]
end

function ast.Number:codegen (ctxt)
  return `[self.value]
end

function ast.Bool:codegen (ctxt)
  if self.value == 'true' then
    return `true
  else 
    return `false
  end
end

function ast.UnaryOp:codegen (ctxt)
  local expr = self.exp:codegen(ctxt)
  local typ  = self.node_type

  if not typ:isVector() then
    if (self.op == '-') then return `-[expr]
    else                     return `not [expr]
    end
  else -- Unary op applied to a vector...
    local binding, coords = let_vec_binding(typ, typ.N, expr)

    -- apply the operation
    if (self.op == '-') then
      for i = 1, typ.N do coords[i] = `-[ coords[i] ] end
    else
      for i = 1, typ.N do coords[i] = `not [ coords[i] ] end
    end

    return quote
      [binding]
    in
      { [coords] }
    end
  end
end

function ast.BinaryOp:codegen (ctxt)
  local lhe = self.lhs:codegen(ctxt)
  local rhe = self.rhs:codegen(ctxt)

  -- handle case of two primitives
  return vec_bin_exp(self.op, self.node_type,
      lhe, rhe, self.lhs.node_type, self.rhs.node_type)


--  if not typ:isVector() then return bin_exp(self.op, lhe, rhe) end
--
--  -- otherwise, bind the expression results into a let-clause
--  -- to ensure each sub-expression is evaluated exactly once.
--  -- Then, construct a vector expression result using these bound
--  -- values.
--  local lhbind, lhcoords = let_vec_binding(self.lhs.node_type, lhe)
--  local rhbind, rhcoords = let_vec_binding(self.rhs.node_type, rhe)
--
--  -- Now, assemble the vector by mashing up the coords
--  local result_coords = {}
--  for i=1, typ.N do
--    result_coords[i] = bin_exp(self.op, lhcoords[i], rhcoords[i])
--  end
--
--  local q = quote
--    [lhbind]
--    [rhbind]
--  in
--    { [result_coords] }
--  end
--
--  return q



  -- primitive types
--  if not self.node_type:isVector() then
--    return bin_exp(self.op, lhe, rhe)
--
--  -- vectors are stored as arrays and must be operated on
--  -- in loops
--  else
--    local s = symbol(self.node_type:terraType()) -- result
--    return quote 
--      var [s]
--      -- temp vars:
--      var l = [&self.lhs.node_type:terraBaseType()](&lhe)
--      var r = [&self.rhs.node_type:terraBaseType()](&rhe)
--      var t = [&self.node_type:terraBaseType()](&s)
--
--      for i = 0, [self.node_type.N] do
--        @t = [bin_exp(self.op, `@l, `@r)]
--        t = t + 1
--        l = l + 1
--        r = r + 1
--      end
--    in
--      [s]
--    end
--  end
end

function ast.LuaObject:codegen (ctxt)
    return `{}
end
function ast.Where:codegen(ctxt)
    local key   = self.key:codegen(ctxt)
    local sType = self.node_type:terraType()
    local index = self.relation._indexdata
    local v = quote
        var k   = [key]
        var idx = [index]
    in 
        sType { idx[k], idx[k+1] }
    end
    return v
end

local function doProjection(obj,field)
    assert(L.is_field(field))
    return `field.data[obj]
end

function ast.GenericFor:codegen (ctxt)
    local set       = self.set:codegen(ctxt)
    local iter      = symbol("iter")
    local rel       = self.set.node_type.relation
    local projected = iter

    for i,p in ipairs(self.set.node_type.projections) do
        local field = rel[p]
        projected   = doProjection(projected,field)
        rel         = field.type.relation
        assert(rel)
    end
    local sym = symbol(L.row(rel):terraType())
    ctxt:enterblock()
        ctxt:localenv()[self.name] = sym
        local body = self.body:codegen(ctxt)
    ctxt:leaveblock()
    local code = quote
        var s = [set]
        for [iter] = s.start,s.finish do
            var [sym] = [projected]
            [body]
        end
    end
    return code
end

--C.utils = {
--  codegen_assignment = codegen_assignment,
--  codegen_binary_op  = codegen_binary_op
--}