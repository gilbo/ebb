local S = {}
package.loaded["compiler.semant"] = S

local ast = require "compiler.ast"
local B   = require "compiler.builtins"
local T   = require "compiler.types"
local L   = require 'compiler.lisztlib'
--[[
    AST:check(ctxt) type checking routines
        These methods define type checking.
        Each check() method is designed to accept a typing context (see below).
        Once called, check() will construct and return a new type-checked
        version of its subtree (new AST nodes)
--]]

------------------------------------------------------------------------------
--[[ Context Definition                                                   ]]--
------------------------------------------------------------------------------
--[[ A Context is passed through type-checking, keeping track of any kind of
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
function Context:liszt()
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

local function try_coerce(target_typ, node, ctxt)
    if node.node_type:isCoercableTo(target_typ) then
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
    if condtype ~= L.error and condtype ~= L.bool then
        ctxt:error(self, "expected bool expression but found " ..
                         condtype:toString())
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

    if condtype ~= L.error and condtype ~= L.bool then
        ctxt:error(self, "expected bool expression but found " ..
                         condtype:toString())
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
    [L.float] = {
        ['+']   = true,
        --['-']   = true,
        ['*']   = true,
        --['/']   = true,
        ['min'] = true,
        ['max'] = true,
    },
    [L.double] = {
        ['+']   = true,
        --['-']   = true,
        ['*']   = true,
        --['/']   = true,
        ['min'] = true,
        ['max'] = true,
    },
    [L.int] = {
        ['min'] = true,
        ['max'] = true,
        ['+']   = true,
        --['-']   = true,
        ['*']   = true,
        --['/']   = true,
    },
    [L.uint64] = {
        ['min'] = true,
        ['max'] = true,
        ['+']   = true,
        --['-']   = true,
        ['*']   = true,
        --['/']   = true,
    },
    [L.bool] = {
        ['and'] = true,
        ['or']  = true,
    },
}

function check_reduce(node, ctxt)
    local op        = node.reduceop
    local lvalue    = node.lvalue
    local ltype     = lvalue.node_type

    -- Centered reductions don't need to be atomic!
    if (node.lvalue:is(ast.FieldAccess) or node.lvalue:is(ast.FieldAccessIndex)) and
        node.lvalue.key.is_centered then return end

    if op == nil then return end

    local reductions = reductions_by_type[ltype:baseType()]
    if not reductions or not reductions[op] then
        ctxt:error(node, 'Reduce operator "'..op..'" for type '..
        '"'..tostring(ltype)..'" is not currently supported.')
    end
end

function ast.Assignment:check(ctxt)
    local node = self:clone()

    -- LHS tracked for phase-checking
    node.lvalue = self.lvalue:check(ctxt)

    local ltype  = node.lvalue.node_type
    if ltype == L.error then return node end

    node.exp     = self.exp:check(ctxt)
    local rtype  = node.exp.node_type
    if rtype == L.error then return node end

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
    elseif node.lvalue.node_type:isKey() and
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
    elseif node.lvalue:is(ast.FieldAccess) or node.lvalue:is(ast.FieldAccessIndex) then
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
            decl.initializer =
                try_coerce(decl.node_type, decl.initializer, ctxt)
        else
            decl.node_type = exptyp
        end
        -- if the rhs is a centered key, try to propagate that information
        -- NOTE: this pseudo-constant propagation is sound b/c
        -- we don't allow re-assignment of key-type variables
        if exptyp:isScalarKey() and decl.initializer.is_centered then
            ctxt:recordcenter(decl.name)
        end
    end

    if decl.node_type ~= L.error and not decl.node_type:isFieldType() then
        ctxt:error(self,"can only assign numbers, bools, "..
                        "or keys to local temporaries")
    end

    ctxt:liszt()[decl.name] = decl.node_type

    return decl
end

function ast.NumericFor:check(ctxt)
    local node = self
    local function check_num_type(tp)
        if tp ~= L.error and not tp:isIntegral() then
            ctxt:error(node,
                "expected an integer-type expression to define the "..
                             "iterator bounds/step (found "..
                             tp:toString()..")")
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
    ctxt:liszt()[numfor.name] = numfor.node_type
    numfor.body = self.body:check(ctxt)
    ctxt:leaveloop()
    ctxt:leaveblock()
    return numfor
end

function ast.GenericFor:check(ctxt)
    local r = self:clone()
    r.name  = self.name
    r.set   = self.set:check(ctxt)
    if not r.set.node_type:isQuery() then
        ctxt:error(self,"for statement expects a query but found type ",r.set.node_type)
        return r
    end
    local rel = r.set.node_type.relation
    for i,p in ipairs(r.set.node_type.projections) do
        if not rel[p] then
            ctxt:error(self,"Could not find field '"..p.."'")
            return r
        end
        rel = rel[p].type.relation
        assert(rel)
    end
    local keyType = L.key(rel)
    ctxt:enterblock()
    ctxt:enterloop()
    ctxt:liszt()[r.name] = keyType
    r.body = self.body:check(ctxt)
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
    local rel           = reltyp:isInternal() and reltyp.value
    if not rel or not L.is_relation(rel) then
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
        if not exp or exp.node_type == L.error then return insert end

        record.exprs[i]     = exp
        rectyp_proto[name]  = exp.node_type
        insert.fieldindex[field] = i
    end

    -- save record type
    record.node_type    = L.record(rectyp_proto)
    insert.record_type  = record.node_type
    -- type compatibility with the relation will be checked
    -- in the phase pass

    return insert
end

function ast.DeleteStatement:check(ctxt)
    local delete = self:clone()

    delete.key   = self.key:check(ctxt)
    local keytyp = delete.key.node_type

    if not keytyp:isScalarKey() or not delete.key.is_centered then
        ctxt:error(self,"Only centered keys may be deleted")
        return delete
    end

    return delete
end

function ast.CondBlock:check(ctxt)
    local new_node  = self:clone()
    new_node.cond   = self.cond:check(ctxt)
    local condtype  = new_node.cond.node_type
    if condtype ~= L.error and condtype ~= L.bool then
        ctxt:error(self, "conditional expression type should be "..
                         "boolean (found " .. condtype:toString() .. ")")
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
  if t1:isScalar() and t2:isScalar() then return true end
  if t1:isVector() and t2:isVector() then
    return t1.N == t2.N
  elseif t1:isMatrix() and t2:isMatrix() then
    return t1.Nrow == t2.Nrow and t1.Ncol == t2.Ncol
  end
  return false
end

local function err (node, ctx, msg)
    node.node_type = L.error
    if msg then ctx:error(node, msg) end
    return node
end


-- NOTE THIS IS UNSAFE.  Caller must check
-- whether or not the coercion is type-safe
local function coerce_base(btyp, node)
  local ntyp = node.node_type
  if btyp == ntyp:baseType() then return node end

  local cast = ast.Cast:DeriveFrom(node)
  if ntyp:isScalar()   then  cast.node_type = btyp end
  if ntyp:isVector()      then  cast.node_type = L.vector(btyp, ntyp.N) end
  if ntyp:isMatrix() then
    cast.node_type = L.matrix(btyp, ntyp.Nrow, ntyp.Ncol)
  end
  cast.value = node
  return cast
end
-- Will coerce base type but not the underlying 
local function try_bin_coerce(binop, errf)
  local join = T.type_join(binop.lhs.node_type:baseType(),
                           binop.rhs.node_type:baseType())
  if join == L.error then return errf() end

  binop.lhs = coerce_base(join, binop.lhs)
  binop.rhs = coerce_base(join, binop.rhs)
  binop.node_type = T.type_join(binop.lhs.node_type, binop.rhs.node_type)

  return binop
end
local function try_bin_coerce_bool(binop, errf)
  local node = try_bin_coerce(binop, errf)
  if node.node_type == L.error then return node end
  node.node_type = L.bool
  return node
end
local function try_mat_prod_coerce(binop, errf, N, M)
  local join = T.type_join(binop.lhs.node_type:baseType(),
                           binop.rhs.node_type:baseType())
  if join == L.error then return errf() end
  binop.lhs = coerce_base(join, binop.lhs)
  binop.rhs = coerce_base(join, binop.rhs)
  if M == nil then binop.node_type = L.vector(join, N)
              else binop.node_type = L.matrix(join, N, M) end
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
  if lt == L.error or rt == L.error then return err(self, ctxt) end

  -- error messages
  local function type_err()
    return err(binop, ctxt, 'incompatible types: ' .. lt:toString() ..
                            ' and ' .. rt:toString())
  end
  local function op_err()
    return err(binop, ctxt, 'invalid types for operator \'' .. binop.op ..
                '\': ' .. lt:toString() .. ' and ' .. rt:toString())
  end

  -- special case for key types
  if lt:isKey() and rt:isKey() and (op == '==' or op == '~=') then
    if lt ~= rt then return type_err() end
    binop.node_type = L.bool
    return binop
  end

  -- prohibit min/max outside of reductions
  if op == 'min' or op == 'max' then
    return err(binop, ctxt, op .. ' is unsupported as a binary operator')
  end

  -- Now make sure we have value types
  if not lt:isValueType() or not rt:isValueType() then
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
    if not lt:isLogical() or not rt:isLogical() then return op_err() end
    if not matching_type_dims(lt,rt) then return type_err() end
    binop.node_type = lt
    return binop
  end

  -- OP: Ord
    -- DIM L/R: scalar
      -- BASETYPE L/R: Numeric
        -- COERCE, return BOOL
  if is_ord_op[op] then
    if not (lt:isNumeric() and lt:isPrimitive() and
            rt:isNumeric() and rt:isPrimitive()) then return op_err() end
    return try_bin_coerce_bool(binop, type_err)
  end

  -- OP: Mod
    -- DIM L/R: scalar
      -- BASETYPE L/R: Integral
        -- COERCE, return coercion
  if op == '%' then
    if not (lt:isIntegral() and lt:isPrimitive() and
            rt:isIntegral() and rt:isPrimitive()) then return op_err() end
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
    if not lt:isNumeric() or not rt:isNumeric() then return op_err() end
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
    if not lt:isNumeric() or not rt:isNumeric() then return op_err() end
    if lt:isPrimitive() or rt:isPrimitive() then
      return try_bin_coerce(binop, type_err)

--    elseif lt:isVector() and rt:isMatrix() and lt.N == rt.Nrow then
--      return try_mat_prod_coerce(binop, type_err, rt.Ncol, nil)
--
--    elseif lt:isMatrix() and rt:isVector() and lt.Ncol == rt.N then
--      return try_mat_prod_coerce(binop, type_err, lt.Nrow, nil)
--
--    elseif lt:isMatrix() and rt:isMatrix() and lt.Ncol == rt.Nrow
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
    if not lt:isNumeric() or not rt:isNumeric() then return op_err() end
    if not rt:isPrimitive()                     then return op_err() end

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
    unop.node_type = L.error -- default

    if unop.op == 'not' then
        if exptype:isLogical() then
            unop.node_type = exptype
        else
            ctxt:error(self, "unary \"not\" expects a boolean operand")
        end
    elseif unop.op == '-' then
        if exptype:isNumeric() then
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
    lo.node_type = L.internal(obj)
    return lo
end

------------------------------------------------------------------------------
--[[                               Variables                              ]]--
------------------------------------------------------------------------------

function ast.Name:check(ctxt)
    -- try to find the name in the local Liszt scope
    local typ = ctxt:liszt()[self.name]
    -- if the name is in the local scope, then it must have been declared
    -- somewhere in the liszt kernel.  Thus, it has to be a primitive, a
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
    err_node.node_type = L.error
    return err_node
end

function ast.Number:check(ctxt)
    local number = self:clone()
    number.value = self.value
    if self.node_type then
        number.node_type = self.node_type
    elseif self.valuetype then
        if     self.valuetype == int then
            number.node_type = L.int
        elseif self.valuetype == double then
            number.node_type = L.double
        elseif self.valuetype == float then
            number.node_type = L.float
        elseif self.valuetype == uint64 then
            number.node_type = L.uint64
        else
            ctxt:error(self, "numeric literal type unsupported by Liszt")
            number.node_type = L.error
        end
    elseif tonumber(self.value) % 1 == 0 then
        number.node_type = L.int
    else
        number.node_type = L.double
    end
    return number
end

function ast.Bool:check(ctxt)
    local boolnode     = self:clone()
    boolnode.node_type = L.bool
    return boolnode
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
    local max_type = vecs[1].node_type:baseType()

    -- take max type
    for i=2,matlit.n do
        local tp = vecs[i].node_type
        if not tp:isVector() then return err(literal, ctxt, tp_error) end
        if tp.N ~= matlit.m  then return err(literal, ctxt, dim_error) end

        if max_type:isCoercableTo(tp:baseType()) then
            max_type = tp:baseType()
        elseif not tp:baseType():isCoercableTo(max_type) then
            return err(literal, ctxt, mt_error)
        end
    end

    -- if the type was explicitly provided...
    if literal.node_type then
        max_type = literal.node_type:baseType()
    end

    -- coerce and re-marshall the entries
    for i = 1, matlit.n do
        for j = 1, matlit.m do
            local idx = (i-1)*matlit.m + (j-1) + 1
            matlit.elems[idx] = try_coerce(max_type, vecs[i].elems[j], ctxt)
        end
    end

    matlit.node_type = L.matrix(max_type, matlit.n, matlit.m)
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
        if tp == L.error then return err(self, ctxt) end
    end

    -- check if matrix applies...
    local max_type  = veclit.elems[1].node_type
    if max_type:isVector() then
        if self.node_type then veclit.node_type = self.node_type end
        -- SPECIAL CASE: a vector of vectors is a matrix and
        --               needs to be treated specially
        return convert_to_matrix_literal(veclit, ctxt)
    elseif not max_type:isScalar() then
        return err(self, ctxt, tp_error)
    end

    -- compute a max type
    for i = 2, #self.elems do
        local tp = veclit.elems[i].node_type
        if not tp:isScalar() then
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
        max_type = self.node_type:baseType()
    end

    -- now coerce all of the entries into the max type
    for i = 1, #veclit.elems do
        veclit.elems[i] = try_coerce(max_type, veclit.elems[i], ctxt)
    end

    veclit.node_type = L.vector(max_type, #veclit.elems)
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
--        if exp.node_type == L.error then found_error = true end
--    end
--
--    if found_error then
--        record.node_type = L.error
--    else
--        record.node_type = L.record(rectyp_proto)
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
        for i=1,#(self.params) do
            func.params[i] = self.params[i]:UniqueCopy()
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

    function ast.LisztKernel:alpha_rename(remap)
        error('should never try to alpha rename a whole kernel')
        -- even though a kernel does define names
    end

    function ast.Name:alpha_rename(remap)
        local n = self:clone()
        n.name = remap[self.name] -- remapped symbol
        return n
    end

------------------------------------------------------------------------------
--[[                         Miscellaneous nodes:                         ]]--
------------------------------------------------------------------------------


local function QuoteParams(all_params_asts)
    local quoted_params = {}
    for i,param_ast in ipairs(all_params_asts) do
        local q = ast.Quote:DeriveFrom(param_ast)
        q.code = param_ast
        if param_ast.is_centered then q.is_centered = true end
        q.node_type = param_ast.node_type -- halt type-checking here
        quoted_params[i] = q
    end
    return quoted_params
end

local function RunMacro(ctxt,src_node,the_macro,params)
    --local quoted_params = QuoteParams(params)
    local param_syms   = {}
    local declarations = {}
    for i, p_ast in ipairs(params) do
        local decl       = ast.DeclStatement:DeriveFrom(src_node)
        decl.name        = ast.GenSymbol('_macro_arg_'..tostring(i))
        decl.initializer = p_ast
        decl.node_type   = p_ast.node_type

        local n     = ast.Name:DeriveFrom(p_ast)
        n.name      = decl.name
        n.node_type = p_ast.node_type

        local q = ast.Quote:DeriveFrom(p_ast)
        q.code = n
        q.node_type = n.node_type

        if p_ast.is_centered then
            n.is_centered = true
            q.is_centered = true
        end

        declarations[i] = decl
        param_syms[i]   = q
    end

    local result = the_macro.genfunc(unpack(param_syms))

    if ast.is_ast(result) and result:is(ast.Quote) then
        local block = ast.Block:DeriveFrom(src_node)
        block.statements = declarations

        local expansion     = ast.LetExpr:DeriveFrom(src_node)
        expansion.block     = block
        expansion.exp       = result
        expansion.node_type = result.node_type
        return expansion
        --local qexp          = ast.Quote:DeriveFrom(src_node)
        --return expansion:check(ctxt)
    else
        ctxt:error(src_node, 'Macros must return quoted code')
        local errnode     = src_node:clone()
        errnode.node_type = L.error
        return errnode
    end
end

local function InlineUserFunc(ctxt, src_node, the_func, param_asts)
    -- note that this alpha rename is safe because the
    -- param_asts haven't been attached yet.  We know from
    -- specialization that there are no free variables in the function.
    local f = the_func.ast:alpha_rename()

    -- check that the correct number of arguments are provided
    if #(f.params) ~= #param_asts then
        ctxt:error(src_node,
            'Expected '..tostring(#(f.params))..' arguments, '..
            'but was supplied '..tostring(#param_asts)..' arguments')
        local errnode     = src_node:clone()
        errnode.node_type = L.error
        return errnode
    end

    -- bind params to variables (use a quote to fake it?)
    local statements = {}
    for i,p_ast in ipairs(QuoteParams(param_asts)) do
        local decl = ast.DeclStatement:DeriveFrom(src_node)
        decl.name = f.params[i]
        decl.initializer = p_ast
        statements[i] = decl
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
        expansion.node_type = L.internal('no-return function')
    end

    -- wrap up in a quote
    local q = ast.Quote:DeriveFrom(src_node)
    q.code = expansion

    -- return the result of typechecking the quote we built
    return q:check(ctxt)
end

function ast.TableLookup:check(ctxt)
    local tab = self.table:check(ctxt)
    local member = self.member
    local ttype = tab.node_type

    if ttype == L.error then
        return err(self, ctxt)
    end

    if ttype:isScalarKey() then
        local luaval = ttype.relation[member]

        -- create a field access normally
        if L.is_field(luaval) then
            local field         = luaval
            local ast_node      = ast.FieldAccess:DeriveFrom(tab)
            ast_node.name       = member
            ast_node.key        = tab
            local name = ast_node.key.name
            if name and ctxt:iscenter(name) then
                ast_node.key.is_centered = true
            end
            ast_node.field      = field
            ast_node.node_type  = field.type
            return ast_node

        -- desugar macro-fields from key.macro to macro(key)
        elseif L.is_macro(luaval) then
            return RunMacro(ctxt,self,luaval,{tab})
        -- desugar function-fields from key.func to func(key)
        elseif L.is_function(luaval) then
            return InlineUserFunc(ctxt,self,luaval,{tab})
        else
            return err(self, ctxt, "Key from "..ttype.relation:Name()..
                                   " does not have field or macro-field "..
                                   "'"..member.."'")
        end
    elseif ttype:isQuery() then
        local rel = ttype.relation
        local ct  = tab:clone()
        for k,v in pairs(tab) do ct[k] = v end
        local projs = {}
        for i,p in ipairs(ttype.projections) do
            table.insert(projs,p)
            rel = rel[p].type.relation 
            assert(rel)
        end
        local field = rel[member]
        if not L.is_field(field) then
            ctxt:error(self, "Relation "..rel:Name().." does not have field "..member)
        else 
            table.insert(projs,member)
        end
        ct.node_type = L.query(ttype.relation,projs)
        return ct
    else
        return err(self, ctxt, "select operator not "..
                               "supported for "..
                               ttype:toString())
    end

end

function ast.SquareIndex:check(ctxt)
    -- mutate squareindex into a node of type base

    local base  = self.base:check(ctxt)
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

    if btyp == L.error or ityp == L.error then
        sqidx.node_type = L.error
        return sqidx
    end

    if not ityp:isIntegral() then return err(idx, ctxt, 'expected an '..
        'integer index, but found '..ityp:toString()) end

    -- matrix case
    if self.index2 then
        local idx2      = self.index2:check(ctxt)
        sqidx.index2     = idx2
        local i2typ     = idx2.node_type

        if i2typ == L.error then
            sqidx.node_type = L.error
            return sqidx
        end

        if not ityp:isIntegral() then return err(idx2, ctxt, 'expected an '..
            'integer index, but found '..i2typ:toString()) end
        if not btyp:isMatrix() then return err(base, ctxt, 'expected '..
            'small matrix to index into, not '.. btyp:toString()) end
    -- vector case
    else
        if not btyp:isVector() then return err(base, ctxt, 'expected '..
            'vector to index into, not '..btyp:toString()) end
    end

    -- is an lvalue only when the base is
    if base.is_lvalue then sqidx.is_lvalue = true end
    sqidx.node_type = btyp:baseType()
    return sqidx
end

function ast.Call:check(ctxt)
    local call = self:clone()
    
    call.node_type = L.error -- default
    local func     = self.func:check(ctxt)
    call.params    = {}

    for i,p in ipairs(self.params) do
        call.params[i] = p:check(ctxt)
    end

    local v = func.node_type:isInternal() and func.node_type.value
    if v and L.is_builtin(v) then
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
    elseif v and L.is_macro(v) then
        -- replace the call node with the inlined AST
        call = RunMacro(ctxt, self, v, call.params)
    elseif v and L.is_function(v) then
        call = InlineUserFunc(ctxt, self, v, call.params)
    elseif v and T.isLisztType(v) and v:isValueType() then
        local params = call.params
        if #params ~= 1 then
            ctxt:error(self, "Cast to " .. v:toString() ..
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
    elseif func.node_type:isScalarKey() then
        local apply_macro = func.node_type.relation.__apply_macro
        if L.is_macro(apply_macro) then
            local params = {func}
            for _,v in ipairs(call.params) do table.insert(params, v) end
            call = RunMacro(ctxt, self, apply_macro, params)
        else
            ctxt:error(self, "Relation "..tostring(func.node_type.relation)..
                             " does not have an __apply_macro"..
                             " macro defined; cannot call key.")
        end
    elseif func.node_type:isError() then
        -- fall through
        -- (do not print error messages for errors already reported)
    else
        ctxt:error(self, "This call was neither a function nor macro.")

    end

    return call
end

function ast.Global:check(ctxt)
    local n     = self:clone()
    n.node_type = self.global.type
    return n
end

function ast.Quote:check(ctxt)
    -- Ensure quotes are only typed once
    -- By typing the quote at declaration, we make it safe
    -- to included it in other code as is
    if self.node_type then
        return self
    else
        local q = self:clone()

        q.code = self.code:check(ctxt)
        q.node_type = q.code.node_type

        return q
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
    assert(self.node_type and self.node_type:isInternal())
    return self
end
function ast.Where:check(ctxt)
    --note: where is generated in an internal macro,
    --      so its fields are already type-checked
    local fieldobj = self.field.node_type
    local keytype  = self.key.node_type
    if not fieldobj:isInternal() or not L.is_field(fieldobj.value) then
        ctxt:error(self,"Expected a field as the first argument but found "
                   ,fieldobj)
    end
    local field = fieldobj.value
    if keytype ~= field.type then
        ctxt:error(self,"Key of where is type ",keytype,
                   " but expected type ",field.type)
    end
    if not field.owner:isGrouped() or
       field.owner:GroupedKeyField() ~= field
    then
        ctxt:error(self,"Relation '"..field.owner:Name().."' is not "..
                        "grouped by Field '"..field:Name().."'")
    end
    local w     = self:clone()
    w.relation  = field.owner
    w.field     = self.field -- for safety/completeness
    w.key       = self.key
    w.node_type = L.query(w.relation,{})
    return w
end

------------------------------------------------------------------------------
--[[ Semantic checking called from here:                                  ]]--
------------------------------------------------------------------------------
function ast.LisztKernel:check(ctxt)
    local kernel              = self:clone()
    kernel.name               = self.name

    local set    = self.set:check(ctxt)
    if set.node_type:isInternal() and
       L.is_relation(set.node_type.value)
    then
        kernel.relation             = set.node_type.value
        local key_type              = L.key(kernel.relation)
        -- record the center
        ctxt:recordcenter(kernel.name)
        ctxt:liszt()[kernel.name]   = key_type
        kernel.body                 = self.body:check(ctxt)
    else
        ctxt:error(kernel.set, "Expected a relation")
    end

    return kernel
end

function S.check(kernel_ast)
    -- environment for checking variables and scopes
    local env  = terralib.newenvironment(nil)
    local diag = terralib.newdiagnostics()
    local ctxt = Context.new(env, diag)

    --------------------------------------------------------------------------

    diag:begin()
    env:enterblock()
    local new_kernel_ast = kernel_ast:check(ctxt)
    env:leaveblock()
    diag:finishandabortiferrors("Errors during typechecking liszt", 1)
    return new_kernel_ast
end
