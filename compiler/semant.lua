local S = {}
package.loaded["compiler.semant"] = S

local ast = require "compiler.ast"
local B   = terralib.require "compiler.builtins"
local T   = terralib.require "compiler.types"

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

local P = terralib.require 'compiler/phase'

local Context = {}
Context.__index = Context

function Context.new(env, diag)
    local ctxt = setmetatable({
        env         = env,
        diag        = diag,
        loop_count  = 0,
        query_count = 0,
        phaseDict   = P.PhaseDict.new(diag),
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
function Context:enterlhs (reduce_op)
    self.phaseDict:enterLhs(reduce_op)
end
function Context:enterrhs ()
    self.phaseDict:enterRhs()
end
function Context:leavelhs ()
    self.phaseDict:leaveLhs()
end
function Context:leaverhs ()
    self.phaseDict:leaveRhs()
end
function Context:log_field_access (field, node)
    return self.phaseDict:store(field, node)
end
function Context:field_usage ()
    return self.phaseDict:export()
end

local function exec_external(exp,ctxt,default)
    local status, v = pcall(function()
        return exp(ctxt:lua())
    end)
    if not status then
        ctxt:error(ast_node, "Error evaluating type annotation: "..typ)
        v = default
    end
    return v
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

-- This is like a type meet.  Except we coerce the arguments
-- as necessary and then return them
local function try_bin_coerce(node_a, node_b)
    if node_a.node_type == node_b.node_type then
        return node_a, node_b
    elseif node_a.node_type:isCoercableTo(node_b.node_type) then
        return try_coerce(node_b.node_type, node_a), node_b
    elseif node_b.node_type:isCoercableTo(node_a.node_type) then
        return node_a, try_coerce(node_a.node_type, node_b)
    else
        return nil, nil
    end
end

-- like above, but if one of the operands is a vector, then
-- only coerce the base types into agreement
local function try_vec_bin_coerce(node_a, node_b)
    if node_a.node_type:isVector() == node_b.node_type:isVector() then
        return try_bin_coerce(node_a, node_b)
    elseif node_b.node_type:isVector() then
        local b, a = try_vec_bin_coerce(node_b, node_a)
        return a, b
    else -- now a is vec, b is primitive
        local abase = node_a.node_type:baseType()
        local N     = node_a.node_type.N
        if abase == node_b.node_type then
            return node_a, node_b
        elseif abase:isCoercableTo(node_b.node_type) then
            return try_coerce(L.vector(node_b.node_type, N), node_a), node_b
        elseif node_b.node_type:isCoercableTo(abase) then
            return node_a, try_coerce(abase, node_b)
        else
            return nil, nil
        end
    end
end

------------------------------------------------------------------------------
--[[ AST semantic checking methods:                                       ]]--
------------------------------------------------------------------------------
function ast.AST:check(ctxt)
    error("Typechecking not implemented for AST node " .. self.kind)
end

function ast.Block:check(ctxt)
    local block = self:clone()

    -- statements
    block.statements = {}
    for id, node in ipairs(self.statements) do
        block.statements[id] = node:check(ctxt)
    end

    return block
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
    local expstmt = self:clone()
    expstmt.exp   = self.exp:check(ctxt)
    return expstmt
end

--function ast.Reduce:check(ctxt)
--    ctxt:error(self, "Reduction should have been refactored in the parser. (--Report to the developers)")
--    return L.error
--    local exp = self.exp:check(ctxt)
--
--    -- When reduced, a scalar can be an lvalue
--    if exp:is(ast.Scalar) then
--        exp.is_lvalue = true
--    end
--
--    -- only lvalues can be "reduced"
--    if exp.is_lvalue then
--        return exp
--
--    else
--        ctxt:error(self, "only lvalues can be reduced.")
--        local errnode = self:clone()
--        errnode.node_type = L.error
--        return errnode
--    end
--end

function ast.Assignment:check(ctxt)
    local assignment = self:clone()
    assignment.reduceop = self.reduceop

    -- LHS tracked for phase-checking
    ctxt:enterlhs(self.reduceop or nil)
    local lhs = self.lvalue:check(ctxt)
    ctxt:leavelhs()

    assignment.lvalue = lhs
    local ltype       = lhs.node_type
    if ltype == L.error then return assignment end

    local rhs         = self.exp:check(ctxt)
    assignment.exp    = rhs
    local rtype       = rhs.node_type
    if rtype == L.error then return assignment end

    -- Promote scalar lhs to lvalue if there was a reduction
    if self.reduceop and lhs:is(ast.Scalar) then
        lhs.is_value = true
    end

    -- enforce that the lhs is an lvalue
    if not lhs.is_lvalue then
        ctxt:error(self.lvalue, "Illegal assignment: left hand side cannot "..
                                "be assigned")
        return assignment
    elseif lhs.node_type:isRow() then
        ctxt:error(self.lvalue, "Illegal assignment: variables of row type "..
                                "cannot be re-assigned")
        return assignment
    end

    -- handle any coercions
    if ltype ~= rtype then
        assignment.exp = try_coerce(ltype, assignment.exp, ctxt)
        if not assignment.exp then return assignment end
    end

    -- replace assignment with a field write if we see a
    -- field access on the left hand side
    if assignment.lvalue:is(ast.FieldAccess) then
        local fw = ast.FieldWrite:DeriveFrom(assignment)
        fw.fieldaccess = assignment.lvalue
        fw.exp         = assignment.exp
        fw.reduceop    = assignment.reduceop
        return fw
    end

    return assignment
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

function ast.DeclStatement:check(ctxt)
    local decl = self:clone()
    decl.name  = self.name
    -- assert(self.typeexpression or self.initializer)

    -- process any explicit type annotation
    if self.typeexpression then
        decl.node_type = exec_type_annotation(self.typeexpression, self, ctxt)
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
    end

    if decl.node_type ~= L.error and not decl.node_type:isFieldType() then
        ctxt:error(self,"can only assign numbers, bools, "..
                        "or rows to local temporaries")
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
    numfor.name      = self.name
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
        rel = rel[p].type.relation
        assert(rel)
    end
    local rowType = L.row(rel)
    ctxt:enterblock()
    ctxt:enterloop()
    ctxt:liszt()[r.name] = rowType
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

--[[ Logic tables for binary expression checking: ]]--
-- terra does not support vector types as operands for this operator
local isNumOp = {
    ['^'] = true
}

-- these operators always return logical types!
local is_eq_compare = {
    ['=='] = true,
    ['~='] = true
}

-- only logical operands
local is_logical_op = {
    ['and'] = true,
    ['or']  = true
}

local is_num_compare_op = {
    ['<='] = true,
    ['>='] = true,
    ['>']  = true,
    ['<']  = true
}

local is_arith_op = {
    ['+']  = true,
    ['-']  = true,
    ['*']  = true,
    ['/']  = true
}

local function err (node, ctx, msg)
    node.node_type = L.error
    if msg then ctx:error(node, msg) end
    return node
end

-- binary expressions
function ast.BinaryOp:check(ctxt)
    local binop         = self:clone()
    binop.op            = self.op
    binop.lhs           = self.lhs:check(ctxt)
    binop.rhs           = self.rhs:check(ctxt)
    local ltype, rtype  = binop.lhs.node_type, binop.rhs.node_type

    -- Silently ignore/propagate errors
    if ltype == L.error or rtype == L.error then return err(self, ctxt) end

    local type_err = 'incompatible types: ' .. ltype:toString() ..
                     ' and ' .. rtype:toString()
    local op_err   = 'invalid types for operator \'' .. binop.op .. '\': ' ..
                     ltype:toString() .. ' and ' .. rtype:toString()

    if not ltype:isValueType() or not rtype:isValueType() then
        return err(self, ctxt, op_err)
    end

    -- operators on booleans
    if is_logical_op[binop.op] then
        if ltype ~= rtype then              return err(binop, ctxt, type_err)
        elseif not ltype:isLogical() then   return err(binop, ctxt, op_err)
        else                                binop.node_type = ltype
                                            return binop end
    end

    -- operators for numeric primitives only
    if is_num_compare_op[binop.op] or binop.op == '^' or binop.op == '%' then
        if not ltype:isPrimitive() or not ltype:isNumeric() or
           not rtype:isPrimitive() or not rtype:isNumeric() or
           (binop.op == '%' and
            (not ltype:isIntegral() or not rtype:isIntegral()))
        then
            return err(binop, ctxt, op_err)
        end

        -- coerce the arguments then
        binop.lhs, binop.rhs = try_bin_coerce(binop.lhs, binop.rhs)
        if not binop.lhs then return err(binop, ctxt, type_err) end

        if is_num_compare_op[binop.op] then 
            binop.node_type = L.bool
        else
            binop.node_type = binop.lhs.node_type
        end
        return binop
    end

    -- equality / inequality (the eternal problem)
    if is_eq_compare[binop.op] then
        -- try coercion
        binop.lhs, binop.rhs = try_bin_coerce(binop.lhs, binop.rhs)
        if not binop.lhs then return err(binop, ctxt, type_err) end

        binop.node_type = L.bool
        return binop
    end

    -- all remaining cases of basic arithmetic ops
    if is_arith_op[binop.op] then
        if not ltype:isNumeric() or not rtype:isNumeric() then
            return err(binop, ctxt, op_err)
        end

        -- check validity for this kind of operator
        if binop.op == '+' or binop.op == '-' then
            if ltype:isVector() ~= rtype:isVector() then
                return err(binop, ctxt, op_err)
            end
        elseif binop.op == '*' then
            if ltype:isVector() and rtype:isVector() then
                return err(binop, ctxt, op_err)
            end
        elseif binop.op == '/' then
            if rtype:isVector() then
                return err(binop, ctxt, op_err)
            end
        end

        -- coerce types
        binop.lhs, binop.rhs = try_vec_bin_coerce(binop.lhs, binop.rhs)
        if not binop.lhs then return err(binop, ctxt, type_err) end

        -- make sure we get a vector type if appropriate
        binop.node_type = binop.lhs.node_type
        if rtype:isVector() then binop.node_type = binop.rhs.node_type end
        return binop
    end

    -- We should never get here...
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
-- This function attempts to produce an AST node which looks as if
-- the resulting AST subtree has just been emitted from the Parser
local function luav_to_ast(luav, src_node)
    -- try to construct an ast node to return...
    local node

    -- Scalar objects are replaced with special Scalar nodes
    if L.is_scalar(luav) then
        node        = ast.Scalar:DeriveFrom(src_node)
        node.scalar = luav

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
    elseif B.isFunc(luav) then
        node = NewLuaObject(src_node,luav)
    elseif L.is_relation(luav) then
        node = NewLuaObject(src_node,luav)
    elseif L.is_macro(luav) then
        node = NewLuaObject(src_node,luav)
    elseif terralib.isfunction(luav) then
        node = NewLuaObject(src_node,B.terra_to_func(luav))
    elseif type(luav) == 'table' then
        -- Determine whether this is an AST node
        if ast.is_ast(luav) and luav:is(ast.QuoteExpr) then
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

-- luav_to_checked_ast wraps luav_to_ast and ensures that
-- a typed AST node is returned.
local function luav_to_checked_ast(luav, src_node, ctxt)
    -- convert the lua value into an ast node
    local ast_node = luav_to_ast(luav, src_node)

    -- on conversion error
    if not ast_node then
        ctxt:error(src_node, "could not convert Lua value to a Liszt value")
        ast_node           = src_node:clone()
        ast_node.node_type = L.error

    -- on successful conversion to an ast node
    else
        ast_node = ast_node:check(ctxt)
    end

    return ast_node
end

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
        return node
    end

    -- Otherwise, does the name exist in the lua scope?
    local luav = ctxt:lua()[self.name]
    if luav then
        -- convert the lua value into an ast node
        return luav_to_checked_ast(luav, self, ctxt)
    end


    -- failed to find this name anywhere
    ctxt:error(self, "variable '" .. self.name .. "' is not defined")
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
    elseif tonumber(self.value) % 1 == 0 then
        number.node_type = L.int
    else
        number.node_type = L.double
    end
    return number
end

function ast.VectorLiteral:check(ctxt)
    local veclit      = self:clone()
    veclit.elems      = {}

    local tp_error = "vector literals can only contain values "..
                     "of boolean or numeric type"
    local mt_error = "vector entries must be of the same type"

    ctxt:enterrhs()

    veclit.elems[1]   = self.elems[1]:check(ctxt)
    local max_type    = veclit.elems[1].node_type
    if max_type == L.error then ctxt:leaverhs(); return err(self, ctxt) end
    if not max_type:isPrimitive() then
        ctxt:leaverhs(); return err(self, ctxt, tp_error) 
    end

    -- scan the remaining entries to compute a max type
    for i = 2, #self.elems do
        veclit.elems[i] = self.elems[i]:check(ctxt)
        local tp        = veclit.elems[i].node_type
        if tp == L.error then ctxt:leaverhs(); return err(self, ctxt) end
        if not tp:isPrimitive() then
            ctxt:leaverhs(); return err(self,ctxt, tp_error)
        end

        if max_type:isCoercableTo(tp) then
            max_type = tp
        elseif not tp:isCoercableTo(max_type) then
            ctxt:leaverhs(); return err(self, ctxt, mt_error)
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

    ctxt:leaverhs()

    veclit.node_type = L.vector(max_type, #veclit.elems)
    return veclit
end

function ast.Bool:check(ctxt)
    local boolnode     = self:clone()
    boolnode.value     = self.value
    boolnode.node_type = L.bool
    return boolnode
end


------------------------------------------------------------------------------
--[[                         Miscellaneous nodes:                         ]]--
------------------------------------------------------------------------------

local function QuoteParam(param_ast)
    local q = ast.QuoteExpr:DeriveFrom(param_ast)
    q.block, q.exp = nil, param_ast
    q.node_type    = param_ast.node_type -- do not try to type-check these
    return q
end

local function RunMacro(ctxt,src_node,the_macro,params)
    local quoted_params = {}
    for i,v in ipairs(params) do
        quoted_params[i] = QuoteParam(params[i])
    end
    local result = the_macro.genfunc(unpack(quoted_params))
    return luav_to_checked_ast(result, src_node, ctxt)
end

function ast.TableLookup:check(ctxt)
    ctxt:enterrhs()
    local tab = self.table:check(ctxt)
    ctxt:leaverhs()
    local member = self.member
    if type(member) == "function" then --member is an escaped lua expression
      member = exec_external(member,ctxt,"<error>")
      if type(member) ~= "string" then
        ctxt:error(self,"expected escape to evaluate to a string but found ", type(member))
        member = "<error>"
      end
    end
    local ttype = tab.node_type

    if ttype == L.error then
        return err(self, ctxt)
    end

    -- table is a lua table
    if ttype:isInternal() then
        local thetable = ttype.value
        -- perform the lookup in the lua table,
        -- which we assume is acting as a namespace
        local luaval = thetable[member]

        if luaval == nil then
            return err(self, ctxt, "lua table " ..
                       "does not have member '" .. member .. "'")
        end

        -- and then convert the lua value into an ast node
        return luav_to_checked_ast(luaval, self, ctxt)
    elseif ttype:isRow() then
        local luaval = ttype.relation[member]

        -- create a field access normally
        if L.is_field(luaval) then
            local field         = luaval
            local ast_node      = ast.FieldAccess:DeriveFrom(tab)
            ast_node.name       = self.member
            ast_node.row        = tab
            ast_node.field      = field
            ast_node.node_type  = field.type

            -- log field accesses for phase analysis
            ctxt:log_field_access(luaval, ast_node)

            return ast_node

        -- desugar macro-fields from row.macro to macro(row)
        elseif L.is_macro(luaval) then
            return RunMacro(ctxt,self,luaval,{tab})
        else
            return err(self, ctxt, "Row "..ttype.relation:Name().." does not "..
                                   "have field or macro-field "..
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

function ast.VectorIndex:check(ctxt)
    local vidx   = self:clone()
    local vec    = self.vector:check(ctxt)
    local idx    = self.index:check(ctxt)
    vidx.vector, vidx.index = vec, idx
    local vectype, idxtype  = vec.node_type, idx.node_type

    -- RHS is an expression of integral type
    -- (make sure this check always runs)
    if idxtype ~= L.error and not idxtype:isIntegral() then
        ctxt:error(self, "expected an integer expression to index into "..
                         "the vector (found ".. idxtype:toString() ..')')
    end

    -- LHS should be a vector
    if vectype ~= L.error and not vectype:isVector() then
        ctxt:error(self, "indexing operator [] not supported for "..
                         "non-vector type " .. vectype:toString())
        vectype = L.error
    end
    if vectype == L.error then
        vidx.node_type = L.error
        return vidx
    end

    -- is an lvalue only when the vector is
    if vec.is_lvalue then vidx.is_lvalue = true end

    vidx.node_type = vectype:baseType()
    return vidx
end

function ast.Call:check(ctxt)
    local call = self:clone()
    
    call.node_type = L.error -- default
    local func     = self.func:check(ctxt)
    call.params    = {}

    ctxt:enterlhs()
    for i,p in ipairs(self.params) do
        call.params[i] = p:check(ctxt)
    end
    ctxt:leavelhs()

    local isinternal = func.node_type:isInternal()
    local v = isinternal and func.node_type.value
    if v and L.is_function(v) then
        call.func      = v
        call.node_type = v.check(call, ctxt)
    elseif v and L.is_macro(v) then
        -- replace the call node with the inlined AST
        call = RunMacro(ctxt, self, v, call.params)
    elseif v and T.isLisztType(v) and v:isValueType() then
        local params = call.params
        call = ast.Cast:DeriveFrom(self)
        if #params == 1 then
            call.value     = params[1]
            call.node_type = v
        else
            ctxt:error(self, "Cast to " .. v.toString() ..
                    " expects exactly 1 argument (instead got " .. #params ..
                    ")")
        end
    elseif func.node_type:isError() then
        -- fall through
        -- (do not print error messages for errors already reported)
    else
        ctxt:error(self, "This call was neither a function nor macro.")

    end

    return call
end

function ast.Scalar:check(ctxt)
    local n     = self:clone()
    n.scalar    = self.scalar
    n.node_type = self.scalar.type
    return n
end

function ast.QuoteExpr:check(ctxt)
    -- Ensure quotes are only typed once
    -- By typing the quote at declaration, we make it safe
    -- to included it in other code as is
    if not self.node_type then
        local q     = self:clone()
        if not self.block and not self.exp then
            ctxt:error('Found a Quote with no block and no expression.')
        else
            ctxt:enterblock()
            if self.block then
                q.block = self.block:check(ctxt)
            end
            q.exp       = self.exp:check(ctxt)
            ctxt:leaveblock()

            q.node_type = q.exp.node_type
        end
        return q
    else
        return self
    end
end

function ast.FieldAccess:check(ctxt)
    local n = self:clone()
    n.field = self.field
    n.row   = self.row
    return n
end

function ast.LuaObject:check(ctxt)
    assert(self.node_type and self.node_type:isInternal())
    return self
end
function ast.Where:check(ctxt)
    --note: where is generated in a macro, so its fields are already type-checked
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
    if field.owner._index ~= field then
        ctxt:error(self,"Field ",field.name,
                   " is not an index of ",field.owner:Name())
    end
    local w     = self:clone()
    w.relation  = field.owner
    w.key       = self.key
    w.node_type = L.query(w.relation,{})
    ctxt:log_field_access(field, w)
    return w
end

------------------------------------------------------------------------------
--[[ Semantic checking called from here:                                  ]]--
------------------------------------------------------------------------------
function ast.LisztKernel:check(ctxt) -- hacked 2nd parameter
    local kernel              = self:clone()
    kernel.name               = self.name

    local set    = self.set:check(ctxt)
    if set.node_type:isInternal() and
       L.is_relation(set.node_type.value)
    then
        kernel.relation             = set.node_type.value
        ctxt:liszt()[kernel.name]   = L.row(kernel.relation)
        kernel.body                 = self.body:check(ctxt)
    else
        ctxt:error(kernel.set, "Expected a relation")
    end

    return kernel
end

function S.check(luaenv, kernel_ast)
    -- environment for checking variables and scopes
    local env  = terralib.newenvironment(luaenv)
    local diag = terralib.newdiagnostics()
    local ctxt = Context.new(env, diag)

    --------------------------------------------------------------------------

    diag:begin()
    env:enterblock()
    local new_kernel_ast = kernel_ast:check(ctxt)
    env:leaveblock()
    diag:finishandabortiferrors("Errors during typechecking liszt", 1)
    new_kernel_ast.field_usage = ctxt:field_usage()
    return new_kernel_ast
end
