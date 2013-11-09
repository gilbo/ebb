local exports = {}

local ast       = require("ast")
local builtins  = terralib.require("include/builtins")
local types     = terralib.require("compiler/types")
local Type, t   = types.Type, types.t
local type_meet = types.type_meet

--[[
    AST:check(ctxt) type checking routines
        These methods define type checking.
        Each check() method is designed to accept a typing context (see below).
        Once called, check() will construct and return a new type-checked
        version of its subtree (new AST nodes)
]]--


------------------------------------------------------------------------------
--[[ Context Definition                                                   ]]--
------------------------------------------------------------------------------
-- A Context is passed through type-checking, keeping track of any kind of
-- store or gadget we want to use, such as
-- the environment object and error diagnostic object.
-- It can be used to quickly thread new stores
-- through the entire typechecker.
local Context = {}
Context.__index = Context

function Context.new(env, diag)
    local ctxt = setmetatable({
        env         = env,
        diag        = diag,
        lhs_count   = 0,
        loop_count  = 0,
        query_count = 0,
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
function Context:error(ast, msg)
    self.diag:reporterror(ast, msg)
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
function Context:in_query ()
    return self.query_count > 0 
end
function Context:enterquery()
    self.query_count = self.query_count + 1
end
function Context:leavequery()
    self.query_count = self.query_count - 1
    if self.query_count < 0 then self.query_count = 0 end
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
    if condtype ~= t.error and condtype ~= t.bool then
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

    if condtype ~= t.error and condtype ~= t.bool then
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

function ast.Reduce:check(ctxt)
    local exp = self.exp:check(ctxt)

    -- When reduced, a scalar can be an lvalue
    if exp:is(ast.Scalar) then
        exp.is_lvalue = true
    end

    -- only lvalues can be "reduced"
    if exp.is_lvalue then
        return exp

    else
        ctxt:error(self, "only lvalues can be reduced.")
        local errnode = self:clone()
        errnode.node_type = t.error
        return errnode
    end
end

function ast.Assignment:check(ctxt)
    local assignment = self:clone()
    local lhs = self.lvalue:check(ctxt)

    assignment.lvalue = lhs
    local ltype       = lhs.node_type
    if ltype == t.error then return assignment end

    local rhs         = self.exp:check(ctxt)
    assignment.exp    = rhs
    local rtype       = rhs.node_type
    if rtype == t.error then return assignment end

    -- If the left hand side was a reduction store the reduction operation
    if self.lvalue:is(ast.Reduce) then
        assignment.reduceop = self.lvalue.op
    end

    -- enforce that the lhs is an lvalue
    if not lhs.is_lvalue then
        -- TODO: less cryptic error messages in this case
        --   Better error messages probably involves switching on kind of lhs
        ctxt:error(self.lvalue, "assignments in a Liszt kernel are only "..
                                "valid to indexed fields or kernel variables")
        return assignment
    elseif lhs.node_type:isRow() then
        ctxt:error(self.lvalue, "cannot re-assign variables of "..
                                "relational row type")
        return assignment
    end

    -- enforce type agreement b/w lhs and rhs
    local derived = type_meet(ltype,rtype)
    if derived == t.error or
       (ltype:isPrimitive() and rtype:isVector()) or
       (ltype:isVector()    and rtype:isVector() and ltype.N ~= rtype.N)
    then
        ctxt:error(self, "invalid conversion from " .. rtype:toString() ..
                         ' to ' .. ltype:toString())
        return
    end

    return assignment
end

local function exec_type_annotation(typexp, ast_node, ctxt)
    local status, typ = pcall(function()
        return types.terraToLisztType(typexp(ctxt:lua()))
    end)

    if not status then
        ctxt:error(ast_node, "Error evaluating type annotation: "..typ)
        typ = t.error
    end
    if not Type.isLisztType(typ) then
        ctxt:error(ast_node, "Expected Liszt type annotation but found " ..
                             type(typ))
        typ = t.error
    end
    return typ
end

function ast.DeclStatement:check(ctxt)
    local decl    = self:clone()
    decl.ref      = ast.LocalVar:DeriveFrom(self.ref)

    -- catch syntactically invalid initializations
    if not self.typeexpression and not self.initializer then
        ctxt:error(self, "Variables must either be initialized or have "..
                         "an explicitly annotated type.")
        decl.ref.node_type = t.error
        return decl
    end

    -- process any explicit type annotation
    local typ
    if self.typeexpression then
        typ = exec_type_annotation(self.typeexpression, self, ctxt)
    end
    -- check the initialization against the annotation or infer type from init.
    if self.initializer then
        decl.initializer = self.initializer:check(ctxt)
        local exptyp     = decl.initializer.node_type
        -- if the type was annotated check consistency
        if typ then
            local mtyp = type_meet(exptyp,typ)
            if typ ~= mtyp then
                ctxt:error(self, "Cannot assign a value of type " ..
                                 exptyp .. " to type " .. typ)
            end
        -- or infer the type as the expression type
        else
            typ = exptyp
        end
    end
    decl.ref.node_type = typ

    if typ ~= t.error and
         not typ:isExpressionType() and not typ:isRow()
    then
        ctxt:error(self,"can only assign numbers, bools, "..
                        "or relational rows to local temporaries")
    end

    local lv = decl.initializer or decl.ref
    lv.is_lvalue = true
    lv.node_type = typ
    ctxt:liszt()[decl.ref.name] = lv

    return decl
end

function ast.NumericFor:check(ctxt)
    local node = self
    local function check_num_type(tp)
        if tp ~= t.error and not tp:isNumeric() then
            ctxt:error(node, "expected a numeric expression to define the "..
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

    local step, step_type
    if self.step then
        numfor.step = self.step:check(ctxt)
        step_type   = numfor.step.node_type
        check_num_type(step_type)
    end

    -- infer iterator type
    numfor.iter           = ast.LocalVar:DeriveFrom(self.iter)
    numfor.iter.node_type = type_meet(lower_type, upper_type)
    if step_type then
        numfor.iter.node_type = type_meet(numfor.iter.node_type, step_type)
    end

    ctxt:enterblock()
    ctxt:enterloop()
    local varname = numfor.iter.name
    ctxt:liszt()[varname] = numfor.iter
    numfor.body = self.body:check(ctxt)
    ctxt:leaveloop()
    ctxt:leaveblock()

    return numfor
end

function ast.GenericFor:check(ctxt)
    ctxt:error(self, "generic for statement not yet implemented", 2)
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
    if condtype ~= t.error and condtype ~= t.bool then
        ctxt:error(self, "conditional expression type should be"..
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
          "expression type " .. self.kind)
end

--[[ Logic tables for binary expression checking: ]]--
-- terra does not support vector types as operands for this operator
local isNumOp = {
    ['^'] = true
}

-- these operators always return logical types!
local isCompOp = {
    ['<='] = true,
    ['>='] = true,
    ['>']  = true,
    ['<']  = true,
    ['=='] = true,
    ['~='] = true
}

-- only logical operands
local isBoolOp = {
    ['and'] = true,
    ['or']  = true
}

local function err (node, ctx, msg)
    node.node_type = t.error
    if msg then ctx:error(node, msg) end
    return node
end

-- binary expressions
function ast.BinaryOp:check(ctxt)
    local binop = self:clone()
    binop.op    = self.op
    binop.lhs   = self.lhs:check(ctxt)
    binop.rhs   = self.rhs:check(ctxt)
    local ltype, rtype = binop.lhs.node_type, binop.rhs.node_type

    -- Silently ignore/propagate errors
    if ltype == t.error or rtype == t.error then return err(self, ctxt) end

    local type_err = 'incompatible types: ' .. ltype:toString() ..
                     ' and ' .. rtype:toString()
    local op_err   = 'invalid types for operator \'' .. binop.op .. '\': ' ..
                     ltype:toString() .. ' and ' .. rtype:toString()

    if not ltype:isExpressionType() or not rtype:isExpressionType() then
        return err(self, ctxt, op_err)
    end

    local derived = type_meet(ltype, rtype)

    -- Numeric op operands cannot be vectors
    if isNumOp[binop.op] then
        if ltype:isVector() or rtype:isVector() then return err(binop, ctxt, op_err) end

    elseif isCompOp[binop.op] then
        if ltype:isLogical() ~= rtype:isLogical() then
            return err(binop, ctxt, op_err)

        -- if the type_meet failed, types are incompatible
        elseif derived == t.error then
            return err(binop, ctxt, type_err)

        -- otherwise, we need to return a logical type
        else
            binop.node_type =
              derived:isPrimitive() and t.bool or t.vector(t.bool, derived.N)
            return binop
        end

    elseif isBoolOp[binop.op] then
        if not ltype:isLogical() or not rtype:isLogical() then
            return err(binop, ctxt, op_err)
        end
    end

    if derived == t.error then
        ctxt:error(self, type_err)
    end

    binop.node_type = derived
    return binop
end

function ast.UnaryOp:check(ctxt)
    local unop     = self:clone()
    unop.op        = self.op
    unop.exp       = self.exp:check(ctxt)
    local exptype  = unop.exp.node_type
    unop.node_type = t.error -- default

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


------------------------------------------------------------------------------
--[[                               Variables                              ]]--
------------------------------------------------------------------------------
-- This function attempts to produce an AST node which looks as if
-- the resulting AST subtree has just been emitted from the Parser
local function luav_to_ast(luav, src_node)
    -- try to construct an ast node to return...
    local node

    -- Scalar objects are replaced with special Scalar nodes
    if Type.isScalar(luav) then
        node        = ast.Scalar:DeriveFrom(src_node)
        node.scalar = luav

    -- Vector objects are expanded into literal AST trees
    elseif Type.isVector(luav) then
        node        = ast.VectorLiteral:DeriveFrom(src_node)
        node.elems  = {}
        for i,v in ipairs(luav.data) do
            node.elems[i] = luav_to_ast(v, src_node)
        end

    elseif Type.isRelation(luav) then
        node           = ast.Relation:DeriveFrom(src_node)
        node.relation  = luav
    
    elseif Type.isField(luav) then
        node       = ast.Field:DeriveFrom(src_node)
        node.field = luav

    elseif Type.isFunction(luav) then
        node      = ast.Function:DeriveFrom(src_node)
        node.func = luav

    elseif Type.isMacro(luav) then
        node = ast.Macro:DeriveFrom(src_node)
        node.macro = luav

    elseif terralib.isfunction(luav) then
        node      = ast.Function:DeriveFrom(src_node)
        node.func = builtins.terra_to_func(luav)

    elseif type(luav) == 'table' and luav.is_liszt_ast then
        -- For macro substitution: typed ASTs
        -- may be external and need to be inlined.
        node = ast.QuoteExpr:DeriveFrom(src_node)
        node.ast = luav

    elseif type(luav) == 'table' then
        node       = ast.Table:DeriveFrom(src_node)
        node.table = luav

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
-- IF allow_lua_return is provided, there is a chance that the returned
-- value is a lua table acting as a namespace, which must then be
-- handled appropriately.
local function luav_to_checked_ast(luav, src_node, ctxt)
    -- convert the lua value into an ast node
    local ast_node = luav_to_ast(luav, src_node)

    -- on conversion error
    if not ast_node then
        ctxt:error(src_node, "could not convert Lua value to a Liszt value")
        ast_node = src_node:clone()
        ast_node.node_type = t.error

    -- on successful conversion to an ast node
    else
        ast_node = ast_node:check(ctxt)
    end

    return ast_node
end

function ast.Name:check(ctxt)
    -- try to find the name in the local Liszt scope
    local lisztv = ctxt:liszt()[self.name]

    -- if the name is in the local scope, then it must have been declared
    -- somewhere in the liszt kernel.  Thus, it has to be a primitive, a
    -- bool, or a topological element.
    if lisztv then
        if lisztv.node_type:isExpressionType() then
            local node = ast.LocalVar:DeriveFrom(lisztv)
            node.node_type = lisztv.node_type
            node.is_lvalue = lisztv.is_lvalue
            node.name      = self.name
            return node
        else
            lisztv.name = self.name
            return lisztv:check(ctxt)
        -- check that we're not reading from an uninitialized variable
        end
    end

    -- Otherwise, does the name exist in the lua scope?
    local luav = ctxt:lua()[self.name]
    if luav then
        -- convert the lua value into an ast node
        local ast_node = luav_to_checked_ast(luav, self, ctxt)

        -- track the name this came from for debuging convenience
        ast_node.name = self.name
        return ast_node
    end


    -- failed to find this name anywhere
    ctxt:error(self, "variable '" .. self.name .. "' is not defined")
    local err_node = self:clone()
    err_node.name  = self.name
    err_node.node_type = t.error
    return err_node
end

function ast.Number:check(ctxt)
    local number = self:clone()
    number.value = self.value
    if tonumber(self.value) % 1 == 0 then
        self.node_type = t.int
        number.node_type = t.int
    else
        self.node_type = t.double
        number.node_type = t.double
    end
    return number
end

function ast.VectorLiteral:check(ctxt)
    local veclit = self:clone()
    veclit.elems = {}
    veclit.elems[1]   = self.elems[1]:check(ctxt)
    local type_so_far = veclit.elems[1].node_type

    local tp_error = "vector literals can only contain values "..
                     "of boolean or numeric type"
    local mt_error = "vector entries must be of the same type"

    if type_so_far == t.error then return err(self, ctxt) end
    if not type_so_far:isPrimitive() then return err(self, ctxt, tp_error) end

    for i = 2, #self.elems do
        veclit.elems[i] = self.elems[i]:check(ctxt)
        local tp        = veclit.elems[i].node_type

        if not tp:isPrimitive() then return err(self, ctxt, tp_error) end

        type_so_far = type_meet(type_so_far, tp)
        if type_so_far:isError() then return err(self, ctxt, mt_error) end
    end

    veclit.node_type = t.vector(type_so_far, #veclit.elems)
    return veclit
end

function ast.Bool:check(ctxt)
    local boolnode = self:clone()
    boolnode.value = self.value
    boolnode.node_type = t.bool
    return boolnode
end


------------------------------------------------------------------------------
--[[                         Miscellaneous nodes:                         ]]--
------------------------------------------------------------------------------
function ast.Tuple:check(ctxt)
    local tuple = self:clone()
    tuple.children = {}
    for i, node in ipairs(self.children) do
        tuple.children[i] = node:check(ctxt)
    end
    return tuple
end

function ast.Tuple:index_check(ctxt)
    -- type checking tuple when it should be a single argument, for
    -- instance, when indexing a field
    if #self.children ~= 1 then
        ctxt:error(self, "can use exactly one argument to index here")
        local errnode = self:clone()
        errnode.node_type = t.error
        return errnode
    end
    local arg_ast = self.children[1]:check(ctxt)
    local argtype = arg_ast.node_type
    assert(argtype ~= nil)
    self.node_type = argtype
    return arg_ast
end

function ast.TableLookup:check(ctxt)
    local table  = self.table:check(ctxt)
    local member = self.member
    local ttype  = table.node_type

    self.name = table.name .. '.' .. member.name

    if ttype == t.error then
        return err(self, ctxt)
    elseif table:is(ast.Scalar) then
        return err(self, ctxt,"select operator not supported "..
                              "for non-table type LScalar")
    elseif table:is(ast.Field) then
        return err(self, ctxt, "select operator not supported "..
                               "for non-table type LField")

    -- Table lookups on these types are allowed
    elseif table:is(ast.Table)     then
    elseif table:is(ast.Row)       then
    elseif table:is(ast.Relation)  then
    elseif table:is(ast.FieldAccess) and table.node_type:isRow() then
    else
        return err(self, ctxt, "select operator not "..
                               "supported for non-table type "..
                               ttype:toString())
    end

    self.name = table.name .. '.' .. member.name

    -- table is a lua table
    if table:is(ast.Table) then
        -- perform the lookup in the lua table as namespace
        local luaval = table.table[member.name]

        if luaval == nil then
            return err(self, ctxt, "lua table " .. table.name ..
                       " does not have member '" .. member.name .. "'")
        end

        -- and then convert the lua value into an ast node
        local ast_node = luav_to_checked_ast(luaval, self, ctxt)
        -- we set a name somewhat appropriately (for debug printing)
        ast_node.name = self.name
        return ast_node

    -- table is a Relation, we expect indices to be fields
    elseif table:is(ast.Relation) then
        local luaval = table.relation[member.name]
        -- relational table members must be fields
        if not Type.isField(luaval) then
            ctxt:error("relation " .. table.name ..
                       ' does not have field member ' .. member.name)
            local ast_node = self:clone()
            ast_node.node_type = t.error
            return ast_node
        end
        -- this will return an ast.Field node
        return luav_to_checked_ast(luaval, self, ctxt)

    -- table is a Row, indices are field accesses
    else
        local luaval = table.relation[member.name]
        if not Type.isField(luaval) then
            return err(self, ctxt, "Row "..table.name.." does not have "..
                                   "field member '"..member.name.."'")
        end

        if luaval.macro then
            -- for row.fieldmacro, desugar to fieldmacro(row)
            local macro_node = ast.Call:DeriveFrom(self)
            macro_node.func = ast.Macro:DeriveFrom(self.member)
            macro_node.func.macro = luaval.macro
            macro_node.params = ast.Tuple:DeriveFrom(self.table)
            macro_node.params.children = {self.table}
            return macro_node:check(ctxt)
        else
            local ast_node     = ast.FieldAccess:DeriveFrom(self.member)
            ast_node.name      = self.name
            ast_node.row       = self.table
            ast_node.relation  = luaval.relation
            ast_node.field     = luaval
            ast_node.node_type = luaval.relation and t.row or luaval.type
            return ast_node
        end
    end
end

function ast.VectorIndex:check(ctxt)
    local vidx   = self:clone()
    local vec    = self.vector:check(ctxt)
    local idx    = self.index:check(ctxt)
    vidx.vector, vidx.index = vec, idx
    local vectype, idxtype = vec.node_type, idx.node_type

    -- RHS is an expression of integral type
    -- (make sure this check always runs)
    if idxtype ~= t.error and not idxtype:isIntegral() then
        ctxt:error(self, "expected an integer expression to index into "..
                         "the vector (found ".. idxtype:toString() ..')')
    end

    -- LHS should be a vector
    if vectype ~= t.error and not vectype:isVector() then
        ctxt:error(self, "indexing operator [] not supported for "..
                         "non-vector type " .. vectype:toString())
        vectype = t.error
    end
    if vectype == t.error then
        vidx.node_type = t.error
        return vidx
    end

    -- is an lvalue only when the vector is
    if vec.is_lvalue then vidx.is_lvalue = true end

    vidx.node_type = vectype:baseType()
    return vidx
end

function ast.Call:check(ctxt)
    local call     = self:clone()
    call.node_type = t.error -- default
    call.func      = self.func:check(ctxt)

    if call.func:is(ast.Function) then
        call.params    = self.params:check(ctxt)
        call.node_type = call.func.func.check(call, ctxt)

    elseif call.func:is(ast.Macro) then
        call.params = self.params:check(ctxt)
        -- replace the call node with the inlined AST
        call = call.func.macro.genfunc(
                        unpack(call.params.children)):check(ctxt)

    elseif call.func.node_type:isError() then
        -- fall through
        -- (do not print error messages for errors already reported)

    else
        ctxt:error(self, "invalid call")

    end

    return call
end

function ast.Scalar:check(ctxt)
    local n     = self:clone()
    n.scalar    = self.scalar
    n.node_type = self.scalar.type
    return n
end

function ast.Function:check(ctxt)
    local n     = self:clone()
    n.func      = self.func
    return n
end

function ast.Macro:check(ctxt)
    local n = self:clone()
    n.macro = self.macro
    return n
end

function ast.QuoteExpr:check(ctxt)
    return self.ast
end

function ast.FieldAccess:check(ctxt)
    local n     = self:clone()
    n.field     = self.field
    n.row       = self.row
    n.relation  = self.relation
    return n
end

function ast.Field:check(ctxt)
    local n = self:clone()
    n.field = self.field
    return n
end

function ast.Table:check(ctxt)
    local n = self:clone()
    n.table = self.table
    return n
end

function ast.Relation:check(ctxt)
    local n    = self:clone()
    n.relation = self.relation
    return n
end

function ast.Row:check(ctxt)
    local n    = self:clone()
    n.relation = self.relation
    n.row      = self.row
    n.field    = self.field
    return n
end

function ast.LocalVar:check(ctxt)
    return self:clone()
end


------------------------------------------------------------------------------
--[[ Semantic checking called from here:                                  ]]--
------------------------------------------------------------------------------
function ast.LisztKernel:check(ctxt)
    local new_kernel_ast = self:clone()

    local set           = self.set:check(ctxt)
    local iter          = ast.Row:DeriveFrom(self.iter)
    new_kernel_ast.set  = set
    iter.name           = self.iter.name
    iter.relation       = new_kernel_ast.set.relation
    new_kernel_ast.iter = iter:check()

    if not set:is(ast.Relation) then
        ctxt:error(new_kernel_ast.set, "Expected a relation")
    else
        ctxt:liszt()[self.iter.name] = iter
    end

    new_kernel_ast.body = self.body:check(ctxt)
    return new_kernel_ast
end

function exports.check(luaenv, kernel_ast, param_type)

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
    return new_kernel_ast
end


return exports
