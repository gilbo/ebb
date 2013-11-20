local S = {}
package.loaded["compiler.semant"] = S

local ast = require "compiler.ast"
local B = terralib.require "compiler.builtins"
local T = terralib.require "compiler.types"

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
        errnode.node_type = L.error
        return errnode
    end
end

function ast.Assignment:check(ctxt)
    local assignment = self:clone()
    local lhs = self.lvalue:check(ctxt)

    assignment.lvalue = lhs
    local ltype       = lhs.node_type
    if ltype == L.error then return assignment end

    local rhs         = self.exp:check(ctxt)
    assignment.exp    = rhs
    local rtype       = rhs.node_type
    if rtype == L.error then return assignment end

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
                                "row type")
        return assignment
    end

    -- enforce type agreement b/w lhs and rhs
    local derived = T.type_meet(ltype,rtype)
    if derived == L.error or
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
        return typexp(ctxt:lua())
    end)

    if not status then
        ctxt:error(ast_node, "Error evaluating type annotation: "..typ)
        typ = L.error
    end
    if not T.isLisztType(typ) then
        ctxt:error(ast_node, "Expected Liszt type annotation but found " ..
                             type(typ))
        typ = L.error
    end
    return typ
end

function ast.DeclStatement:check(ctxt)
    local decl    = self:clone()
    decl.name     = self.name

    -- catch syntactically invalid initializations
    if not self.typeexpression and not self.initializer then
        ctxt:error(self, "Variables must either be initialized or have "..
                         "an explicitly annotated type.")
        decl.node_type = L.error
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
            local mtyp = T.type_meet(exptyp,typ)
            if typ ~= mtyp then
                ctxt:error(self, "Cannot assign a value of type ",
                                  exptyp, " to type ", typ)
            end
        -- or infer the type as the expression type
        else
            typ = exptyp
        end
    end
    decl.node_type = typ

    if typ ~= L.error and
         not typ:isValueType() and not typ:isRow()
    then
        ctxt:error(self,"can only assign numbers, bools, "..
                        "or rows to local temporaries")
    end

    ctxt:liszt()[decl.name] = typ

    return decl
end

function ast.NumericFor:check(ctxt)
    local node = self
    local function check_num_type(tp)
        if tp ~= L.error and not tp:isNumeric() then
            ctxt:error(node, "expected a numeric expression to define the "..
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

    local step, step_type
    if self.step then
        numfor.step = self.step:check(ctxt)
        step_type   = numfor.step.node_type
        check_num_type(step_type)
    end

    -- infer iterator type
    numfor.name           = self.name
    numfor.node_type = T.type_meet(lower_type, upper_type)
    if step_type then
        numfor.node_type = T.type_meet(numfor.node_type, step_type)
    end

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
    r.name = self.name
    r.set = self.set:check(ctxt)
    if not r.set.node_type:isSet() then
        ctxt:error(self,"for statement expects a set but found type ",r.set.node_type)
    end
    local rowType = L.row(r.set.node_type.relation)
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
          "expression type " .. self.kind, 2)
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
    node.node_type = L.error
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
    if ltype == L.error or rtype == L.error then return err(self, ctxt) end

    local type_err = 'incompatible types: ' .. ltype:toString() ..
                     ' and ' .. rtype:toString()
    local op_err   = 'invalid types for operator \'' .. binop.op .. '\': ' ..
                     ltype:toString() .. ' and ' .. rtype:toString()

    if not ltype:isValueType() or not rtype:isValueType() then
        return err(self, ctxt, op_err)
    end

    local derived = T.type_meet(ltype, rtype)

    -- Numeric op operands cannot be vectors
    if isNumOp[binop.op] then
        if ltype:isVector() or rtype:isVector() then return err(binop, ctxt, op_err) end

    elseif isCompOp[binop.op] then
        if ltype:isLogical() ~= rtype:isLogical() then
            return err(binop, ctxt, op_err)

        -- if the type_meet failed, types are incompatible
        elseif derived == L.error then
            return err(binop, ctxt, type_err)

        -- otherwise, we need to return a logical type
        else
            binop.node_type =
              derived:isPrimitive() and L.bool or L.vector(L.bool, derived.N)
            return binop
        end

    elseif isBoolOp[binop.op] then
        if not ltype:isLogical() or not rtype:isLogical() then
            return err(binop, ctxt, op_err)
        end
    end

    if derived == L.error then
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
    local lo = ast.LuaObject:DeriveFrom(anchor)
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
        node        = ast.VectorLiteral:DeriveFrom(src_node)
        node.elems  = {}
        for i,v in ipairs(luav.data) do
            node.elems[i] = luav_to_ast(v, src_node)
        end
    elseif B.isFunc(luav) then
        node = NewLuaObject(src_node,luav)
    elseif L.is_relation(luav) then
        node = NewLuaObject(src_node,luav)
    elseif L.is_macro(luav) then
        node = NewLuaObject(src_node,luav)
    elseif terralib.isfunction(luav) then
        node = NewLuaObject(src_node,B.terra_to_func(luav))
    elseif type(luav) == 'table' and luav.is_liszt_ast then
        -- For macro substitution: typed ASTs
        -- may be external and need to be inlined.
        node = ast.QuoteExpr:DeriveFrom(src_node)
        node.ast = luav
    elseif type(luav) == 'table' then
        node = NewLuaObject(src_node, luav)
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
        local node = ast.Name:DeriveFrom(self)
        node.node_type = typ
        node.name      = self.name
        return node
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
    err_node.node_type = L.error
    return err_node
end

function ast.Number:check(ctxt)
    local number = self:clone()
    number.value = self.value
    if tonumber(self.value) % 1 == 0 then
        self.node_type = L.int
        number.node_type = L.int
    else
        self.node_type = L.double
        number.node_type = L.double
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

    if type_so_far == L.error then return err(self, ctxt) end
    if not type_so_far:isPrimitive() then return err(self, ctxt, tp_error) end

    for i = 2, #self.elems do
        veclit.elems[i] = self.elems[i]:check(ctxt)
        local tp        = veclit.elems[i].node_type

        if not tp:isPrimitive() then return err(self, ctxt, tp_error) end

        type_so_far = T.type_meet(type_so_far, tp)
        if type_so_far:isError() then return err(self, ctxt, mt_error) end
    end

    veclit.node_type = L.vector(type_so_far, #veclit.elems)
    return veclit
end

function ast.Bool:check(ctxt)
    local boolnode = self:clone()
    boolnode.value = self.value
    boolnode.node_type = L.bool
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
        errnode.node_type = L.error
        return errnode
    end
    local arg_ast = self.children[1]:check(ctxt)
    local argtype = arg_ast.node_type
    assert(argtype ~= nil)
    self.node_type = argtype
    return arg_ast
end


local function RunMacro(ctxt,src_node,v,params)
  local r = v.genfunc(unpack(params))
  return luav_to_checked_ast(r, src_node, ctxt)
end
function ast.TableLookup:check(ctxt)
    local table     = self.table:check(ctxt)
    local member    = self.member
    local ttype     = table.node_type

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
            local ast_node      = ast.FieldAccess:DeriveFrom(member)
            ast_node.row        = table
            ast_node.field      = field
            ast_node.node_type  = field.type
            return ast_node

        -- desugar macro-fields from row.macro to macro(row)
        elseif L.is_macro(luaval) then
            return RunMacro(ctxt,self,luaval,{table})
        else
            return err(self, ctxt, "Row "..table.name.." does not "..
                                   "have field or macro-field "..
                                   "'"..member.."'")
        end
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
    local vectype, idxtype = vec.node_type, idx.node_type

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
    local call     = self:clone()
    
    call.node_type = L.error -- default
    local func      = self.func:check(ctxt)
    call.params = self.params:check(ctxt)
        
    local isinternal = func.node_type:isInternal()
    local v = isinternal and func.node_type.value
    if v and L.is_function(v) then
        call.func = v
        call.node_type = v.check(call, ctxt)
    elseif v and L.is_macro(v) then
        -- replace the call node with the inlined AST
        call = RunMacro(ctxt, self, v, call.params.children)
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
    return self.ast
end

function ast.FieldAccess:check(ctxt)
    local n     = self:clone()
    n.field     = self.field
    n.row       = self.row
    return n
end

function ast.LuaObject:check(ctxt)
    assert(self.node_type and self.node_type:isInternal())
    return self
end
function ast.Where:check(ctxt)
    --note: where is generated in a macro, so its fields are already type-checked
    local fieldobj = self.field.node_type
    local keytype = self.key.node_type
    if not fieldobj:isInternal() or not L.is_field(fieldobj.value) then
        ctxt:error(self,"Expected a field as the first argument but found ",fieldobj)
    end
    local field = fieldobj.value
    if keytype ~= field.type then
        ctxt:error(self,"Key of where is type ",keytype," but expected type ",field.type)
    end
    if field.owner._index ~= field then
        ctxt:error(self,"Field ",field.name, " is not an index of ",field.owner:Name())
    end
    local w = self:clone()
    w.relation = field.owner
    w.key = self.key
    w.node_type = L.set(w.relation)
    return w
end

------------------------------------------------------------------------------
--[[ Semantic checking called from here:                                  ]]--
------------------------------------------------------------------------------
function ast.LisztKernel:check(ctxt)
    local kernel            = self:clone()
    local set = self.set:check(ctxt)
    if not set.node_type:isInternal() or not L.is_relation(set.node_type.value) then
        ctxt:error(kernel.set, "Expected a relation")
    else 
        kernel.name      = self.name
        kernel.relation  = set.node_type.value
        kernel.node_type = L.row(kernel.relation)
        ctxt:liszt()[kernel.name] = kernel.node_type
        kernel.body = self.body:check(ctxt)
    end

    return kernel
end

function S.check(luaenv, kernel_ast, param_type)

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
