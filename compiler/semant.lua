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

local Context = {}
Context.__index = Context

function Context.new(env, diag)
    local ctxt = setmetatable({
        env         = env,
        diag        = diag,
        loop_count  = 0,
        centers     = {}, -- variable symbols bound to the centered row
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

function check_reduce(node, ctxt)
    local op        = node.reduceop
    local lvalue    = node.lvalue
    local ltype     = lvalue.node_type

    --if ltype:baseType() ~= L.float then
    --    ctxt:error(node, 'Reduce operator"'..op..'" for type '..
    --        '"'..tostring(ltype)..'" is not currently supported.')
    --end
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
    if node.lvalue:is(ast.Global) and not self.reduceop then
        ctxt:error(self.lvalue, "Cannot write to globals in kernels")
        node.lvalue.is_lvalue = true
    end

    -- enforce that the lhs is an lvalue
    if not node.lvalue.is_lvalue then
        ctxt:error(self.lvalue, "Illegal assignment: left hand side cannot "..
                                "be assigned")
        return node
    elseif node.lvalue.node_type:isRow() and
           not node.lvalue:is(ast.FieldAccess)
    then
        ctxt:error(self.lvalue, "Illegal assignment: variables of row type "..
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
    if node.lvalue:is(ast.Global) then
        check_reduce(node, ctxt)
        local gr = ast.GlobalReduce:DeriveFrom(node)
        gr.global      = node.lvalue
        gr.exp         = node.exp
        gr.reduceop    = node.reduceop
        return gr
    -- replace assignment with a field write if we see a
    -- field access on the left hand side
    elseif node.lvalue:is(ast.FieldAccess) then
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
        -- if the rhs is a centered row, try to propagate that information
        -- NOTE: this pseudo-constant propagation is strong b/c
        -- we don't allow re-assignment of row-type variables
        if exptyp:isRow() and decl.initializer.is_centered then
            ctxt:recordcenter(decl.name)
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

    delete.row   = self.row:check(ctxt)
    local rowtyp = delete.row.node_type

    if not rowtyp:isRow() or not delete.row.is_centered then
        ctxt:error(self,"Only centered rows may be deleted")
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

    local coarse_check = ltype:isValueType() and rtype:isValueType()
    if is_eq_compare[binop.op] then
        coarse_check = ltype:isFieldType() and rtype:isFieldType()
    end
    if not coarse_check then
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

    -- We will get here if the user attempts to use the min/max operators inline.
    -- Currently they are only supported as reductions.
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

function ast.Bool:check(ctxt)
    local boolnode     = self:clone()
    boolnode.node_type = L.bool
    return boolnode
end

function ast.VectorLiteral:check(ctxt)
    local veclit      = self:clone()
    veclit.elems      = {}

    local tp_error = "vector literals can only contain values "..
                     "of boolean or numeric type"
    local mt_error = "vector entries must be of the same type"

    veclit.elems[1]   = self.elems[1]:check(ctxt)
    local max_type    = veclit.elems[1].node_type
    if max_type == L.error then return err(self, ctxt) end
    if not max_type:isPrimitive() then
        return err(self, ctxt, tp_error) 
    end

    -- scan the remaining entries to compute a max type
    for i = 2, #self.elems do
        veclit.elems[i] = self.elems[i]:check(ctxt)
        local tp        = veclit.elems[i].node_type
        if tp == L.error then return err(self, ctxt) end
        if not tp:isPrimitive() then
            return err(self,ctxt, tp_error)
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
    local quoted_params = QuoteParams(params)
    local result = the_macro.genfunc(unpack(quoted_params))

    if ast.is_ast(result) and result:is(ast.Quote) then
        return result
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

    if ttype:isRow() then
        local luaval = ttype.relation[member]

        -- create a field access normally
        if L.is_field(luaval) then
            local field         = luaval
            local ast_node      = ast.FieldAccess:DeriveFrom(tab)
            ast_node.name       = member
            ast_node.row        = tab
            local name = ast_node.row.name
            if name and ctxt:iscenter(name) then
                ast_node.row.is_centered = true
            end
            ast_node.field      = field
            ast_node.node_type  = field.type
            return ast_node

        -- desugar macro-fields from row.macro to macro(row)
        elseif L.is_macro(luaval) then
            return RunMacro(ctxt,self,luaval,{tab})
        -- desugar function-fields from row.func to func(row)
        elseif L.is_user_func(luaval) then
            return InlineUserFunc(ctxt,self,luaval,{tab})
        else
            return err(self, ctxt, "Row "..ttype.relation:Name()..
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

    for i,p in ipairs(self.params) do
        call.params[i] = p:check(ctxt)
    end

    local v = func.node_type:isInternal() and func.node_type.value
    if v and L.is_builtin(v) then
        call.func      = v
        call.node_type = v.check(call, ctxt)
    elseif v and L.is_macro(v) then
        -- replace the call node with the inlined AST
        call = RunMacro(ctxt, self, v, call.params)
    elseif v and L.is_user_func(v) then
        call = InlineUserFunc(ctxt, self, v, call.params)
    elseif v and T.isLisztType(v) and v:isValueType() then
        local params = call.params
        if #params ~= 1 then
            ctxt:error(self, "Cast to " .. v.toString() ..
                    " expects exactly 1 argument (instead got " .. #params ..
                    ")")
        else
            -- TODO: We should have more aggresive casting protection.
            -- i.e. we should allow anything reasonable in Terra/C but
            -- no more.
            local pretype = params[1].node_type
            local casttype = v
            local one_vec = pretype:isVector() or casttype:isVector()
            local matching_vecs = pretype:isVector() and casttype:isVector()
                              and pretype.N == casttype.N
            if one_vec and not matching_vecs then
                ctxt:error(self, "Can only cast vectors to "..
                    "other vectors with matching dimensions")
            else
                call = ast.Cast:DeriveFrom(self)
                call.value     = params[1]
                call.node_type = v
            end
        end
    -- __apply_macro  i.e.  c(1,0)  for offsetting in a grid
    elseif func.node_type:isRow() then
        local apply_macro = func.node_type.relation.__apply_macro
        local params = {func}
        for _,v in ipairs(call.params) do table.insert(params, v) end
        call = RunMacro(ctxt, self, apply_macro, params)

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
    fa.row = self.row:check(ctxt)
    return fa
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
    if not field.owner._grouping or
       field.owner._grouping.key_field ~= field
    then
        ctxt:error(self,"Relation '"..field.owner:Name().."' is not "..
                        "grouped by Field '"..field.name.."'")
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
        local row_type              = L.row(kernel.relation)
        -- record the center
        ctxt:recordcenter(kernel.name)
        ctxt:liszt()[kernel.name]   = row_type
        kernel.body                 = self.body:check(ctxt)
    else
        ctxt:error(kernel.set, "Expected a relation")
    end

    return kernel
end

function S.check(luaenv, kernel_ast)
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
