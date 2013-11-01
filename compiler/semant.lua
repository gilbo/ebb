local exports = {}

local ast   = require("ast")
local types = terralib.require("compiler/types")
local tutil     = types.usertypes
local Type      = types.Type
local t         = types.t
local type_meet = types.type_meet
--local Scope     = types.Scope

-- Phases used in assignment statements
local _FIELD_WRITE   = 'FIELD_WRITE'
local _FIELD_REDUCE  = 'FIELD_REDUCE'
local _SCALAR_REDUCE = 'SCALAR_REDUCE'
exports._FIELD_WRITE 		= _FIELD_WRITE
exports._FIELD_REDUCE 	= _FIELD_REDUCE
exports._SCALAR_REDUCE 	= _SCALAR_REDUCE

--[[
	AST:check(ctxt) type checking routines
		These methods define type checking.
		Each check() method is designed to accept a typing context (see below).
		Once called, check() will construct and return a new type-checked
		version of its subtree (new AST nodes)
]]--


------------------------------------------------------------------------------
--[[ Context Definition   	                                              ]]--
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
		env        = env,
		diag       = diag,
		lhs_count  = 0,
		loop_count = 0,
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
function Context:in_lhs()
	return self.lhs_count > 0
end
function Context:enterlhs()
	self.lhs_count = self.lhs_count + 1
end
function Context:leavelhs()
	self.lhs_count = self.lhs_count - 1
	if self.lhs_count < 0 then self.lhs_count = 0 end
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
--[[ Small Helper Functions                                               ]]--
------------------------------------------------------------------------------
local function clone_name(name_node)
	local copy = name_node:clone()
	copy.name = name_node.name
	return copy
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
		ctxt:error(self, "expected bool expression but found " .. condtype:toString())
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

	ctxt:enterlhs()
	local lhs = self.lvalue:check(ctxt)
	ctxt:leavelhs()


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
		if rtype:isLogical() then
			assignment.reduceop = "l"..assignment.reduceop
		end
	end

	-- When the left hand side is declared, but uninitialized, set the type
	if ltype == t.unknown then
		ltype                            = rtype
		lhs.node_type                    = rtype
		ctxt:liszt()[lhs.name].node_type = rtype
	end

	-- enforce that the lhs is an lvalue
	if not lhs.is_lvalue then
		-- TODO: less cryptic error messages in this case
		-- 			 Better error messages probably involes switching on kind of lhs
		ctxt:error(lhs, "assignments in a Liszt kernel are only valid to indexed fields or kernel variables")
		return assignment
	end

	-- enforce type agreement b/w lhs and rhs
	local derived = type_meet(ltype,rtype)
	if derived == t.error or
	   (ltype:isPrimitive() and rtype:isVector()) or
	   (ltype:isVector()    and rtype:isVector() and ltype.N ~= rtype.N) then
		ctxt:error(self, "invalid conversion from " .. rtype:toString() ..
										 ' to ' .. ltype:toString())
		return
	end

	return assignment
end

function ast.InitStatement:check(ctxt)
	local initstmt  = self:clone()
	initstmt.ref    = clone_name(self.ref)
	initstmt.exp    = self.exp:check(ctxt)
	local typ       = initstmt.exp.node_type

	if typ == t.error then return end

	-- Local temporaries can only refer to numeric/boolean/vector data or
	-- topolocal element types, and variables referring to topo types can
	-- never be re-assigned.  That way, we don't allow users to write
	-- code that makes stencil analysis intractable.
	if not typ:isLogical() and not typ:isNumeric() and not typ:isTopo() then
		ctxt:error(self,"can only assign numbers, bools, " ..
		                "or topological elements to local temporaries")
	end

	initstmt.ref.node_type          = typ
	ctxt:liszt()[initstmt.ref.name] = initstmt.exp

	return initstmt
end

function ast.DeclStatement:check(ctxt)
	local decl 	        = self:clone()
	decl.ref            = clone_name(self.ref)
	decl.ref.node_type  = t.unknown
	ctxt:liszt()[decl.ref.name] = decl.ref
	return decl
end

function ast.NumericFor:check(ctxt)
	local node = self
	local function check_num_type(tp)
		if tp ~= t.error and not tp:isNumeric() then
			ctxt:error(node, "expected a numeric expression to define the "..
											 "iterator bounds/step (found "..tp:toString()..")")
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
	numfor.iter           = clone_name(self.iter)
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
		if ltype:isLogical() ~= rtype:isLogical() then return err(binop, ctxt, op_err)

		-- if the type_meet failed, types are incompatible
		elseif derived == t.error then return err(binop, ctxt, type_err)

		-- otherwise, we need to return a logical type
		else
			binop.node_type = derived:isPrimitive() and t.bool or t.vector(t.bool, derived.N)
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

	-- Field objects are replaced with special Field nodes
	elseif Type.isField(luav) then
		node        = ast.Field:DeriveFrom(src_node)
		node.field  = luav

    elseif Type.isFunction(luav) then
		node      = ast.Function:DeriveFrom(src_node)
		node.func = luav

    elseif terralib.isfunction(luav) then
        node      = ast.Function:DeriveFrom(src_node)
        node.func = builtins.terra_to_macro(luav)

	elseif type(luav) == 'table' then
		node       = ast.Table:DeriveFrom(src_node)
		node.table = luav
		return node

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
		local new_node     = self:clone()
		new_node.name      = self.name
		local typ          = lisztv.node_type
		new_node.node_type = typ

		-- check that we're not reading from an uninitialized variable
		if typ == t.unknown and not ctxt:in_lhs() then
			ctxt:error(self, "variable '" .. self.name .. "' is not initialized")
			new_node.node_type = t.error
		end

		-- check for conditions where the name can't be an lvalue
		-- setting this flag prevents variables that refer to topological types from being reassigned
		new_node.is_lvalue = not typ:isTopo()

		return new_node
	end

	-- Otherwise, does the name exist in the lua scope?
	local luav = ctxt:lua()[self.name]
	if luav then
		-- convert the lua value into an ast node
		local ast_node = luav_to_checked_ast(luav, self, ctxt)

		-- track the name this came from for debuging convenience
		if ast_node:is(ast.Field) or ast_node:is(ast.Scalar) or ast_node:is(ast.Table) then
			ast_node.name = self.name
		end
		
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
		self.node_type = t.float
		number.node_type = t.float
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
	--if argtype == nil then
	--	return nil
	--end
	self.node_type = argtype
	return arg_ast
end

function ast.TableLookup:check(ctxt)
	local table  = self.table:check(ctxt)
	local member = self.member
	local ttype  = table.node_type

	if ttype == t.error then
		return err(self, ctxt)
	elseif table:is(ast.Scalar) then
		return err(self, ctxt, "select operator not supported for non-table type L.Scalar")
	elseif table:is(ast.Field) then
		return err(self, ctxt, "select operator not supported for non-table type L.Field")
	elseif not table:is(ast.Table) then
		return err(self, ctxt, "select operator not supported for non-table type " .. ttype:toString())
	end

	self.name = table.name .. '.' .. member.name

	-- otherwise, we have a lua table in lookup.table
	-- so we can perform the lookup in the lua table as namespace
	local luaval = table.table[member.name]
	if luaval == nil then
		return err(self, ctxt, "lua table " .. table.name .. " does not have member '" .. member.name .. "'")
	end

	-- and then go ahead and convert the lua value into an ast node
	local ast_node = luav_to_checked_ast(luaval, self, ctxt)

	-- we set a name somewhat appropriately (could capture namespace...)
	if ast_node:is(ast.Field) or ast_node:is(ast.Scalar) or ast_node:is(ast.Table) then
		ast_node.name = table.name .. '.' .. member.name
	end

	return ast_node
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
	local call = self:clone()
	call.node_type = t.error -- default
	call.func   = self.func:check(ctxt)
	local ftype = call.func.node_type

	if call.func:is(ast.Function) then
        call.params    = self.params:check(ctxt)
        call.node_type = call.func.func.check(call, ctxt)

	elseif call.func:is(ast.Field) then
		local ftopo = ftype:topoType()
		local fdata = ftype:dataType()

		local arg_ast = self.params:index_check(ctxt)
		local argtype = arg_ast.node_type
		if (argtype == t.error) then
			-- error fall through

		elseif argtype:isTopo() then
			-- handle as yet un-typed topo variables
			if argtype == t.topo then
				-- set type for topological object
				ctxt:liszt()[arg_ast.name].node_type = ftopo
				argtype = ftopo
			end

			-- typecheck that the field is present on this kind of topo
			if argtype == ftopo then
				local access     = ast.FieldAccess:DeriveFrom(self)
				access.field     = call.func.field
				access.topo      = arg_ast
				access.node_type = fdata
				access.is_lvalue = true -- can be an lvalue
				return access
			else
				ctxt:error(self, ftype:toString() .. " indexed by incorrect type " ..
												 argtype:toString())
			end

		else
			ctxt:error(self, ftype:toString() .. " indexed by incorrect type " ..
											 argtype:toString())
		end

	elseif ftype:isError() then
		-- fall through (do not print error messages for errors already reported)

    else
		ctxt:error(self, "invalid call")

	end

	return call
end

function ast.Scalar:check(ctxt)
	local new_node     = self:clone()
	new_node.scalar    = self.scalar
	new_node.name      = self.name
	new_node.node_type = self.scalar.type
	return new_node
end

function ast.Field:check(ctxt)
	local new_node     = self:clone()
	new_node.field     = self.field
	new_node.name      = self.name
	new_node.node_type = t.field(self.field.topo, self.field.type)
	return new_node
end

function ast.Table:check(ctxt)
	local new_node     = self:clone()
	new_node.table     = self.table
	new_node.name      = self.name
	new_node.node_type = t.table
	return new_node
end

function ast.Function:check(ctxt)
	local new_node     = self:clone()
	new_node.func      = self.func
	new_node.node_type = self.node_type
	new_node.name      = self.name
	return new_node
end

------------------------------------------------------------------------------
--[[ Semantic checking called from here:                                  ]]--
------------------------------------------------------------------------------
local function check_kernel_param(param, ctxt)
	local new_param     = clone_name(param)
	new_param.node_type	= t.topo

	ctxt:liszt()[param.name] = new_param

	return new_param
end

function exports.check(luaenv, kernel_ast)

	-- environment for checking variables and scopes
	local env  = terralib.newenvironment(luaenv)
	local diag = terralib.newdiagnostics()
	local ctxt = Context.new(env, diag)

	--------------------------------------------------------------------------

	local new_kernel_ast = kernel_ast:clone()

	diag:begin()
	env:enterblock()

	new_kernel_ast.param = check_kernel_param(kernel_ast.param, ctxt)
	new_kernel_ast.body  = kernel_ast.body:check(ctxt)

	env:leaveblock()
	diag:finishandabortiferrors("Errors during typechecking liszt", 1)

	return new_kernel_ast
end


return exports
