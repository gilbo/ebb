local exports   = {}

ast = require("ast")
local types = terralib.require("compiler/types")
local tutil     = types.usertypes
local Type      = types.Type
local t         = types.t
local type_meet = types.type_meet
--local Scope     = types.Scope

-- Phases used in assignment statements
_FIELD_WRITE   = 'FIELD_WRITE'
_FIELD_REDUCE  = 'FIELD_REDUCE'
_SCALAR_REDUCE = 'SCALAR_REDUCE'
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
--[[ Context Definition   														                    ]]--
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
		env = env,
		diag = diag,
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


------------------------------------------------------------------------------
--[[ Small Helper Functions														                    ]]--
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

	whilestmt.cond 			 = self.cond:check(ctxt)
	local condtype       = whilestmt.cond.node_type
	if condtype ~= t.error and condtype ~= t.bool then
		ctxt:error(self, "Expected bool expression but found " ..
										 condtype:toString())
	end

	ctxt:enterblock()
	whilestmt.body = self.body:check(ctxt)
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

	repeatstmt.cond = self.cond:check(ctxt)
	local condtype  = repeatstmt.cond.node_type
	repeatstmt.body = self.body:check(ctxt)

	if condtype ~= t.error and condtype ~= t.bool then
		ctxt:error(self, "Expected bool expression but found " ..
										 condtype:toString())
	end

	ctxt:leaveblock()
	return repeatstmt
end

function ast.ExprStatement:check(ctxt)
	local expstmt = self:clone()
	expstmt.exp   = self.exp:check(ctxt)
	return expstmt
end

function ast.Reduce:check(ctxt)
	-- SHOULD CHECK whether or not this obj can be reduced...
	local exp    = self.exp:check_lhs(ctxt)

	-- A scalar can only be a lvalue when reduced
	if exp:is(ast.Scalar) then
		exp.is_lvalue = true
	end
	-- only lvalues can be "reduced"
	if exp.is_lvalue then
		return exp
	else
		ctxt:error(self, "Only lvalues can be reduced.")
		local errnode = self:clone()
		errnode.node_type = t.error
		return errnode
	end
end

function ast.Assignment:check(ctxt)
	local assignment = self:clone()

	local lhs         = self.lvalue:check_lhs(ctxt)
	assignment.lvalue = lhs
	local ltype       = lhs.node_type
	if ltype == t.error then return assignment end

	local rhs         = self.exp:check(ctxt)
	assignment.exp    = rhs
	local rtype       = rhs.node_type
	if rtype == t.error then return assignment end

	-- handle assignments that are reductions
	if self.lvalue:is(ast.Reduce) then
		assignment.reduceop = self.lvalue.op
		if rtype:isLogical() then
			assignment.reduceop = "l"..assignment.reduceop
		end
	end

	-- is this the first use of a declared, but not defined, variable?
	if ltype == t.unknown then
		ltype                            = rtype
		lhs.node_type                    = rtype
		ctxt:liszt()[lhs.name].node_type = rtype
	end

	-- enforce lvalues
	if not lhs.is_lvalue then
		-- TODO: less cryptic error messages in this case
		-- 			 Better error messages probably involes switching on kind of lhs
		ctxt:error(lhs, "The left side of an assignment must be an lvalue")
		return assignment
	end

	-- make sure the types on both sides agree
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
	local initstmt = self:clone()
	initstmt.ref   = clone_name(self.ref)
	initstmt.exp   = self.exp:check(ctxt)
	local typ      = initstmt.exp.node_type

	if typ == t.error then return end

	-- Local temporaries can only refer to numeric/boolean/vector data or
	-- topolocal element types, and variables referring to topo types can
	-- never be re-assigned.  That way, we don't allow users to write
	-- code that makes stencil analysis intractable.
	if not typ:isLogical() and not typ:isNumeric() and not typ:isTopo() then
		ctxt:error(self,"Can only assign numbers, bools, "..
										"or topological elements to local temporaries")
	end

	initstmt.ref.node_type           = typ
	ctxt:liszt()[initstmt.ref.name]  = initstmt.exp

	return initstmt
end

function ast.DeclStatement:check(ctxt)
	local decl 			= self:clone()
	decl.ref 			  = clone_name(self.ref)
	decl.ref.node_type = t.unknown
	ctxt:liszt()[decl.ref.name] = decl.ref
	return decl
end

function ast.AssertStatement:check(ctxt)
	local assertstmt  = self:clone()
	assertstmt.test   = self.test:check(ctxt)
	local ttype       = assertstmt.test.node_type

	if ttype ~= t.error and ttype ~= t.bool then
		ctxt:error(self, "Expected a boolean as the test for assert statement")
	end

	return assertstmt
end

function ast.PrintStatement:check(ctxt)
	local printstmt    = self:clone()
	printstmt.output   = self.output:check(ctxt)
	local otype        = printstmt.output.node_type

	if otype ~= t.error and not (otype:isNumeric() or otype:isLogical()) then
		ctxt:error(self, "Only numbers, bools, and vectors can be printed")
	end

	return printstmt
end

function ast.NumericFor:check(ctxt)
	local node = self
	local function check_num_type(tp)
		if tp ~= t.error and not tp:isNumeric() then
			ctxt:error(node, "Expected a numeric expression to define the "..
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
		local varname = numfor.iter.name
		ctxt:liszt()[varname] = numfor.iter
		numfor.body = self.body:check(ctxt)
	ctxt:leaveblock()

	return numfor
end

function ast.GenericFor:check(ctxt)
	ctxt:error(self, "Generic for statement not yet implemented", 2)
end

function ast.Break:check(ctxt)
	-- TODO: Break should check whether or not it is contained
	-- within a loop somehow.  If not, something should raise an error
	return self:clone()
end

function ast.CondBlock:check(ctxt)
	local new_node  = self:clone()
	new_node.cond   = self.cond:check(ctxt)
	local condtype  = new_node.cond.node_type
	if condtype ~= t.error and condtype ~= t.bool then
		ctxt:error(self, "Conditional expression type should be"..
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

-- binary expressions
function ast.BinaryOp:check(ctxt)
	local binop = self:clone()
	binop.op 		= self.op
	binop.lhs   = self.lhs:check(ctxt)
	binop.rhs   = self.rhs:check(ctxt)
	local ltype, rtype = binop.lhs.node_type, binop.rhs.node_type
	binop.node_type = t.error

	-- factors out common code
	local node  = self
	local function err(msg)
		if msg then ctxt:error(node, msg) end
		return binop
	end

	-- Silently ignore/propagate errors
	if ltype == t.error or rtype == t.error then return err() end

	local type_err = "incompatible types: " .. ltype:toString() ..
									 ' and ' .. rtype:toString()
	local op_err  = "invalid types for operator " .. binop.op .. ': ' ..
									ltype:toString() .. ' and ' .. rtype:toString()

	if not ltype:isExpressionType() or not rtype:isExpressionType() then
		return err(type_err)
	end

	local derived = type_meet(ltype, rtype)

	if isNumOp[binop.op] then
		if ltype:isVector() or rtype:isVector() then return err(op_err) end
	elseif isCompOp[binop.op] then
		if ltype:isLogical() ~= rtype:isLogical() then return err(op_err)

		-- if the type_meet failed, types are incompatible
		elseif derived == t.error then return err(type_err)

		-- otherwise, we need to return a logical type
		elseif derived:isPrimitive() then
			binop.node_type = t.bool
			return binop

		else
			binop.node_type = t.vector(t.bool,derived.N)
			return binop
		end

	elseif isBoolOp[binop.op] then
		if not ltype:isLogical() or not rtype:isLogical() then
			return err(op_err)
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
	unop.op 	     = self.op
	unop.exp       = self.exp:check(ctxt)
	local exptype  = unop.exp.node_type
	unop.node_type = t.error -- default

	if unop.op == 'not' then
		if exptype:isLogical() then
			unop.node_type = exptype
		else
			ctxt:error(self, "Unary \"not\" expects a boolean operand")
		end
	elseif unop.op == '-' then
		if exptype:isNumeric() then
			unop.node_type = exptype
		else
			ctxt:error(self, "Unary minus expects a numeric operand")
		end
	else
		ctxt:error(self, "Unknown unary operator \'".. self.op .."\'")
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

	-- If luav is a table, it may be a Liszt object like a Relation or Field
	if type(luav) == 'table' then
		-- Scalar objects are replaced with special Scalar nodes
		if Type.isScalar(luav) then
			node 			  = ast.Scalar:DeriveFrom(src_node)
			node.scalar = luav

		-- Vector objects are expanded into literal AST trees
		elseif Type.isVector(luav) then
			node   			= ast.VectorLiteral:DeriveFrom(src_node)
			node.elems  = {}
			for i,v in ipairs(luav.data) do
				node.elems[i] = luav_to_ast(v, src_node)
			end

		-- Field objects are replaced with special Field nodes
		elseif Type.isField(luav) then
			node  		  = ast.Field:DeriveFrom(src_node)
			node.field  = luav

		-- failing all else, we'll assume this lua table is being used
		-- as a namespace, so we need to return it for further expansion
		else
			return luav
		end

	elseif type(luav) == 'number' then
		node    = ast.Number:DeriveFrom(src_node)
		node.value    = luav

	elseif type(luav) == 'boolean' then
		node    = ast.Bool:DeriveFrom(src_node)
		node.value    = luav

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
local function luav_to_checked_ast(luav, src_node, ctxt, allow_lua_return)
	-- convert the lua value into an ast node
	local ast_node = luav_to_ast(luav, src_node)

	-- on conversion error
	if not ast_node then
		ctxt:error(src_node, "Could not convert Lua value to a Liszt value")
		ast_node = src_node:clone()
		ast_node.node_type = t.error

	-- on conversion to a namespace, not value
	elseif not ast_node.is_liszt_ast then
		if not allow_lua_return then
			ctxt:error(src_node, "Cannot use lua tables as Liszt values.")
			ast_node = src_node:clone()
			ast_node.node_type = t.error
		else
			return ast_node -- is actually a lua table here
		end

	-- on successful conversion to an ast node
	else
		ast_node = ast_node:check(ctxt)
	end

	return ast_node
end

-- TODO: remove LValue type from the AST hierarchy...
function ast.LValue:check_lhs(ctxt)
	return self:check(ctxt)
end

function ast.Name:check_lhs(ctxt)
	return self:check_maybe_lua(ctxt, false, true)
end
function ast.Name:check(ctxt)
	return self:check_maybe_lua(ctxt, false, false)
end

function ast.Name:check_maybe_lua(ctxt, allow_lua_return, assign)
	-- try to find the name in the local Liszt scope
	local lisztv = ctxt:liszt()[self.name]
	if lisztv then
		local new_node = self:clone()
		new_node.name  = self.name
		local typ      = lisztv.node_type
		new_node.node_type = typ

		-- check that we're not reading from an uninitialized variable
		if typ == t.unknown and not assign then
			ctxt:error(self, "Variable '" .. self.name .. "' is not initialized")
			new_node.node_type = t.error
		end

		-- check for conditions where the name can't be an lvalue
		if not (typ and typ:isTopo()) then
			new_node.is_lvalue = true
		end

		return new_node
	end

	-- Otherwise, does the name exist in the lua scope?
	local luav = ctxt:lua()[self.name]
	if luav then
		-- convert the lua value into an ast node
		local ast_node =
			luav_to_checked_ast(luav, self, ctxt, allow_lua_return)

		-- track the name this came from for debuging convenience
		if ast_node.is_liszt_ast and 
			 (ast_node:is(ast.Field) or ast_node:is(ast.Scalar))
		then
			ast_node.name = self.name
		end
		
		return ast_node
	end

	-- failed to find this name anywhere
	ctxt:error(self, "Variable '" .. self.name .. "' is not defined")
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

	local tp_error = "Vector literals can only contain literals "..
									 "of boolean or numeric type"
	local mt_error = "Vector entries must be of the same type"
	local node  = self
	local function err(msg)
		if msg then ctxt:error(node, msg) end
		veclit.node_type = t.error
		return veclit
	end

	if type_so_far == t.error then return err() end
	if not type_so_far:isPrimitive() then return err(tp_error) end

	for i = 2, #self.elems do
		veclit.elems[i] = self.elems[i]:check(ctxt)
		local tp 				= veclit.elems[i].node_type

		if not tp:isPrimitive() then return err(tp_error) end

		type_so_far = type_meet(type_so_far, tp)
		if not type_so_far then return err(mt_error) end
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
		ctxt:error(self, "Can use exactly one argument to index here")
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
	return self:check_maybe_lua(ctxt, false, false)
end
function ast.TableLookup:check_lhs(ctxt)
	return self:check_maybe_lua(ctxt, false, true)
end

function ast.TableLookup:check_maybe_lua(ctxt, allow_lua_return, assign)
	local table   = self.table:check_maybe_lua(ctxt, true, assign)
	local member  = self.member
	local ttype   = table.node_type

	local node 		= self
	local function err(msg)
		if msg then ctxt:error(node, msg) end
		local errnode = node:clone()
		errnode.node_type = t.error
		return errnode
	end

	if ttype and ttype == t.error then
		return err()
	elseif table.is_liszt_ast then
		-- we did not get back a lua table on the left.
		return err("select operator not supported for non-table type " ..
							 ttype:toString())
	end
	-- otherwise, we have a lua table in lookup.table
	-- so we can perform the lookup in the lua table as namespace
	local luaval = table[member.name]
	if luaval == nil then
		return err("lua table does not have member " .. member)
	end

	-- and then go ahead and convert the lua value into an ast node
	local ast_node = luav_to_checked_ast(luaval, self, ctxt, allow_lua_return)

	-- we set a name somewhat appropriately (could capture namespace...)
	if ast_node.is_liszt_ast and 
		 (ast_node:is(ast.Field) or ast_node:is(ast.Scalar))
	then
		ast_node.name = lookup.member.name
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
		ctxt:error(self, "Expected an integer expression to index into "..
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
	-- call name can be a field only in current implementation
	call.func   = self.func:check(ctxt)
	local ftype = call.func.node_type

	if ftype == t.error then
		-- error fall through
	elseif ftype:isField() then
		local ftopo = ftype:topoType()
		local fdata = ftype:dataType()

		local arg_ast  = self.params:index_check(ctxt)
		local argtype  = arg_ast.node_type
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

	else
		ctxt:error(self, "Invalid call")
	end

	return call
end

function ast.Scalar:check(ctxt)
	local new_node  = self:clone()
	local scalar    = self.scalar
	new_node.scalar = scalar
	if self.name then new_node.name = self.name end

	new_node.node_type = scalar.type

	return new_node
end

function ast.Field:check(ctxt)
	local new_node  = self:clone()
	local field 		= self.field
	new_node.field  = field
	if self.name then new_node.name = self.name end

	new_node.node_type = t.field(field.topo, field.type)

	return new_node
end



------------------------------------------------------------------------------
--[[ Semantic checking called from here:                                  ]]--
------------------------------------------------------------------------------
local function check_kernel_param(param, ctxt)
	local new_param 			= clone_name(param)
	new_param.node_type		= t.topo

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


