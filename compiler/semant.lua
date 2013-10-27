local exports   = {}

ast = require("ast")
local types = terralib.require("compiler/types")
local tutil     = types.usertypes
local Type      = types.Type
local t         = types.t
local type_meet = types.type_meet
local Scope     = types.Scope

-- Phases used in assignment statements
_FIELD_WRITE   = 'FIELD_WRITE'
_FIELD_REDUCE  = 'FIELD_REDUCE'
_SCALAR_REDUCE = 'SCALAR_REDUCE'
exports._FIELD_WRITE 		= _FIELD_WRITE
exports._FIELD_REDUCE 	= _FIELD_REDUCE
exports._SCALAR_REDUCE 	= _SCALAR_REDUCE


--[[
	AST:check(env, diag) type checking routines
		These methods define type checking.
		Each check() method is designed to accept a typing environment &
	a diagnostic object.  Once called, it will construct a new type-checked
	version of its subtree (all new AST nodes) and return that subtree
	along with its computed type (or nil if a statement etc.)
		If only the new AST is desired, the second return value can be ignored.
		If only the type is desired, the first value can be ignored as
			_, type = abc:check(env, diag)

]]--



------------------------------------------------------------------------------
--[[ Small Helper Functions														                    ]]--
------------------------------------------------------------------------------
local function clone_name(name_node)
	local copy = name_node:clone()
	copy.name = name_node.name
	return copy
end

------------------------------------------------------------------------------
--[[ Stand-in for the luaval of an indexed global field                   ]]--
------------------------------------------------------------------------------
local FieldIndex = { kind = Type.kinds.fieldindex}
FieldIndex.__index = FieldIndex

function FieldIndex.New(field, objtype)
	return setmetatable({field=field,type=objtype}, FieldIndex)
end


------------------------------------------------------------------------------
--[[ BinaryOp Reduction detection                                         ]]--
------------------------------------------------------------------------------
local commutative = {
	['+']   = true,
	['-']   = true,
	['*']   = true,
	['/']   = true,
	['and'] = true,
	['or']  = true
}

function ast.BinaryOp:operatorCommutes ()
	return commutative[self.op]
end


------------------------------------------------------------------------------
--[[ Scoping methods for LValues                                          ]]--
------------------------------------------------------------------------------
function ast.LValue:setLocalScope()
	self.scope = Scope.liszt
end

function ast.LValue:setGlobalScope()
	self.scope = Scope.lua
end

function ast.LValue:isLocalScope()
	return self.scope == Scope.liszt
end

function ast.LValue:isGlobalScope()
	return self.scope == Scope.lua
end


------------------------------------------------------------------------------
--[[ AST semantic checking methods:                                       ]]--
------------------------------------------------------------------------------
function ast.AST:check(env, diag) -- diag(nostic)
	error("Typechecking not implemented for AST node " .. self.kind)
end

function ast.Block:check(env, diag)
	local block = self:clone()

	-- statements
	block.statements = {}
	for id, node in ipairs(self.statements) do
		block.statements[id] = node:check(env, diag)
	end

	return block
end

function ast.IfStatement:check(env, diag)
	local ifstmt = self:clone()

	ifstmt.if_blocks = {}
	for id, node in ipairs(self.if_blocks) do
		env:enterblock()
		ifstmt.if_blocks[id] = node:check(env, diag)
		env:leaveblock()
	end
	if self.else_block then
		env:enterblock()
		ifstmt.else_block = self.else_block:check(env, diag)
		env:leaveblock()
	end

	return ifstmt
end

function ast.WhileStatement:check(env, diag)
	local whilestmt = self:clone()

	local cond, condtype = self.cond:check(env, diag)
	whilestmt.cond 			 = cond
	if condtype ~= t.error and condtype ~= t.bool then
		diag:reporterror(self, "Expected bool expression but found " ..
													 condtype:toString())
	end

	env:enterblock()
	whilestmt.body = self.body:check(env, diag)
	env:leaveblock()

	return whilestmt
end

function ast.DoStatement:check(env, diag)
	local dostmt = self:clone()

	env:enterblock()
	dostmt.body = self.body:check(env, diag)
	env:leaveblock()

	return dostmt
end

function ast.RepeatStatement:check(env, diag)
	local repeatstmt = self:clone()
	env:enterblock()

	repeatstmt.body = self.body:check(env, diag)

	local cond, condtype = self.cond:check(env, diag)
	repeatstmt.cond 		 = cond
	if condtype ~= t.error and condtype ~= t.bool then
		diag:reporterror(self, "Expected bool expression but found " .. condtype:toString())
	end

	env:leaveblock()
	return repeatstmt
end

function ast.ExprStatement:check(env, diag)
	local expstmt = self:clone()
	expstmt.exp   = self.exp:check(env, diag)
	return expstmt
end

function ast.Assignment:check(env, diag)
	local assignment = self:clone()

	local lhs, ltype  = self.lvalue:check_lhs(env, diag, true)
	assignment.lvalue = lhs
	if ltype == t.error then return end

	-- Only refactor the left binaryOp tree if we could potentially
	-- be commiting a field/scalar write
	if self.lvalue:isGlobalScope() and
		 Type.isScalar(self.lvalue.luaval) or
		 Type.isFieldIndex(self.lvalue.luaval) and
	   ast.BinaryOp:is(self.exp) then
		self.exp:refactorReduction()
	end

	local rhs, rtype = self.exp:check(env, diag)
	assignment.exp   = rhs
	if rtype == t.error then return end

	-- is this the first use of a declared, but not defined, variable?
	if ltype == t.unknown then
		ltype                 = rtype
		self.lvalue.node_type = rtype
		--assignment.lvalue.node_type = rtype

		--env:localenv()[assignment.lvalue.name].node_type = rtype
		env:localenv()[self.lvalue.name].node_type = rtype
	end

	local err_msg = "Global assignments only valid for indexed fields or scalars (did you mean to use a scalar here?)"
	-- if the lhs is from the global scope, then it must be an indexed field or a scalar:
	if self.lvalue:isGlobalScope() then
		if not Type.isFieldIndex(self.lvalue.luaval) and not Type.isScalar(self.lvalue.luaval) then
			diag:reporterror(self.lvalue, err_msg)
			return
		end
	end

	-- local temporaries can only be updated if they refer to
	-- numeric/boolean/vector data types.  We do not allow users to
	-- re-assign variables of topological element type, etc.
	-- This way, they do not confuse our stencil analysis.
	if self.lvalue:isLocalScope() and not ltype:isExpressionType() then
		diag:reporterror(self.lvalue, "Cannot update local variables referring to objects of topological type")
		return
	end

	local derived = type_meet(ltype,rtype)

	if derived == t.error or
	   (ltype:isPrimitive() and rtype:isVector()) or
	   (ltype:isVector()    and rtype:isVector() and ltype.N ~= rtype.N) then
		diag:reporterror(self, "invalid conversion from " .. rtype:toString() .. ' to ' .. ltype:toString())
		return
	end

	-- Determine if this assignment is a field write or a reduction (since this requires special codegen)
	if self.lvalue:isLocalScope() then return end

	local lval = self.lvalue.luaval
	local rexp = self.exp

	if Type.isFieldIndex(lval) then
		local lfield = self.lvalue.func.luaval
		self.field   = lfield -- lua object
		self.topo    = self.lvalue.params.children[1] -- liszt ast node

		if not ast.BinaryOp:is(rexp) or not Type:isFieldIndex(rexp.lhs.luaval) then
			self.fieldop = _FIELD_WRITE
		else
			local rfield = rexp.lhs.func.luaval
			self.fieldop = (lfield == rfield and rexp:operatorCommutes()) and _FIELD_REDUCE or _FIELD_WRITE
			if self.fieldop == _FIELD_REDUCE then
				self.rexp  = rexp.rhs -- liszt ast node
			end
		end
	end

	if Type.isScalar(lval) then
		if not ast.BinaryOp:is(rexp) or not Type.isScalar(rexp.lhs.luaval) then
			diag:reporterror(self, "Scalar variables can only be modified through reductions")

		else
			-- Make sure the scalar objects on the lhs and rhs match.  Otherwise, we are 
			-- looking at a write to the lhs scalar, which is illegal.
			local lsc = self.lvalue.luaval
			local rsc = rexp.lhs.luaval

			if lsc ~= rsc then
				diag:reporterror(self, "Scalar variables can only be modified through reductions")
			else
				self.fieldop = _SCALAR_REDUCE
				self.scalar  = lsc
				self.rexp    = rexp.rhs
			end
		end
	end
end

function ast.InitStatement:check(env, diag)
	local initstmt 			= self:clone()
		initstmt.ref 			= clone_name(self.ref)
	local varname  			= self.ref.name
	local rhs, rhstype 	= self.exp:check(env, diag)
		initstmt.exp = rhs

	if rhstype == t.error then return end

	-- Local temporaries can only refer to numeric/boolean/vector data or
	-- topolocal element types, and variables referring to topo types can
	-- never be re-assigned.  That way, we don't allow users to write
	-- code that makes stencil analysis intractable.
	if not rhstype:isExpressionType() and not rhstype:isTopo() then
		diag:reporterror(self,"Can only assign numbers, bools, "..
													"or topological elements to local temporaries")
	end

	self.ref.node_type = rhstype
	self.ref:setLocalScope()
	env:localenv()[varname] = self.exp
	--initstmt.ref.node_type = rhstype
	--initstmt.ref:setLocalScope()
	--env:localenv()[varname] = initstmt.exp

	return initstmt
end

function ast.DeclStatement:check(env, diag)
	local decl 			= self:clone()
		decl.ref 			= clone_name(self.ref)
	self.ref.node_type = t.unknown
	env:localenv()[self.ref.name] = self.ref
	--decl.ref.node_type = t.unknown
	--env:localenv()[decl.ref.name] = decl.ref
	return decl
end

function ast.AssertStatement:check(env, diag)
	local assertstmt = self:clone()
	local test, test_type = self.test:check(env, diag)
	assertstmt.test 			= test
	if test_type ~= t.error and test_type ~= t.bool then
		diag:reporterror(self,
						"Expected a boolean as the test for assert statement")
	end
	return assertstmt
end

function ast.PrintStatement:check(env, diag)
	local printstmt = self:clone()
	local out, outtype = self.output:check(env, diag)
	printstmt.out 		 = out
	if outtype ~= t.error and not outtype:isExpressionType() then
		diag:reporterror(self, "Only numbers, bools, "..
													 "and vectors can be printed")
	end
	return printstmt
end

local function enforce_numeric_type (node, tp)
	if tp ~= t.error and not tp:isNumeric() then
		diag:reporterror(node, "Expected a numeric expression "..
													 "to define the iterator bounds/step (found "..
													 tp:toString() .. ')')
	end
end

function ast.NumericFor:check(env, diag)
	local numfor = self:clone()
	local lower, lower_type = self.lower:check(env, diag)
	local upper, upper_type = self.upper:check(env, diag)
	numfor.lower, numfor.upper = lower, upper
	enforce_numeric_type(self, lower_type)
	enforce_numeric_type(self, upper_type)
	local step, step_type
	if self.step then
		step, step_type = self.step:check(env, diag)
		numfor.step = step
		enforce_numeric_type(self, step_type)
	end

	-- infer iterator type
	self.iter.node_type = type_meet(lower_type, upper_type)
	numfor.node_type = type_meet(lower_type, upper_type)
	if step_type then
		self.iter.node_type = type_meet(self.iter.node_type, step_type)
	end

	self.iter:setLocalScope()

	env:enterblock()
	local varname = self.iter.name
	env:localenv()[varname] = self.iter
	self.body:check(env, diag)
	env:leaveblock()

	return numfor
end

function ast.GenericFor:check(env, diag)
	diag:reporterror(self, "Generic for statement not yet implemented", 2)
end

function ast.Break:check(env, diag)
	-- nothing needed
	return self:clone()
end

function ast.CondBlock:check(env, diag)
	local condblock = self:clone()
	local cond, condtype = self.cond:check(env, diag)
		condblock.cond  	 = cond
	if condtype ~= t.error and condtype ~= t.bool then
		diag:reporterror(self, "Conditional expression type should be"..
												   "boolean (found " .. condtype:toString() .. ")")
	end

	env:enterblock()
	condblock.body = self.body:check(env, diag)
	env:leaveblock()

	return condblock
end

------------------------------------------------------------------------------
--[[                         Expression Checking:                         ]]--
------------------------------------------------------------------------------
function ast.Expression:check(env, diag)
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

    -- can "(a <lop> b) <rop> c" be refactored into "a <lop'> (b <rop'> c)"?
local function commutes (lop, rop)
	local additive       = { ['+'] = true, ['-'] = true }
	local multiplicative = { ['*'] = true, ['/'] = true }
	if additive[lop]       and additive[rop]       then return true end
	if multiplicative[lop] and multiplicative[rop] then return true end
	if lop == 'and' and rop == 'and' then return true end
	if lop == 'or'  and rop == 'or'  then return true end
	return false
end

function ast.BinaryOp:refactorReduction ()
	-- recursively refactor the left-hand child
	if self.lhs.kind ~= 'binop' then return end
	self.lhs:refactorReduction()

	local rop   = self.op
	local lop   = self.lhs.op

	local simple = {['+'] = true, ['*'] = true, ['and'] = true, ['or'] = true}
	if commutes(lop, rop) then
		--[[ 
		We want to pull up the left grandchild to be the left child of
		this node.  Thus, we'll need to refactor the tree like so:

		  self ->     *                  *
		             / \                / \
		            /   \              /   \
		  left ->  *     C    ==>     A     *
		          / \                      / \
		         /   \                    /   \
		        A     B                  B     C
		]]--
		local left = self.lhs
		local A    = left.lhs
		local B    = left.rhs
		local C    = self.rhs
		self.lhs = A
		self.rhs = left
		left.lhs = B
		left.rhs = C

		-- if the left operator is an inverse operator, we'll need to
		-- change the right operator as well.
		if not simple[lop] then
    		-- switch the right operator to do the inverse operation
    		local inv_map = { ['+'] = '-', ['-'] = '+', ['*'] = '/', ['/'] = '*' }
    		rop         = inv_map[rop]
			self.op     = lop
			self.rhs.op = rop    			
    	end
    end
end

	-- binary expressions
function ast.BinaryOp:check(env, diag)
	local binop = self:clone()
	binop.op 		= self.op
	local lefttype, righttype
	binop.lhs, lefttype  = self.lhs:check(env, diag)
	binop.rhs, righttype = self.rhs:check(env, diag)
	local op    = self.op
	local node  = self

	-- factors out common code
	local function err(msg)
		if msg then diag:reporterror(node, msg) end
		node.node_type = t.error
		binop.node_type = t.error
		return binop, t.error
	end

	-- Silently ignore/propagate errors
	if lefttype == t.error or righttype == t.error then return err() end

	local type_err = "incompatible types: " .. lefttype:toString() ..
									 ' and ' .. righttype:toString()
	local op_err  = "invalid types for operator " .. op .. ': ' ..
									lefttype:toString() .. ' and ' .. righttype:toString()

	if not lefttype:isExpressionType() or not righttype:isExpressionType() then
		return err(type_err)
	end

	local derived = type_meet(lefttype, righttype)

	if isNumOp[op] then
		if lefftype:isVector() or righttype:isVector() then return err(op_err) end
	elseif isCompOp[op] then
		if lefttype:isLogical() ~= righttype:isLogical() then return err(op_err)

		-- if the type_meet failed, types are incompatible
		elseif derived == t.error then return err(type_err)

		-- otherwise, we need to return a logical type
		elseif derived:isPrimitive() then
			self.node_type = t.bool
			binop.node_type = t.bool
			return binop, t.bool

		else
			self.node_type = t.vector(t.bool,derived.N)
			binop.node_type = t.vector(t.bool,derived.N)
			return binop, binop.node_type
		end

	elseif isBoolOp[op] then
		if not lefttype:isLogical() or not righttype:isLogical() then
			return err(op_err)
		end
	end

	if derived == t.error then
		diag:reporterror(self, type_err)
	end

	self.node_type = derived
	binop.node_type = derived
	return binop, binop.node_type
end

function ast.UnaryOp:check(env, diag)
	local unop = self:clone()
	unop.op 	 = self.op
	local exprtype
	unop.exp, exprtype = self.exp:check(env, diag)

	-- default to error, try to prove otherwise
	self.node_type = t.error
	unop.node_type = t.error

	if self.op == 'not' then
		if exprtype:isLogical() then
			self.node_type = exprtype
			unop.node_type = exprtype
		else
			diag:reporterror(self, "Unary \"not\" expects a boolean operand")
		end
	elseif self.op == '-' then
		if exprtype:isNumeric() then
			self.node_type = exprtype
			unop.node_type = exprtype
		else
			diag:reporterror(self, "Unary minus expects a numeric operand")
		end
	else
		diag:reporterror(self, "Unknown unary operator \'".. self.op .."\'")
	end

	return unop, unop.node_type
end

------------------------------------------------------------------------------
--[[                               Variables                              ]]--
------------------------------------------------------------------------------
-- Infer liszt type for the lua variable
local function lua_to_liszt(luav)
	-- all liszt objects must be of lua type table...
	if type(luav) == 'table' then
		-- vectors, scalars
		if Type.isVector(luav) or Type.isScalar(luav) then
			return luav.type

		-- fields
		elseif Type.isField(luav) then
			return t.field(luav.topo,luav.type)

		-- topological set
		elseif  Type.isSet(luav) then
			error("Typechecking for TopoSets not yet implemented", 3)

		-- table does not represent a liszt type, but needs to be returned
		-- since we support the select operator
		-- terra globals
		elseif luav.isglobal then
			return tutil.ltype(luav.type)

		else
			return t.table
		end

	elseif (type(luav) == 'number') then
		if luav % 1 == luav then return t.int else return t.float end

	elseif (type(luav) == 'boolean') then
		return t.bool

	else
		return t.error
	end
end

function ast.LValue:check_lhs(env, diag)
	return self:check(env, diag)
end

function ast.Name:check_lhs(env, diag)
	return self:check(env, diag, true)
end

function ast.Name:check(env, diag, assign)
	local name = self:clone()
	name.name  = self.name

	-- try to find the name in the local scope
	local locv = env:localenv()[name.name]
	if locv then
		local loctype = locv.node_type
		-- cannot read from an uninitialized variable
		if loctype == t.unknown and not assign then
			diag:reporterror(self, "Variable '" .. self.name .. "' is not defined")
			loctype = t.error
		else
			self:setLocalScope()
			name:setLocalScope()
		end
		self.node_type = loctype
		name.node_type = loctype
		return name, loctype
	end

	-- Otherwise, does the name exist in the global scope?
	local luav = env:luaenv()[name.name]
	if luav then
		self.luaval 	 = luav
		name.node_type = lua_to_liszt(luav)
		self.node_type = name.node_type
		if name.node_type == t.error then
			diag:reporterror(self, "Cannot convert the lua value to a liszt value")
		end
		self:setGlobalScope()
		name:setGlobalScope()
		return name, name.node_type
	end

	-- failed to find this name anywhere
	diag:reporterror(self, "Variable '" .. self.name .. "' is not defined")
	self.node_type = t.error
	name.node_type = t.error
	return name, t.error
end

function ast.Number:check(env, diag)
	local number = self:clone()
	number.value = self.value
	if tonumber(self.value) % 1 == 0 then
		self.node_type = t.int
		number.node_type = t.int
	else
		self.node_type = t.float
		number.node_type = t.float
	end
	return number, number.node_type
end

function ast.VectorLiteral:check(env, diag)
	local veclit = self:clone()
	veclit.elems = {}
	local type_so_far
	veclit.elems[1], type_so_far = self.elems[1]:check(env, diag)
	local node  = self

	local tp_error = "Vector literals can only contain literals "..
									 "of boolean or numeric type"
	local mt_error = "Vector entries must be of the same type"
	local function err(msg)
		if msg then diag:reporterror(node, msg) end
		node.node_type = t.error
		veclit.node_type = t.error
		return veclit, t.error
	end

	if type_so_far == t.error then return err() end
	if not type_so_far:isPrimitive() then return err(tp_error) end

	for i = 2, #self.elems do
		local tp
		veclit.elems[i], tp = self.elems[i]:check(env, diag)
		if not tp:isPrimitive() then return err(tp_error) end

		type_so_far = type_meet(type_so_far, tp)
		if not type_so_far then return err(mt_error) end
	end

	self.node_type = t.vector(type_so_far, #self.elems)
	veclit.node_type = t.vector(type_so_far, #veclit.elems)
	return veclit, veclit.node_type
end

function ast.Bool:check(env, diag)
	local boolnode = self:clone()
	boolnode.value = self.value
	boolnode.node_type = t.bool
	self.node_type = t.bool
	return boolnode, boolnode.node_type
end

------------------------------------------------------------------------------
--[[                         Miscellaneous nodes:                         ]]--
------------------------------------------------------------------------------
function ast.Tuple:check(env, diag)
	local tuple = self:clone()
	tuple.children = {}
	for i, node in ipairs(self.children) do
		tuple.children[i] = node:check(env, diag)
	end
	return tuple
end

function ast.Tuple:index_check(env, diag)
	-- type checking tuple when it should be a single argument, for
	-- instance, when indexing a field
	if #self.children ~= 1 then
		diag:reporterror(self, "Can use exactly one argument to index here")
		return nil
	end
	local _, argobj = self.children[1]:check(env, diag)
	if argobj == nil then
		return nil
	end
	self.node_type = argobj
	return argobj
end

function ast.TableLookup:check(env, diag)
	local lookup  = self:clone()
	lookup.member = clone_name(self.member)
	local ttype 
	lookup.table, ttype = self.table:check(env, diag)
	local node 		= self

	local function err(msg)
		if msg then diag:reporterror(node, msg) end
		node.node_type = t.error
		lookup.node_type = t.error
		return lookup, t.error
	end

	if ttype == t.error then return err()
	elseif ttype ~= t.table then
		return err("select operator not supported for non-table type " ..
							 ttype:toString())
	end

	-- RHS is a member of the LHS
	self.luaval = self.table.luaval[self.member.name]
	self:setGlobalScope()

	if self.luaval == nil then
		return err("Object does not have member " .. member)
	else
		lookup.node_type = lua_to_liszt(self.luaval)
		self.node_type = lookup.node_type
		if self.node_type == t.error then
			diag:reporterror(self, "Cannot convert the lua value to a Liszt value")
		end
	end

	return lookup, lookup.node_type
end

function ast.VectorIndex:check(env, diag)
	local vidx = self:clone()
	local vector_type, index_type
	vidx.vector, vector_type = self.vector:check(env, diag)
	vidx.index,  index_type  = self.index:check(env,diag)

	-- RHS is an expression of integral type
	-- (make sure this check always runs)
	if index_type ~= t.error and not index_type:isIntegral() then
		diag:reporterror(self, "Expected an integer expression to index into "..
													 "the vector (found ".. index_type:toString() ..')')
	end

	-- LHS is a vector
	if vector_type ~= t.error and not vector_type:isVector() then
		diag:reporterror(self, "indexing operator [] not supported for "..
													 "non-vector type " .. vector_type:toString())
		vector_type = t.error
	end
	if vector_type == t.error then
		self.node_type = t.error
		vidx.node_type = t.error
		return vidx, t.error
	end

	self.node_type = vector_type:baseType()
	vidx.node_type = vector_type:baseType()
	return vidx, vidx.node_type
end

function ast.Call:check(env, diag)
	local call = self:clone()
	-- call name can be a field only in current implementation
	local ftype
	call.func, ftype = self.func:check(env, diag)
	if ftype == t.error then
		call.node_type = t.error

	elseif ftype:isField() then
		local ftopo = ftype:topoType()
		local fdata = ftype:dataType()

		local argtype = self.params:index_check(env,diag)
		if (argtype == t.error) then
			call.node_type = t.error

		elseif argtype == ftopo then
			call.node_type = fdata
			self.luaval    = FieldIndex.New(self.func.luaval, fdata)

		-- infer type of argument to field topological type
		elseif argtype == t.topo then
			local pname = self.params.children[1].name
			env:localenv()[pname].node_type = ftopo -- update type in local environment

			call.node_type = fdata
			self.luaval = FieldIndex.New(self.func.luaval, fdata)

		else
			diag:reporterror(self, ftype:toString(), " indexed by incorrect type ", argtype:toString())
			call.node_type = t.error
		end

	else
		diag:reporterror(self, "Invalid call")
		call.node_type = t.error
	end

	self.node_type = call.node_type
	self:setGlobalScope()
	return call, call.node_type
end



------------------------------------------------------------------------------
--[[ Semantic checking called from here:                                  ]]--
------------------------------------------------------------------------------
local function check_kernel_param(param, env, diag)
	local new_param 			= clone_name(param)
	new_param.node_type		= t.topo

	env:localenv()[param.name] = new_param

	return new_param
end

function exports.check(luaenv, kernel_ast)

	-- environment for checking variables and scopes
	local env  = terralib.newenvironment(luaenv)
	local diag = terralib.newdiagnostics()

	--------------------------------------------------------------------------

	local new_kernel_ast = kernel_ast:clone()

	diag:begin()
	env:enterblock()

	new_kernel_ast.param = check_kernel_param(kernel_ast.param, env, diag)
	new_kernel_ast.body  = kernel_ast.body:check(env, diag)

	env:leaveblock()
	diag:finishandabortiferrors("Errors during typechecking liszt", 1)

	return new_kernel_ast
end


return exports


