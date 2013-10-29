local exports   = {}

ast = require("ast")
local types = terralib.require("compiler/types")
local tutil     = types.usertypes
local Type      = types.Type
local t         = types.t
local type_meet = types.type_meet
local Scope     = types.Scope

-- Phases used in assignment statements
local _FIELD_WRITE   = 'FIELD_WRITE'
local _FIELD_REDUCE  = 'FIELD_REDUCE'
local _SCALAR_REDUCE = 'SCALAR_REDUCE'
exports._FIELD_WRITE 		= _FIELD_WRITE
exports._FIELD_REDUCE 	= _FIELD_REDUCE
exports._SCALAR_REDUCE 	= _SCALAR_REDUCE



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
	-- statements
	for id, node in ipairs(self.statements) do
		node:check(env, diag)
	end
end

function ast.IfStatement:check(env, diag)
	for id, node in ipairs(self.if_blocks) do
		env:enterblock()
		node:check(env, diag)
		env:leaveblock()
	end
	if self.else_block then
		env:enterblock()
		self.else_block:check(env, diag)
		env:leaveblock()
	end
end

function ast.WhileStatement:check(env, diag)
	local condtype = self.cond:check(env, diag)
	if condtype ~= t.error and condtype ~= t.bool then
		diag:reporterror(self, "Expected bool expression but found " .. condtype:toString())
	end

	env:enterblock()
	self.body:check(env, diag)
	env:leaveblock()
end

function ast.DoStatement:check(env, diag)
	env:enterblock()
	self.body:check(env, diag)
	env:leaveblock()
end

function ast.RepeatStatement:check(env, diag)
	env:enterblock()
	self.body:check(env, diag)

	local condtype = self.cond:check(env, diag)
	if condtype ~= t.error and condtype ~= t.bool then
		diag:reporterror(self, "Expected bool expression but found " .. condtype:toString())
	end
	env:leaveblock()
end

function ast.ExprStatement:check(env, diag)
	self.exp:check(env, diag)
end

function ast.Assignment:check(env, diag)
	local ltype = self.lvalue:check_lhs(env, diag, true)
	if ltype == t.error then return end

	-- Only refactor the left binaryOp tree if we could potentially
	-- be commiting a field/scalar write
	if self.lvalue:isGlobalScope() and
		 Type.isScalar(self.lvalue.luaval) or
		 Type.isFieldIndex(self.lvalue.luaval) and
	   ast.BinaryOp:is(self.exp) then
		self.exp:refactorReduction()
	end

	local rtype = self.exp:check(env, diag)
	if rtype == t.error then return end

	-- is this the first use of a declared, but not defined, variable?
	if ltype == t.unknown then
		ltype                 = rtype
		self.lvalue.node_type = rtype

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
	local varname = self.ref.name
	local rhstype = self.exp:check(env, diag)

	if rhstype == t.error then return end

	-- Local temporaries can only refer to numeric/boolean/vector data or
	-- topolocal element types, and variables referring to topo types can
	-- never be re-assigned.  That way, we don't allow users to write
	-- code that makes stencil analysis intractable.
	if not rhstype:isExpressionType() and not rhstype:isTopo() then
		diag:reporterror(self, "Can only assign numbers, bools, or topological elements to local temporaries")
	end

	self.ref.node_type = rhstype
	self.ref:setLocalScope()
	env:localenv()[varname] = self.exp
end

function ast.DeclStatement:check(env, diag)
	self.ref.node_type = t.unknown
	env:localenv()[self.ref.name] = self.ref
end

local function enforce_numeric_type (node, tp)
	if tp ~= t.error and not tp:isNumeric() then
		diag:reporterror(self, "Expected a numeric expression to define the iterator bounds/step (found " .. tp:toString() .. ')')
	end
end

function ast.NumericFor:check(env, diag)
	local lower, upper = self.lower:check(env, diag), self.upper:check(env, diag)
	local step
	enforce_numeric_type(self, lower)
	enforce_numeric_type(self, upper)
	if self.step then
		step = self.step:check(env, diag)
		enforce_numeric_type(self, step)
	end

	-- infer iterator type
	self.iter.node_type = type_meet(lower, upper)
	if step then
		self.iter.node_type = type_meet(self.iter.node_type, step)
	end

	self.iter:setLocalScope()

	env:enterblock()
	local varname = self.iter.name
	env:localenv()[varname] = self.iter
	self.body:check(env, diag)
	env:leaveblock()
end

function ast.GenericFor:check(env, diag)
	diag:reporterror(self, "Generic for statement not yet implemented", 2)
end

function ast.Break:check(env, diag)
	-- nothing needed
end

function ast.CondBlock:check(env, diag)
	local condtype = self.cond:check(env, diag)
	if condtype ~= t.error and condtype ~= t.bool then
		diag:reporterror(self, "Conditional expression type should be boolean (found " .. condtype:toString() .. ")")
	end

	env:enterblock()
	self.body:check(env, diag)
	env:leaveblock()
end

------------------------------------------------------------------------------
--[[                         Expression Checking:                         ]]--
------------------------------------------------------------------------------
function ast.Expression:check(env, diag)
	error("Semantic checking has not been implemented for expression type " .. self.kind)
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
	local lefttype  = self.lhs:check(env, diag)
	local righttype = self.rhs:check(env, diag)
	local op        = self.op

	-- Silently ignore/propogate errors
	if lefttype == t.error or righttype == t.error then
		self.node_type = t.error
		return self.node_type
	end

	local typ_err = "incompatible types: " .. lefttype:toString() .. ' and ' .. righttype:toString()
	local op_err  = "invalid types for operator " .. op .. ': ' .. lefttype:toString() .. ' and ' .. righttype:toString()

	if not lefttype:isExpressionType() or not righttype:isExpressionType() then
		diag:reporterror(self, type_err)
		self.node_type = t.error
		return self.node_type
	end

	local derived = type_meet(lefttype, righttype)

	if isNumOp[op] then
		if lefftype:isVector() or righttype:isVector() then
			diag:reporterror(self, op_err)
			self.node_type = t.error
			return self.node_type
		end

	elseif isCompOp[op] then
		if lefttype:isLogical() ~= righttype:isLogical() then
			diag:reporterror(self, op_err)
			self.node_type = t.error
			return self.node_type

		-- if we can't compute a valid derived type, then the operands are not
		-- comparable, so issue and error
		elseif derived == t.error then
			diag:reporterror(self, typ_err)
			self.node_type = t.error
			return self.node_type

		-- otherwise, we need to return a logical type
		elseif derived:isPrimitive() then
			self.node_type = t.bool
			return self.node_type

		else
			self.node_type = t.vector(t.bool,derived.N)
			return self.node_type
		end

	elseif isBoolOp[op] then
		if not lefttype:isLogical() or not righttype:isLogical() then
			diag:reporterror(self, op_err)
			self.node_type = t.error
			return node_type
		end
	end

	if derived == t.error then
		diag:reporterror(self, typ_err)
	end

	self.node_type = derived
	return self.node_type
end

function ast.UnaryOp:check(env, diag)
	local exprtype = self.exp:check(env, diag)
	if exprtype == t.error then
		self.node_type = t.error

	elseif self.op == 'not' then
		if not exprtype:isLogical() then
			diag:reporterror(self, "Unary \"not\" expects a boolean operand")
			self.node_type = t.error
		else
			self.node_type = exprtype
		end

	elseif self.op == '-' then
		if not exprtype:isNumeric() then
			diag:reporterror(self, "Unary minus expects a numeric operand")
			self.node_type = t.error
		else
			self.node_type = exprtype
		end

	else
		diag:reporterror(self, "Unknown unary operator \'"..op.."\'")
		self.node_type = t.error
	end

	return self.node_type
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

        elseif Type.isFunction(luav) then
            return t.func

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
	local node = env:localenv()[self.name]
	local locv = node and node.node_type -- we store the types of local variables in the environment

	-- if we're trying to read an undefined variable, report an error
	if locv == t.unknown and not assign then
		diag:reporterror(self, "Variable '" .. self.name .. "' is not defined")
		self.node_type = t.error
		return self.node_type

	-- if we found a type in the environment, the name must be defined locally
	elseif locv then
		self:setLocalScope()
		self.node_type = locv
		return self.node_type
	end

	-- Otherwise, does the name exist in the global scope?
	local luav = env:luaenv()[self.name]
	if not luav then
		diag:reporterror(self, "Variable '" .. self.name .. "' is not defined")
		self.node_type = t.error
		return self.node_type
	end

	self.luaval    = luav
	self.node_type = lua_to_liszt(luav)
	if self.node_type == t.error then
		diag:reporterror(self, "Cannot convert the lua value to a liszt value")
	end

	self:setGlobalScope()
	return self.node_type
end

function ast.Number:check(env, diag)
	if tonumber(self.value) % 1 == 0 then
		self.node_type = t.int
	else
		self.node_type = t.float
	end
	return self.node_type
end

function ast.VectorLiteral:check(env, diag)
	local first = self.elems[1]:check(env, diag)
	if first == t.error then
		return first
	end

	local tp_error = "Vector literals can only contain expressions of boolean or numeric type"
	local mt_error = "Vectors cannot have mixed numeric and boolean types"

	if not first:isNumeric() then
		diag:reporterror(self, tp_error)
		self.node_type = t.error
		return self.node_type
	end

	local meeted_type = first
	for i = 2, #self.elems do
		local tp = self.elems[i]:check(env, diag)
		if not tp:isPrimitive() then
			diag:reporterror(self, tp_error)
			self.node_type = t.error
			return self.node_type
		end
		meeted_type = type_meet(meeted_type, tp)
		if not meeted_type then
			diag:reporterror(self, mt_error)
			self.node_type = t.error
			return self.node_type
		end
	end

	self.node_type = t.vector(meeted_type, #self.elems)
	return self.node_type
end

function ast.Bool:check(env, diag)
	self.node_type = t.bool
	return self.node_type
end

------------------------------------------------------------------------------
--[[                         Miscellaneous nodes:                         ]]--
------------------------------------------------------------------------------
function ast.Tuple:check(env, diag)
	for i, node in ipairs(self.children) do
		node:check(env, diag)
	end
end

function ast.Tuple:index_check(env, diag)
	-- type checking tuple when it should be a single argument, for
	-- instance, when indexing a field
	if #self.children ~= 1 then
		diag:reporterror(self, "Can use exactly one argument to index here")
		return nil
	end
	local argobj = self.children[1]:check(env, diag)
	if argobj == nil then
		return nil
	end
	self.node_type = argobj
	return argobj
end

function ast.TableLookup:check(env, diag)
	local ttype = self.table:check(env, diag)

	if ttype == t.error then
		self.node_type = t.error
		return self.node_type

	-- Enforce table type for lhs
	elseif ttype ~= t.table then
		diag:reporterror(self, "select operator not supported for non-table type " .. ttype:toString())
		self.node_type = t.error
		return self.node_type
	end

	-- RHS is a member of the LHS
	self.luaval = self.table.luaval[self.member.name]

	if self.luaval == nil then
		self.node_type = t.error
		diag:reporterror(self, "Object does not have member " .. member)

	else
		self.node_type = lua_to_liszt(self.luaval)
		if self.node_type == t.error then
			diag:reporterror(self, "Cannot convert the lua value to a Liszt value")
		end
	end

	self:setGlobalScope()
	return self.node_type
end

function ast.VectorIndex:check(env, diag)
	local vector_type = self.vector:check(env, diag)

	if vector_type == t.error then
		self.node_type = t.error
		return self.node_type

	-- Enforce vector type for LHS
	elseif not vector_type:isVector() then
		diag:reporterror(self, "indexing operator [] not supported for non-vector type " .. vector_type:toString())
		self.node_type = t.error
		return self.node_type
	end

	-- RHS is an expression of integral type
	local index_type = self.index:check(env,diag)
	if index_type ~= t.error and not index_type:isIntegral() then
		diag:reporterror(self, "Expected a numeric expression to index into the vector (found " .. index_type:toString() .. ')')
	end

	self.node_type = vector_type:baseType()

	return self.node_type
	--return ast.AST.check(self,env,diag) -- UNTIL WE'RE DONE DEVELOPING...
end

function ast.Call:check(env, diag)
	-- call name can only be a field or macro in current implementation
	local ftype = self.func:check(env, diag)
	if ftype == t.error then
		self.node_type = t.error

	elseif ftype:isField() then
		local ftopo = ftype:topoType()
		local fdata = ftype:dataType()

		local argtype = self.params:index_check(env,diag)
		if (argtype == t.error) then
			self.node_type = t.error

		elseif argtype == ftopo then
			self.node_type = fdata
			self.luaval    = FieldIndex.New(self.func.luaval, self.node_type)

		-- infer type of argument to field topological type
		elseif argtype == t.topo then
			local pname = self.params.children[1].name
			env:localenv()[pname].node_type = ftopo -- update type in local environment

			self.node_type = fdata
			self.luaval = FieldIndex.New(self.func.luaval, self.node_type)

		else
			diag:reporterror(self, ftype:toString(), " indexed by incorrect type ", argtype:toString())
			self.node_type = t.error
		end

	elseif ftype:isFunction() then
        self.node_type = self.func.luaval.check(self, env, diag)
    else
		diag:reporterror(self, "Invalid call")
		self.node_type = t.error
	end

	self:setGlobalScope()
	return self.node_type
end


------------------------------------------------------------------------------
--[[ Semantic checking called from here:                                  ]]--
------------------------------------------------------------------------------
function exports.check(luaenv, kernel_ast)

	-- environment for checking variables and scopes
	local env  = terralib.newenvironment(luaenv)
	local diag = terralib.newdiagnostics()

	--------------------------------------------------------------------------

	diag:begin()
	env:enterblock()

	kernel_ast.param.node_type = t.topo
	env:localenv()[kernel_ast.param.name] = kernel_ast.param
	kernel_ast.body:check(env, diag)

	env:leaveblock()
	diag:finishandabortiferrors("Errors during typechecking liszt", 1)
end


return exports


