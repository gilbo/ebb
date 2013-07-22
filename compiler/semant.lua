module(... or 'semant', package.seeall)

ast = require("ast")

--[[ Declare types
--]]
_NOSCOPE_STR = 'noscope'
_LUA_STR = 'lua'
_LISZT_STR = 'liszt'

_NOTYPE_STR = 'notype'
_INT_STR = 'int'
_FLOAT_STR = 'float'
_VECTOR_STR = 'vector'
_NUM_STR = 'number'
_TAB_STR = 'table'

-- root variable type
_NOTYPE = 
{
	name = _NOTYPE_STR,
	parent = {},
	children = {}
}
-- liszt variable type
_INT = 
{
	name = _INT_STR,
	parent = {},
	children = {}
}
-- liszt variable type
_FLOAT = 
{
	name = _FLOAT_STR,
	parent = {},
	children = {}
}
-- liszt variable type
_VECTOR = 
{
	name = _VECTOR_STR,
	parent = {},
	children = {}
}
-- number type
_NUM = 
{
	name = _NUM_STR,
	parent = {},
	children = {}
}
-- table type
_TAB  =
{
	name = _TAB_STR,
	parent = {},
	children = {}
}

ObjType = 
{
	objtype = _NOTYPE,
	scope = _NOSCOPE_STR,
	elemtype = _NOTYPE,
	size = 0
}

function ObjType:new()
	local newtype = {}
	setmetatable(newtype, {__index = self})
	newtype.objtype = self.typename
	newtype.scope = self.scope
	newtype.elemtype = self.elemtype
	newtype.size = self.size
	return newtype
end

_TR = _NOTYPE
_TR.parent = _NOTYPE

_TR.children = {_NUM, _VECTOR}
_NUM.parent = _NOTYPE
_VECTOR.parent = _NOTYPE

_NUM.children = {_INT, _FLOAT}
_INT.parent = _NUM
_FLOAT.parent = _NUM

------------------------------------------------------------------------------

--[[ TODO: Implement semantic checking here
--]]
function check(luaenv, kernel_ast)

	print("AST")
	--	terralib.tree.printraw(kernel_ast)
	-- environment for checking variables and scopes
	local env = terralib.newenvironment(luaenv)
	local diag = terralib.newdiagnostics()

	------------------------------------------------------------------------------

	--[[ Check if lhstype conforms to rhstype
	--]]
	local function conform(lhstype, rhstype)
		if lhstype == _TR then
			return true
		elseif rhstype == lhstype then
			return true
		elseif rhstype == _TR then
			return false
		else
			return conform(lhstype, rhstype.parent)
		end
	end

	local function settype(lhsobj, rhsobj)
	end

	------------------------------------------------------------------------------

	--[[ Cases not handled
	--]]
	function ast.AST:check()
		print("To implement semantic checking for", self.kind)
		diag:reporterror(self, "No known method to typecheck"..self.kind)
	end

	------------------------------------------------------------------------------

	--[[ TODO: Block
	--]]
	function ast.Block:check()
		-- statements
		for id, node in ipairs(self.children) do
			node:check()
		end
	end

	------------------------------------------------------------------------------

	--[[ TODO: Statements
	--]]
	function ast.Statement:check()
		-- not needed
		for id, node in ipairs(self.children) do
			node:check()
		end
	end

	function ast.IfStatement:check()
		-- condblock and block
		for id, node in ipairs(self.children) do
			node:check()
		end
	end

	function ast.WhileStatement:check()
		-- condition expression and block
		for id, node in ipairs(self.children) do
			node:check()
		end
	end

	function ast.DoStatement:check()
		-- block
		for id, node in ipairs(self.children) do
			node:check()
		end
	end

	function ast.RepeatStatement:check()
		-- condition expression, block
	end

	function ast.ExprStatement:check()
		-- expression
	end

	function ast.Assignment:check()
		-- lhs, rhs expressions
		local lhsobj = self.children[1]:check()
		local rhsobj = self.children[2]:check()
		local validassgn = conform(lhsobj.objtype, rhsobj.objtype)
		if not validassgn then
			diag:reporterror(self,"Inferred RHS type ", rhsobj.objtype.name,
			" does not conform to inferred LHS type ", lhsobj.objtype.name)
		end
	end

	function ast.InitStatement:check()
		-- name, expression
		-- TODO: should redefinitions be allowed?
		local nameobj = ObjType:new()
		nameobj.scope = _LISZT_STR
		env:localenv()[self.children[1]] = nameobj
		local rhsobj = self.children[2]:check()
		local validassgn = conform(nameobj.objtype, rhsobj.objtype)
		if not validassgn then
			diag:reporterror(self,"Inferred RHS type ", rhsobj.objtype.name,
			" does not conform to inferred LHS type ", nameobj.objtype.name)
		else
			settype(nameobj, rhsobj)
		end
		return nameobj
	end

	function ast.DeclStatement:check()
		-- name
		-- TODO: should redefinitions be allowed?
		local nameobj = ObjType:new()
		nameobj.scope = _LISZT_STR
		env:localenv()[self.children[1]] = nameobj
		return nameobj
	end

	function ast.NumericFor:check()
	end

	function ast.GenericFor:check()
	end

	function ast.Break:check()
		-- nothing needed
	end

	function ast.CondBlock:check()
		-- condition expression, block
	end

	------------------------------------------------------------------------------

	--[[ TODO: Expressions
	--]]
	function ast.Expression:check()
	end

	------------------------------------------------------------------------------

	--[[ TODO: Misc
	--]]
	function ast.LValue:check()
	end

	function ast.BinaryOp:check()
	end

	function ast.UnaryOp:check()
	end

	function ast.Tuple:check()
	end

	function ast.TableLookup:check()
	end

	function ast.Call:check()
	end

	------------------------------------------------------------------------------

	--[[ TODO: Variables
	--]]
	function ast.Name:check()
		local locv = env:localenv()[self.children[1]]
		local nametype
		if locv then
			-- if liszt local variable, type stored in environment
			nametype = locv
		else
			local locv = env:luaenv()[self.children[1]]
			if not locv then
				diag:reporterror(self, "Assignment Left hand variable \'" .. 
					self.children[1] .. "\' is not defined")
			else
				-- TODO: infer liszt type for the lua variable
				nametype = ObjType:new()
				nametype.objtype = type(locv)
				nametype.scope = _LUA_STR
				nametype.size = 1
			end
		end
		return nametype
	end

	function ast.Number:check()
		local numtype = ObjType:new()
		numtype.objtype = _NUM
		numtype.scope = _LISZT_STR
		numtype.size = 1
		return numtype
	end

	function ast.Bool:check()
	end

	------------------------------------------------------------------------------

	-- begin actual typechecking
	diag:begin()
	env:enterblock()
	for id, node in ipairs(kernel_ast.children) do
		if (node.kind ~= nil) then
			node:check()
		end
	end
	env:leaveblock()
	diag:finishandabortiferrors("Errors during typechecking liszt", 1)
end
