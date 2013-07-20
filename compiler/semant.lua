module(... or 'semant', package.seeall)

ast = require("ast")

--[[ Declare types
--]]
local NOSCOPE = 'noscope'
local SLUA = 'lua'
local SLISZT = 'liszt'
local NOTYPE = 'notype'
local INT = 'int'
local FLOAT = 'float'
local VECTOR = 'vector'
local NUM = 'number'
local TAB = 'table'

Type = {}
function Type:new()
	local newtype = 
	{
		inftype = NOTYPE,
		scope = NOSCOPE,
		size = 1
	}
	setmetatable(newtype, {__index = self})
	return newtype
end

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
	local function conform(lhsobj, rhsobj)
		return true
--		if rhsobj.inftype == lhstype.inftype then
--			return true
--		else
--			return false
--		end
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
		print("Child 1 type:", self.children[1].kind)
		print("Child 2 type:", self.children[2].kind)
		local lhsobj = self.children[1]:check()
		local rhsobj = self.children[2]:check()
		local validassgn = conform(lhsobj, rhsobj)
		print("LHS", lhsobj, "RHS", rhsobj.inftype)
		if not validassgn then
			diag:reporterror(self,"Inferred RHS type", rhstype, "does not conform to inferred LHS type", lhstype)
		end
	end

	function ast.InitStatement:check()
		-- name, expression
	end

	function ast.DeclStatement:check()
		-- name
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
		if locv then

		else
			lv = env:luaenv()[self.children[1]]
			if not lv then
				diag:reporterror(self, "Left hand variable not defined in an assignment")
			else
				-- TODO: lua value
				local nametype = Type:new()
				ametype.inftype = type(lv)
				ametype.scope = SLUA
				ametype.size = 1
			end
		end
	end

	function ast.Number:check()
		local numtype = Type:new()
		numtype.inftype = FLOAT
		numtype.scope = SLISZT
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
