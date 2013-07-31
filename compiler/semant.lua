module(... or 'semant', package.seeall)

ast = require("ast")

------------------------------------------------------------------------------
--[[NOTES: Variables may be introduced in the lua code, or in liszt code
--through initialization/ declaration statements/ in for loops.
--]]
------------------------------------------------------------------------------

--[[ Declare types
--]]
_NOSCOPE_STR = 'noscope'
_LUA_STR = 'lua'
_LISZT_STR = 'liszt'

_NOTYPE_STR = 'notype'
_INT_STR = 'int'
_FLOAT_STR = 'float'
_BOOL_STR = 'bool'
_VECTOR_STR = 'vector'
_NUM_STR = 'number'
_TAB_STR = 'table'
_MESH_STR = 'mesh'
_CELL_STR = 'cell'
_FACE_STR = 'face'
_EDGE_STR = 'edge'
_VERTEX_STR = 'vertex'
_TOPOSET_STR = 'toposet'

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
_BOOL = 
{
	name = _BOOL_STR,
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
-- mesh type
_MESH  =
{
	name = _MESH_STR,
	parent = {},
	children = {}
}
-- cell type
_CELL  =
{
	name = _CELL_STR,
	parent = {},
	children = {}
}
-- face type
_FACE  =
{
	name = _FACE_STR,
	parent = {},
	children = {}
}
-- edge type
_EDGE  =
{
	name = _EDGE_STR,
	parent = {},
	children = {}
}
-- vertex type
_VERTEX  =
{
	name = _VERTEX_STR,
	parent = {},
	children = {}
}
-- topological set type
_TOPOSET  =
{
	name = _TOPOSET_STR,
	parent = {},
	children = {}
}

local ObjType = 
{
	objtype = _NOTYPE,
	scope = _LISZT_STR,
	elemtype = _NOTYPE,
    luaval = {},
	size = 0,
	defn = {}
}

function ObjType:new()
	local newtype = {}
	setmetatable(newtype, {__index = self})
	newtype.objtype = self.objtype
	newtype.scope = self.scope
	newtype.elemtype = self.elemtype
	newtype.size = self.size
	return newtype
end

-- class tree
_TR = _NOTYPE
_TR.parent = _NOTYPE
--
_TR.children =
{
    _NUM,
    _BOOL,
    _VECTOR,
    _MESH,
    _CELL,
    _FACE,
    _EDGE,
    _VERTEX,
    _TOPOSET
}
_NUM.parent = _NOTYPE
_BOOL.parent = _NOTYPE
_VECTOR.parent = _NOTYPE
_MESH.parent = _NOTYPE
_CELL.parent = _NOTYPE
_FACE.parent = _NOTYPE
_EDGE.parent = _NOTYPE
_VERTEX.parent = _NOTYPE
_TOPOSET.parent = _NOTYPE
--
_NUM.children = {_INT, _FLOAT}
_INT.parent = _NUM
_FLOAT.parent = _NUM

------------------------------------------------------------------------------

--[[ Semantic checking called from here
--]]
function check(luaenv, kernel_ast)

--	print("**** Untyped AST")
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

	local function set_type(lhsobj, rhsobj)
		if lhsobj.objtype == _NOTYPE then
			lhsobj.objtype = rhsobj.objtype
			lhsobj.scope = rhsobj.scope
			lhsobj.elemtype = rhsobj.elemtype
			lhsobj.size = rhsobj.size
			lhsobj.defn.node_type = rhsobj.objtype.name
		end
	end

	------------------------------------------------------------------------------

	--[[ Cases not handled
	--]]
	function ast.AST:check()
		print("To implement semantic checking for", self.kind)
		diag:reporterror(self, "No known method to typecheck"..self.kind)
	end

	------------------------------------------------------------------------------

	--[[ Block
	--]]
	function ast.Block:check()
		-- statements
		for id, node in ipairs(self.children) do
			node:check()
		end
	end

	------------------------------------------------------------------------------

	--[[ Statements
	--]]
	function ast.Statement:check()
		-- not needed
		env:enterblock()
		for id, node in ipairs(self.children) do
			node:check()
		end
		env:leaveblock()
	end

	function ast.IfStatement:check()
		-- condblock and block
		for id, node in ipairs(self.children) do
			env:enterblock()
			node:check()
			env:leaveblock()
		end
	end

	function ast.WhileStatement:check()
		-- condition (expression) and block
		env:enterblock()
		local condobj = self.children[1]:check()
		if not conform(_BOOL, condobj.objtype) then
			diag:reporterror(self, 
				"Expected boolean value for while statement condition")
		else
			env:enterblock()
			self.children[2]:check()
			env:leaveblock()
		end
		env:leaveblock()
	end

	function ast.DoStatement:check()
		-- block
		env:enterblock()
		for id, node in ipairs(self.children) do
			node:check()
		end
		env:leaveblock()
	end

	--TODO: discuss repeat statement semantics
	function ast.RepeatStatement:check()
		-- condition expression, block
	end

	function ast.Assignment:check()
		-- lhs, rhs expressions
		local lhsobj = self.children[1]:check()
		local rhsobj = self.children[2]:check()
		local validassgn = conform(lhsobj.objtype, rhsobj.objtype)
		if not validassgn then
			diag:reporterror(self,"Inferred RHS type ", rhsobj.objtype.name,
			" does not conform to inferred LHS type ", lhsobj.objtype.name)
		else
			set_type(lhsobj, rhsobj)
			self.node_type = rhsobj.objtype.name
		end
		return rhsobj
		-- TODO: add type information to the environment
	end

	function ast.InitStatement:check()
		-- name, expression
		-- TODO: should redefinitions be allowed?
		local nameobj = ObjType:new()
		nameobj.defn = self.children[1]
		local varname = self.children[1].children[1]
		local rhsobj = self.children[2]:check()
		set_type(nameobj, rhsobj)
		env:localenv()[varname] = nameobj
		self.node_type = nameobj.objtype.name
		return nameobj
	end

	function ast.DeclStatement:check()
		-- name
		-- TODO: should redefinitions be allowed?
		local nameobj = ObjType:new()
		nameobj.defn = self.children[1]
		local varname = self.children[1].children[1]
		env:localenv()[varname] = nameobj
		self.node_type = nameobj.objtype.name
		return nameobj
	end

	-- TODO: discuss for statements
	function ast.NumericFor:check()
	end

	function ast.GenericFor:check()
        env:enterblock()
        local setobj = self.children[2]:check()
        local itobj = ObjType:new()
        itobj.defn = self.children[1]
        itobj.objtype = setobj.elemtype
        itobj.scope = _LISZT_STR
        itobj.defn.node_type = setobj.elemtype.name
        local varname = self.children[1].children[1]
        env:localenv()[varname] = itobj
--        local forobj = self.children[3]:check()
        env:leaveblock()
--        return forobj
	end

	function ast.Break:check()
		-- nothing needed
	end

	function ast.CondBlock:check()
		-- condition (expression), block
		env:enterblock()
		local condobj = self.children[1]:check()
		if not conform(_BOOL, condobj.objtype) then
			diag:reporterror(self, "Expected boolean value here")
		else
			env:enterblock()
			self.children[2]:check()
			env:leaveblock()
		end
		env:leaveblock()
	end

	------------------------------------------------------------------------------

	--[[ Expressions
	--]]
	function ast.Expression:check()
	end

	-- expression helper functions
	local function match_num_vector(node, leftobj, rightobj)
		local binterms = conform(_NUM, leftobj.objtype) and
			conform(_NUM, rightobj.objtype)
		if not binterms then
			binterms = conform(_VECTOR, leftobj.objtype) and
				conform(_VECTOR, rightobj.objtype)
		end
		binterms = binterms or (conform(_NUM, leftobj.elemtype)
			and conform(_NUM, rightobj.elemtype))
		if not binterms then
			diag:reporterror(node,
				"Atleast one of the terms is not ",
				"a number or a vector of numbers")
			return false
		elseif not (conform(leftobj.elemtype, rightobj.elemtype)
			or conform(rightobj.elemtype, leftobj.elemtype)) then
			diag:reporterror(node, "Mismatch between element types ",
				"for the two terms")
			return false
		elseif leftobj.size ~= rightobj.size then
			diag:reporterror(node, "Mismatch in size of two terms")
			return false
		else
			return true
		end
	end

	-- binary expressions
	function ast.BinaryOp:check()
		local leftobj = self.children[1]:check()
		local rightobj = self.children[3]:check()
		local op = self.children[2]
		local exprobj = ObjType:new()
		if op == '+' or op == '-' or op == '*' or op == '/' then
			if match_num_vector(self, leftobj, rightobj) then
				if conform(leftobj.objtype, rightobj.objtype) then
					exprobj.objtype = leftobj.objtype
					exprobj.elemtype = leftobj.objtype
				else
					exprobj.objtype = rightob.objtype
					exprobj.elemtype = rightobj.objtype
				end
			exprobj.size = leftobj.size
			end
		--TODO: discuss semantics of ^
		elseif op == '^' then
		elseif op == '<=' or op == '>=' or op == '<' or op == '>' then
			if conform(_NUM, leftobj.objtype) and 
				conform(_NUM, rightobj.objtype) then
				exprobj.objtype = _BOOL
				exprobj.size = 1
			else
				diag:reporterror(self, "One of the terms compared using \'",
				op, "\' is not a number")
			end
		elseif op == '==' or op == '~=' then
			exprobj.objtype = _BOOL
			exprobj.size = 1
		elseif op == 'and' or op == 'or' then
			if conform(_BOOL, leftobj.objtype) and 
				conform(_BOOL, rightobj.objtype) then
				exprobj.objtype = _BOOL
				exprobj.size = 1
			else
				diag:reporterror(self, "One of the terms in the \' ", op,
					"\' expression is not a boolean value")
			end
		else
			diag:reporterror(self, "Unknown operator \'", op, "\'")
		end
		self.node_type = exprobj.objtype.name
		return exprobj
	end

	------------------------------------------------------------------------------

	--[[ TODO: Misc
	--]]
	function ast.LValue:check()
        print("Unreconginzed LValue, yet to implement type checking")
	end

	function ast.UnaryOp:check()
	end

	function ast.Tuple:check()
	end

	function ast.TableLookup:check()
        local tableobj = ObjType:new()
        tableobj.defn = self
        -- LHS could be another LValue, or name
        local lhsobj = self.children[1]:check()
        -- RHS is a member of the LHS
        local member = self.children[3].children[1]
        local luaval = lhsobj.luaval[member]
        if luaval == nil then
            diag:reporterror(self, "LHS value does not have member ", member)
        else
            if not lua_to_liszt(luaval, tableobj) then
                diag:reporterror(self,
                "Cannot convert the lua value to a liszt value")
            end
        end
        self.node_type = tableobj.objtype.name
        return tableobj
	end

	function ast.Call:check()
	end

	------------------------------------------------------------------------------

	--[[ Variables
	--]]

	-- TODO: infer liszt type for the lua variable
	function lua_to_liszt(luav, nameobj)
		nameobj.scope = _LUA_STR
        nameobj.luaval = luav
		if type(luav) == _TAB_STR then
            print("Lua variable of type ", luav.kind)
			if luav.kind == _VECTOR_STR then
				nameobj.objtype = _VECTOR
				nameobj.elemtype = _INT
				nameobj.size = luav.size
				return true
            elseif luav.kind == _MESH_STR then
                nameobj.objtype = _MESH
                return true
            elseif luav.kind == _CELL_STR then
                nameobj.objtype = _CELL
                return true
            elseif luav.kind == _FACE_STR then
                nameobj.objtype = _FACE
                return true
            elseif luav.kind == _EDGE_STR then
                nameobj.objtype = _EDGE
                return true
            elseif luav.kind == _VERTEX_STR then
                nameobj.objtype = _VERTEX
                return true
            elseif luav.kind == _TOPOSET_STR then
                nameobj.objtype = _TOPOSET
                nameobj.elemtype = luav.elemtype
                return true
            end
        else
			return false
		end
	end

	function ast.Name:check()
		local nameobj = ObjType:new()
        nameobj.defn = self
		local locv = env:localenv()[self.children[1]]
		if locv then
			-- if liszt local variable, type stored in environment
			nameobj = locv
		else
			local luav = env:luaenv()[self.children[1]]
			if not luav then
				diag:reporterror(self, "Variable \'" .. 
					self.children[1] .. "\' is not defined")
			elseif not lua_to_liszt(luav, nameobj) then
				diag:reporterror(self,
				"Cannot convert the lua value to a liszt value")
			end
		end
		self.node_type = nameobj.objtype.name
		return nameobj
	end

	function ast.Number:check()
		local numobj = ObjType:new()
		numobj.objtype = _NUM
		numobj.elemtype = _NUM
		numobj.size = 1
		self.node_type = numobj.objtype.name
		return numobj
	end

	function ast.Bool:check()
		local boolobj = ObjType:new()
		boolobj.objtype = _BOOL
		boolobj.elemtype = _NUM
		boolobj.size = 1
		self.node_type = boolobj.objtype.name
		return boolobj
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

	print("**** Typed AST")
	terralib.tree.printraw(kernel_ast)

	diag:finishandabortiferrors("Errors during typechecking liszt", 1)

end
