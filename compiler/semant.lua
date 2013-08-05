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
_LUA_STR     = 'lua'
_LISZT_STR   = 'liszt'

_NOTYPE_STR  = 'notype'
_INT_STR     = 'int'
_FLOAT_STR   = 'float'
_BOOL_STR    = 'bool'
_VECTOR_STR  = 'vector'
_NUM_STR     = 'number'
_TAB_STR     = 'table'
_MESH_STR    = 'mesh'
_CELL_STR    = 'cell'
_FACE_STR    = 'face'
_EDGE_STR    = 'edge'
_VERTEX_STR  = 'vertex'
_TOPOSET_STR = 'toposet'
_FIELD_STR = 'field'

-- root variable type
_NOTYPE = 
{
	name     = _NOTYPE_STR,
	parent   = {},
	children = {}
}
-- liszt variable type
_INT = 
{
	name     = _INT_STR,
	parent   = {},
	children = {}
}
-- liszt variable type
_FLOAT = 
{
	name     = _FLOAT_STR,
	parent   = {},
	children = {}
}
-- liszt variable type
_BOOL = 
{
	name     = _BOOL_STR,
	parent   = {},
	children = {}
}
-- liszt variable type
_VECTOR = 
{
	name     = _VECTOR_STR,
	parent   = {},
	children = {}
}
-- number type
_NUM = 
{
	name     = _NUM_STR,
	parent   = {},
	children = {}
}
-- table type
_TAB  =
{
	name     = _TAB_STR,
	parent   = {},
	children = {}
}
-- mesh type
_MESH  =
{
	name     = _MESH_STR,
	parent   = {},
	children = {}
}
-- cell type
_CELL  =
{
	name     = _CELL_STR,
	parent   = {},
	children = {}
}
-- face type
_FACE  =
{
	name     = _FACE_STR,
	parent   = {},
	children = {}
}
-- edge type
_EDGE  =
{
	name     = _EDGE_STR,
	parent   = {},
	children = {}
}
-- vertex type
_VERTEX  =
{
	name     = _VERTEX_STR,
	parent   = {},
	children = {}
}
-- topological set type
_TOPOSET  =
{
	name     = _TOPOSET_STR,
	parent   = {},
	children = {}
}
-- field type
_FIELD  =
{
	name = _FIELD_STR,
	parent = {},
	children = {}
}

local ObjType = 
{
    -- type of the object
	objtype = _NOTYPE,
    -- if object consists of elements, then type of elements
	elemtype = _NOTYPE,
    -- if object is over a topological set, then the corresponding topological
    -- element
    topotype = _NOTYPE,
	-- scope
	scope = _LISZT_STR,
	-- size
	size = 0,
	-- lua value
    luaval = {},
	-- ast node which has the definition
	defn = {}
}

function ObjType:new()
	local newtype = {}
	setmetatable(newtype, {__index = self})
	newtype.objtype = self.objtype
	newtype.elemtype = self.elemtype
    newtype.topotype = self.elemtype
	newtype.scope = self.scope
	newtype.size = self.size
	newtype.luaval = self.luaval
	newtype.defn = self.defn
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
    _TOPOSET,
    _FIELD
}
_NUM.parent     = _NOTYPE
_BOOL.parent    = _NOTYPE
_VECTOR.parent  = _NOTYPE
_MESH.parent    = _NOTYPE
_CELL.parent    = _NOTYPE
_FACE.parent    = _NOTYPE
_EDGE.parent    = _NOTYPE
_VERTEX.parent  = _NOTYPE
_TOPOSET.parent = _NOTYPE
_FIELD.parent = _NOTYPE
--
_NUM.children = {_INT, _FLOAT}
_INT.parent   = _NUM
_FLOAT.parent = _NUM

------------------------------------------------------------------------------

--[[ Semantic checking called from here
--]]
function check(luaenv, kernel_ast)

	print("**** Untyped AST")
	terralib.tree.printraw(kernel_ast)
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

    local function is_valid_type(dtype)
        if is_valid_type_rec(_NOTYPE, dtype) then
            return true
        else
            return false
        end
    end

    local function is_valid_type_rec(node, dtype)
        if dtype == node.name then
            return true
        else
            if node.children ~= nil then
                for child in node.children do
                    is_valid_type_rec(child, dtype)
                end
            else
                return false
            end
        end
    end

	local function set_type(lhsobj, rhsobj)
		if lhsobj.objtype == _NOTYPE then
			lhsobj.objtype        = rhsobj.objtype
			lhsobj.scope          = rhsobj.scope
			lhsobj.elemtype       = rhsobj.elemtype
			lhsobj.size           = rhsobj.size
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
        local blockobj
		for id, node in ipairs(self.children) do
			blockobj = node:check()
		end
        if blockobj ~= nil then
            self.node_type = blockobj.objtype.name
        end
        return blockobj
	end

	------------------------------------------------------------------------------

	--[[ Statements
	--]]
	function ast.Statement:check()
		-- not needed
		env:enterblock()
        local stblock;
		for id, node in ipairs(self.children) do
			stblock = node:check()
		end
        if stblock ~= nil then
            self.node_type = stblock.objtype.name
        end
		env:leaveblock()
        return stblock
	end

	function ast.IfStatement:check()
		-- condblock and block
        local stblock;
		for id, node in ipairs(self.children) do
			env:enterblock()
			stblock = node:check()
			env:leaveblock()
		end
        if stblock ~= nil then
            self.node_type = stblock.objtype.name
        end
        return stblock
	end

	function ast.WhileStatement:check()
		-- condition (expression) and block
		env:enterblock()
        local stblock
		local condobj = self.children[1]:check()
		if condobj ~= nil then
			if not conform(_BOOL, condobj.objtype) then
				diag:reporterror(self, 
				"Expected boolean value for while statement condition")
			end
		end
		env:enterblock()
		stblock = self.children[2]:check()
		env:leaveblock()
		if stblock ~= nil then
            self.node_type = stblock.objtype.name
        end
		env:leaveblock()
        return stblock
	end

	function ast.DoStatement:check()
		-- block
		env:enterblock()
        local stblock
		for id, node in ipairs(self.children) do
			stblock = node:check()
		end
        if stblock ~= nil then
            self.node_type = stblock.objtype.name
        end
		env:leaveblock()
        return stblock
	end

	function ast.RepeatStatement:check()
		-- condition expression, block
		env:enterblock()
		local stblock = self.children[2]:check()
		if stblock ~= nil then
			self.node_type = stblock.objtype.name
		end
		local condobj = self.children[1]:check()
		if condobj ~= nil then
			if not conform(_BOOL, condobj.objtype) then
				diag:reporterror(self,
				"Expected boolean value for repeat statement condition")
			end
		end
		env:leaveblock()
		return stblock
	end

	function ast.Assignment:check()
		-- lhs, rhs expressions
		local lhsobj = self.children[1]:check()
		local rhsobj = self.children[2]:check()
		if lhsobj == nil or rhsobj == nil then
			return nil
		end
		local validassgn = conform(lhsobj.objtype, rhsobj.objtype)
		if not validassgn then
			diag:reporterror(self,"Inferred RHS type ", rhsobj.objtype.name,
			" does not conform to inferred LHS type ", lhsobj.objtype.name,
			" in the assginment expression")
			return nil
		else
			set_type(lhsobj, rhsobj)
			self.node_type = rhsobj.objtype.name
		end
		return rhsobj
	end

	function ast.InitStatement:check()
		-- name, expression
		local nameobj = ObjType:new()
		nameobj.defn  = self.children[1]
		local varname = self.children[1].children[1]
		local rhsobj  = self.children[2]:check()
		if rhsobj == nil then
			return nil
		end
		set_type(nameobj, rhsobj)
		env:localenv()[varname] = nameobj
		self.node_type = nameobj.objtype.name
		return nameobj
	end

	function ast.DeclStatement:check()
		-- name
		local nameobj = ObjType:new()
		nameobj.defn  = self.children[1]
		local varname = self.children[1].children[1]
		env:localenv()[varname] = nameobj
		self.node_type = nameobj.objtype.name
		return nameobj
	end

	-- TODO: discuss for statements
	function ast.NumericFor:check()
		env:enterblock()
		local intexpr = 0
		local floatexpr = 0
		local otherexpr = 0
		local expr1obj = self.children[2]:check()
		if expr1obj == nil then
			diag:reporterror(self,
			"Expected a number for defining the iterator")
		else
			if not conform(_NUM, expr1obj.objtype) then
				diag:reporterror(self,
				"Expected a number for defining the iterator")
			end
			if expr1obj.objtype == _INT then
				intexpr = intexpr + 1
			elseif expr1obj.objtype == _FLOAT then
				floatexpr = floatexpr + 1
			end
		end
		local expr2obj = self.children[3]:check()
		if expr2obj == nil then
			diag:reporterror(self,
			"Expected a number for defining the iterator")
		else
			if not conform(_NUM, expr2obj.objtype) then
				diag:reporterror(self,
				"Expected a number for defining the iterator")
			end
			if expr2obj.objtype == _INT then
				intexpr = intexpr + 1
			elseif expr2obj.objtype == _FLOAT then
				floatexpr = floatexpr + 1
			end
		end
		if #self.children == 5 then
			local expr3obj = self.children[4]:check()
			if expr3obj == nil then
				diag:reporterror(self,
				"Expected a number for defining the iterator")
			else
				if not conform(_NUM, expr3obj.objtype) then
					diag:reporterror(self,
					"Expected a number for defining the iterator")
				end
				if expr3obj.objtype == _INT then
					intexpr = intexpr + 1
				elseif expr3obj.objtype == _FLOAT then
					floatexpr = floatexpr + 1
				end
			end
		end
		local itobj = ObjType:new()
		itobj.defn = self.children[1]
		itobj.objtype = setobj.elemtype
		itobj.scope = _LISZT_STR
		env:leaveblock()
	end

	function ast.GenericFor:check()
        env:enterblock()
        local setobj = self.children[2]:check()
		if setobj == nil then
			return nil
		end
        local itobj = ObjType:new()
        itobj.defn = self.children[1]
        itobj.objtype = setobj.elemtype
        itobj.scope = _LISZT_STR
        itobj.defn.node_type = setobj.data_type
        local varname = self.children[1].children[1]
        env:localenv()[varname] = itobj
        local forobj = self.children[3]:check()
        env:leaveblock()
        if forobj ~= nil then
            self.node_type = forobj.objtype.name
        end
        return forobj
	end

	function ast.Break:check()
		-- nothing needed
	end

	function ast.CondBlock:check()
		-- condition (expression), block
		env:enterblock()
		local condblock
		local condobj = self.children[1]:check()
		if condobj ~= nil then
			if not conform(_BOOL, condobj.objtype) then
				diag:reporterror(self, "Expected boolean value here")
			end
		end
		env:enterblock()
		condblock = self.children[2]:check()
		env:leaveblock()
		if condblock ~= nil then
			self.node_type = condblock.objtype.name
		end
		env:leaveblock()
		return condblock
	end

	------------------------------------------------------------------------------

	--[[ Expressions
	--]]
	function ast.Expression:check()
	end

	-- expression helper functions
	local function match_num_vector(node, leftobj, rightobj)
		if leftobj == nil or rightobj == nil then
			return false
		end
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
		if leftobj == nil or rightobj == nil then
			return nil
		end
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

	function ast.UnaryOp:check()
		local op = self.children[1]
		local exprobj = self.children[2]:check()
		if exprobj == nil then
			return nil
		end
		if op == 'not' then
			if not conform(_BOOL, exprobj.objtype) then
				diag:reporterror(self, "Expected a boolean expression here")
				return nil
			else
				self.node_type = exprobj.objtype.name
				return exprobj
			end
		elseif op == '-' then
			local binterms = conform(_NUM, exprobj.objtype)
			if not binterms then
				binterms = conform(_VECTOR, exprobj.objtype) and
				conform(_NUM, exprobj.elemtype)
			end
			if not binterms then
				diag:reporterror(self,
				"Atleast one of the terms is not ",
				"a number or a vector of numbers")
				return nil
			else
				self.node_type = exprobj.objtype.name
				return exprobj
			end
		else
			diag:reporterror(self, "Unknown unary operator \'"..op.."\'")
			return nil
		end
	end

	------------------------------------------------------------------------------

	--[[ Misc
	--]]
	function ast.LValue:check()
        print("Unreconginzed LValue, yet to implement type checking")
	end

	function ast.Tuple:check()
		-- TODO: not needed right now, but to implement functions later on
		print("Inside tuple type checking")
	end

	function ast.Tuple:index_check()
		-- type checking tuple when it should be a single argument, for
		-- instance, when indexing a field
		if #self.children ~= 1 then
			diag:reporterror(self, "Can use exactly one argument to index here")
			return nil
		end
		local argobj = self.children[1]:check()
		if argobj == nil then
			return nil
		end
		return argobj
	end

	function ast.TableLookup:check()
        -- LHS could be another LValue, or name
        local lhsobj = self.children[1]:check()
		if lhsobj == nil then
			return nil
		end
        local tableobj = ObjType:new()
        tableobj.defn = self
        -- RHS is a member of the LHS
        local member = self.children[3].children[1]
        local luaval = lhsobj.luaval[member]
        if luaval == nil then
            diag:reporterror(self, "LHS value does not have member ", member)
			return nil
        else
            if not lua_to_liszt(luaval, tableobj) then
                diag:reporterror(self,
                "Cannot convert the lua value to a liszt value @ ")
			return nil
            end
        end
        self.node_type = tableobj.objtype.name
        return tableobj
	end

	function ast.Call:check()
		-- call name can be a field only in current implementation
		print("Inside call type checking")
		local callobj = self.children[1]:check()
		if callobj == nil then
			diag:reporterror(self, "Undefined call")
			return nil
		elseif callobj.objtype.name == _FIELD_STR then
			local argobj = self.children[2]:index_check()
			if argobj == nil then
				return nil
			end
			if argobj.objtype.name == callobj.topotype.name then
				local retobj = callobj.elemtype:new()
				self.node_type = retobj.objtype.name
				return retobj
			else
				diag:reporterror(self,
				"Field over ", callobj.topotype.name,
				"s is indexed by a ", argobj.objtype.name)
				return nil 
			end
		else
			-- TOO: How should function calls be allowed??
			diag:reporterror(self, "Invalid call")
			return nil
		end
		return nil
	end

	------------------------------------------------------------------------------

	--[[ Variables
	--]]

	-- Infer liszt type for the lua variable
	function lua_to_liszt(luav, nameobj)
		nameobj.scope = _LUA_STR
        nameobj.luaval = luav
		-- TODO: scalars
		if type(luav) == _TAB_STR then
			-- vectors
			if luav.kind == _VECTOR_STR then
				nameobj.objtype = _VECTOR
				if luav.data_type == int then
					nameobj.elemtype = _INT
				elseif luav.data_type == float then
					nameobj.elemtype = _FLOAT
				else
					return false
				end
				nameobj.size = luav.size
				return true
			-- mesh
            elseif luav.kind == _MESH_STR then
                nameobj.objtype = _MESH
                return true
			-- cell
            elseif luav.kind == _CELL_STR then
                nameobj.objtype = _CELL
                return true
			-- face
            elseif luav.kind == _FACE_STR then
                nameobj.objtype = _FACE
                return true
			-- edge
            elseif luav.kind == _EDGE_STR then
                nameobj.objtype = _EDGE
                return true
			-- vertex
            elseif luav.kind == _VERTEX_STR then
                nameobj.objtype = _VERTEX
                return true
			-- topological set
            elseif luav.kind == _TOPOSET_STR then
                nameobj.objtype = _TOPOSET
                if luav.data_type == _VERTEX_STR then
                    nameobj.elemtype = _VERTEX
                elseif luav.data_type == _EDGE_STR then
                    nameobj.elemtype = _EDGE
                elseif luav.data_type == _FACE_STR then
                    nameobj.elemtype = _FACE
                elseif luav.data_type == _CELL_STR then
                    nameobj.elemtype = _CELL
                else
                    return false
                end
				return true
			-- field
            elseif luav.kind == _FIELD_STR then
                nameobj.objtype = _FIELD
                local dobj = luav.data_type
				local elemobj = ObjType:new()
				if dobj.obj_type == _INT_STR then
					elemobj.objtype = _INT
					elemobj.elemtype = _INT
					elemobj.size = 1
					elemobj.scope = _LUA_STR
				elseif dobj.obj_type == _FLOAT_STR then
					elemobj.obj_type = _FLOAT
					elemobj.elemtype = _FLOAT
					elemobj.size = 1
					elemobj.scope = _LUA_STR
				elseif dobj.obj_type == _VECTOR_STR then
					elemobj.objtype = _VECTOR
					if dobj.elem_type == _INT_STR then
						elemobj.elemtype = INT
					elseif dobj.elem_type == _FLOAT_STR then
						elemobj.elemtype = _FLOAT
					else
						return false
					end
					elemobj.size = dobj.size
					elemobj.scope = _LUA_STR
				else
					return false
				end
				nameobj.elemtype = elemobj
                local ttype = luav.topo_type
				if ttype == _CELL_STR then
					nameobj.topotype = _CELL
				elseif ttype == _FACE_STR then
					nameobj.topotype = _FACE
				elseif ttype == _EDGE_STR then
					nameobj.topotype = _EDGE
				elseif ttype == _VERTEX_STR then
					nameobj.topotype = _VERTEX
				else
					return false
				end
					return true
			-- none of the above in table
            else
                return false
            end
        else
			-- not number/ table
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
				return nil
			elseif not lua_to_liszt(luav, nameobj) then
				diag:reporterror(self,
				"Cannot convert the lua value to a liszt value")
				return nil
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

--	print("**** Typed AST")
	terralib.tree.printraw(kernel_ast)

	diag:finishandabortiferrors("Errors during typechecking liszt", 1)

end
