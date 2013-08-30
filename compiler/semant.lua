module(... or 'semant', package.seeall)

ast = require("ast")

--local DEBUG_PRINT = function (...) print(...) end
local DEBUG_PRINT = function (...) end
           
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
_NOT_LUA_STR = 'notlua'

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
_FIELD_STR   = 'field'
_ELEM_STR    = 'elem'

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
-- generic topological element type
_ELEM =
{
	name     = _ELEM_STR,
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
	name     = _FIELD_STR,
	parent   = {},
	children = {}
}

--[[ Tables for simplifying semantic checking logic: ]]
local strToTopoType = {
	[_MESH_STR]   = _MESH,
	[_CELL_STR]   = _CELL,
	[_FACE_STR]   = _FACE,
	[_EDGE_STR]   = _EDGE,
	[_VERTEX_STR] = _VERTEX
}

local strToElemType = {
	--[_ELEM_STR]   = _ELEM,
	[_CELL_STR]   = _CELL,
	[_FACE_STR]   = _FACE,
	[_EDGE_STR]   = _EDGE,
	[_VERTEX_STR] = _VERTEX
}

-- valid types for liszt vectors:
local strToVectorType = {
	[_INT_STR]   = _INT,
	[_FLOAT_STR] = _FLOAT,
	--[_BOOL_STR]  = _BOOL,
}

local luaToVectorType = {
	[int]   = _INT,
	[float] = _FLOAT
}

-- integral data types of liszt:
local strToIntegralFieldType = {
	[_INT_STR]   = _INT,
	[_FLOAT_STR] = _FLOAT,
	[_BOOL_STR]  = _BOOL,
}

local ObjType = 
{
    -- base type of the object
	objtype = _NOTYPE,

    -- if object consists of elements, then type of elements
    -- vector - integral type of vector components
    -- field  - type of stored field data
	elemtype = _NOTYPE,

    -- if object is over a topological set, then the corresponding topological
    -- element (fields, boundary sets, toposets)
    topotype = _NOTYPE,

	-- scope (liszt_str is a kernel temporary)
	scope = _LISZT_STR,

	-- used for vectors
	size = 0,

	-- object referred to
    luaval = _NOT_LUA_STR,

	-- ast node which has the definition
	defn = {}
}

function ObjType:new()
	return setmetatable({}, {__index = self})
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
    _ELEM,
    _TOPOSET,
    _FIELD
}

_NUM.parent     = _NOTYPE
_BOOL.parent    = _NOTYPE
_VECTOR.parent  = _NOTYPE
_MESH.parent    = _NOTYPE
_TOPOSET.parent = _NOTYPE
_FIELD.parent   = _NOTYPE
--
_NUM.children = {_INT, _FLOAT}
_INT.parent   = _NUM
_FLOAT.parent = _NUM

_ELEM.children = { _CELL, _FACE, _EDGE, _VERTEX }
_CELL.parent    = _ELEM
_FACE.parent    = _ELEM
_EDGE.parent    = _ELEM
_VERTEX.parent  = _ELEM


------------------------------------------------------------------------------

--[[ Semantic checking called from here
--]]
function check(luaenv, kernel_ast)

	-- environment for checking variables and scopes
	local env  = terralib.newenvironment(luaenv)

	local diag = terralib.newdiagnostics()

	------------------------------------------------------------------------------

	--[[ Check if lhstype conforms to rhstype
	--]]
	local function conforms(lhstype, rhstype)
		if lhstype == _TR then
			return true
		elseif rhstype == lhstype then
			return true
		elseif rhstype == _TR then
			return false
		-- float ops with ints or numbers get casted to float
		elseif lhstype == _FLOAT and (rhstype == _INT or rhstype == _NUM) then
			return true
		else
			return conforms(lhstype, rhstype.parent)
		end
	end

    local function is_valid_type(dtype, node)
    	if node == nil then node = _NOTYPE end
        if dtype == node.name then
            return true
        else
            if node.children ~= nil then
                for child in node.children do
                    is_valid_type(dtype, child)
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
			lhsobj.defn.node_type = lhsobj
		end
	end

	------------------------------------------------------------------------------

	--[[ Cases not handled
	--]]
	function ast.AST:check()
		print("To implement semantic checking for", self.kind)
		diag:reporterror(self, "No known method to typecheck "..self.kind)
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
        self.node_type = blockobj
        return blockobj
	end

	------------------------------------------------------------------------------

	--[[ Statements
	--]]
	function ast.Statement:check()
		error("Unimplemented semantic checking for " .. self.kind)
	end

	function ast.IfStatement:check()
		-- condblock and block
		for id, node in ipairs(self.children) do
			env:enterblock()
			self.node_type = node:check()
			env:leaveblock()
		end
		return self.node_type
	end

	function ast.WhileStatement:check()
		-- condition (expression) and block
		local condobj = self.children[1]:check()
		if condobj and not conforms(_BOOL, condobj.objtype) then
			diag:reporterror(self, 
			"Expected boolean value for while statement condition")
		end
		env:enterblock()
		self.node_type = self.children[2]:check()
		env:leaveblock()
        return self.node_type
	end

	function ast.DoStatement:check()
		-- block
		env:enterblock()
		for id, node in ipairs(self.children) do
			self.node_type = node:check()
		end
		env:leaveblock()
        return self.node_type
	end

	function ast.RepeatStatement:check()
		-- condition expression, block
		env:enterblock()
		self.node_type = self.children[2]:check()

		local condobj = self.children[1]:check()
		if condobj and not conforms(_BOOL, condobj.objtype) then
			diag:reporterror(self,
			"Expected boolean value for repeat statement condition")
		end
		env:leaveblock()
		return self.node_type
	end

	function ast.ExprStatement:check()
		self.node_type = self.children[1]:check()
		return self.node_type
	end

	function ast.Assignment:check()
		-- lhs, rhs expressions
		local lhsobj = self.children[1]:check()
		local rhsobj = self.children[2]:check()
		if lhsobj == nil or rhsobj == nil then
			return nil
		end
		-- only those lua objects that have belong to liszt type will be allowed
		if not lhsobj.luav == _NOT_LUA_STR then
			if not type(luav) == _TAB_STR or luav.isglobal then
				diag:reporterror(self, "Can not write to ai value of non liszt type")
			end
			return nil
		end
		-- disallow writes to topological sets/ elements/ fields
		-- allow writes to only scalars and field values
		if not (lhsobj.objtype == _VECTOR or conforms(lhsobj.objtype, _NUM)) then
			diag::reporterror(self, "Can not write to ", rhs.objtype.name)
			return nil
		end
		local validassgn = conforms(lhsobj.objtype, rhsobj.objtype)
		if not validassgn then
			diag:reporterror(self, "Inferred RHS type ", rhsobj.objtype.name,
			" does not conform to inferred LHS type ", lhsobj.objtype.name,
			" in the assignment expression")
			return nil
		end
		set_type(lhsobj, rhsobj)
		self.node_type = lhsobj
		DEBUG_PRINT(self.children[1].children[1] .. " (node type: " .. self.kind .. ") is of type " .. self.node_type.objtype.name)
		return self.node_type
	end

	function ast.InitStatement:check()
		-- name, expression
		self.node_type       = ObjType:new()
		self.node_type.defn  = self.children[1]
		local varname        = self.children[1].children[1]
		local rhsobj         = self.children[2]:check()

		if rhsobj == nil then
			return nil
		end

		set_type(self.node_type, rhsobj)
		env:localenv()[varname] = self.node_type
		DEBUG_PRINT(self.children[1].children[1] .. " (node type: " .. self.kind .. ") is of type " .. self.node_type.objtype.name)
		return self.node_type
	end

	function ast.DeclStatement:check()
		-- name
		self.node_type = ObjType:new()
		nameobj.defn  = self.children[1]
		local varname = self.children[1].children[1]
		env:localenv()[varname] = self.node_type
		return self.node_type
	end

	function ast.NumericFor:check()
		for i = 2, #self.children-1 do
			local exprobj = self.children[i]:check()
			if exprobj == nil or not conforms(_NUM, exprobj.objtype) then
				diag:reporterror(self, "Expected a number for defining the iterator")
			end
		end

		local itobj          = ObjType:new()
		itobj.defn           = self.children[1]
		itobj.objtype        = _NUM
		itobj.elemtype       = _NUM
		itobj.size           = 1
		itobj.scope          = _LISZT_STR
		itobj.defn.node_type = itobj

		env:enterblock()
		local varname = self.children[1].children[1]
		env:localenv()[varname] = itobj
		self.node_type = self.children[#self.children]:check()
		env:leaveblock()
		return self.node_type
	end

	function ast.GenericFor:check()
        local setobj = self.children[2]:check()

        -- TODO: is some kind of error checking supposed to be here?
		if setobj == nil then
		end

        local itobj          = ObjType:new()
        itobj.defn           = self.children[1]
        itobj.objtype        = setobj.topotype
        itobj.scope          = _LISZT_STR
        itobj.defn.node_type = setobj.data_type

        env:enterblock()
        local varname = self.children[1].children[1]
        env:localenv()[varname] = itobj
        self.node_type = self.children[3]:check()
        env:leaveblock()
        return self.node_type
	end

	function ast.Break:check()
		-- nothing needed
	end

	function ast.CondBlock:check()
		-- condition (expression), block
		local condobj = self.children[1]:check()
		if condobj and not conforms(_BOOL, condobj.objtype) then
			diag:reporterror(self, "Expected boolean value here")
		end

		env:enterblock()
		self.node_type = self.children[2]:check()
		env:leaveblock()

		return self.node_type
	end

	------------------------------------------------------------------------------

	--[[ Expressions
	--]]
	function ast.Expression:check()
		error("Semantic checking has not been implemented for expression type " .. self.kind)
	end

	-- expression helper functions
	local function vector_length_matches(node, leftobj, rightobj)
		if leftobj == nil or rightobj == nil then
			return false
		end
		local binterms = conforms(_NUM, leftobj.objtype) and
			conforms(_NUM, rightobj.objtype)
		if not binterms then
			binterms = conforms(_VECTOR, leftobj.objtype) and
				conforms(_VECTOR, rightobj.objtype)
		end
		binterms = binterms or (conforms(_NUM, leftobj.elemtype)
			and conforms(_NUM, rightobj.elemtype))
		if not binterms then
			diag:reporterror(node,
				"At least one of the terms is not ",
				"a number or a vector of numbers")
			return false
		elseif  not (conforms(leftobj.elemtype, rightobj.elemtype)
			     or  conforms(rightobj.elemtype, leftobj.elemtype)) then
			diag:reporterror(node, "Mismatch between operand types of binary expression")
			return false
		elseif leftobj.size ~= rightobj.size then
			diag:reporterror(node, "Mismatch of vector length of operands",
							       " in binary expression")
			return false
		else
			return true
		end
	end

    --[[ Logic tables for binary expression checking: ]]--
    -- vector ops can take either numbers or vectors as arguments
    local isVecOp = {
       ['+'] = true,
       ['-'] = true,
       ['*'] = true,
       ['/'] = true
    }

    local isNumOp = {
    	['^'] = true
	}

    local isCompOp = {
       ['<='] = true,
       ['>='] = true,
       ['>']  = true,
       ['<']  = true
    }

    local isEqOp = {
       ['=='] = true,
       ['~='] = true
    }

    local isBoolOp = {
       ['and'] = true,
       ['or']  = true,
    }

	-- binary expressions
	function ast.BinaryOp:check()
		local leftobj  = self.children[1]:check()
		local rightobj = self.children[3]:check()
		local op = self.children[2]
		local exprobj = ObjType:new()

		if leftobj == nil or rightobj == nil then
			return nil
		end

		if isVecOp[op] then
			if vector_length_matches(self, leftobj, rightobj) then
				if conforms(leftobj.objtype, rightobj.objtype) then
					exprobj.objtype  = leftobj.objtype
					exprobj.elemtype = leftobj.objtype
				else
					exprobj.objtype  = rightobj.objtype
					exprobj.elemtype = rightobj.objtype
				end
				exprobj.size = leftobj.size
			end
			-- BUG/TODOS: what if vector lengths don't match?
			-- int op float -> float
			-- vector * constant or vector / constant should be valid expressions, but
			-- vector(n) + vector(m) is not valid (m != n)
			-- also vector * vector and vector / vector don't make sense

		elseif isNumOp[op] then
			if not conforms(_NUM, leftobj.objtype) then
				diag:reporterror(self, "Expected a number here")
			end
			if not conforms(_NUM, rightobj.objtype) then
				diag:reporterror(self, "Expected a number here")
			end
            -- int * int -> int
			if conforms(leftobj.objtype, _INT) and conforms(rightobj.objtype, _INT) then
				exprobj.objtype  = _INT
				exprobj.elemtype = _INT
            -- float * Num -> float
			elseif conforms(leftobj.objtype, _FLOAT) or conforms(leftobj, _FLOAT) then
				exprobj.objtype  = _FLOAT
				exprobj.elemtype = _FLOAT
			else
				exprobj.objtype  = _NUM
				exprobj.elemtype = _NUM
			end
			exprobj.size = 1

		elseif isCompOp[op] then
			if conforms(_NUM, leftobj.objtype) and 
				conforms(_NUM, rightobj.objtype) then
				exprobj.objtype = _BOOL
				exprobj.size    = 1
			else
				diag:reporterror(self, "One of the terms compared using \'",
				op, "\' is not a number")
			end

		elseif isEqOp[op] then
			exprobj.objtype = _BOOL
			exprobj.size    = 1

		elseif isBoolOp[op] then
			if conforms(_BOOL, leftobj.objtype) and 
				conforms(_BOOL, rightobj.objtype) then
				exprobj.objtype = _BOOL
				exprobj.size    = 1
			else
				diag:reporterror(self, "One of the terms in the \' ", op,
					"\' expression is not a boolean value")
			end

		else -- not a recognized operator (?)
			diag:reporterror(self, "Unknown operator \'", op, "\'")
		end
    	self.node_type = exprobj
    	DEBUG_PRINT(op .. " " .. self.kind .. " of type " .. self.node_type.objtype.name)
		return exprobj
	end

	function ast.UnaryOp:check()
		local op = self.children[1]
		local exprobj = self.children[2]:check()
		if exprobj == nil then
			return nil
		end
		if op == 'not' then
			if not conforms(_BOOL, exprobj.objtype) then
				diag:reporterror(self, "\"not\" operator expects a boolean expression")
				return nil
			else
				self.node_type = exprobj
				return exprobj
			end
		elseif op == '-' then
			local binterms = conforms(_NUM, exprobj.objtype)
			if not binterms then
				binterms = conforms(_VECTOR, exprobj.objtype) and
				conforms(_NUM, exprobj.elemtype)
			end
			if not binterms then
				diag:reporterror(self, "Unary minus expects a number or a vector of numbers")
				return nil
			else
				self.node_type = exprobj
				DEBUG_PRINT(self.kind .. " of type " .. self.node_type.objtype.name)
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
	function ast.Tuple:check()
		for i, node in ipairs(self.children) do
			node:check()
		end
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
		self.node_type = argobj
		return argobj
	end

	function ast.TableLookup:check()
        -- LHS could be another LValue, or name
        local lhsobj = self.children[1]:check()
		if lhsobj == nil then
			return nil
		end
        local tableobj = ObjType:new()
        tableobj.defn  = self
        -- RHS is a member of the LHS
        local member = self.children[3].children[1]
        local luaval = lhsobj.luaval[member]
        if luaval == nil then
            diag:reporterror(self, "LHS value does not have member ", member)
			return nil
        else
            if not lua_to_liszt(luaval, tableobj) then
                diag:reporterror(self,
                "Cannot convert the lua value to a liszt value")
			return nil
            end
        end
        self.node_type = tableobj
        return tableobj
	end

	function ast.Call:check()
		-- call name can be a field only in current implementation
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
				self.node_type = callobj.elemtype:new()
				return self.node_type
			elseif argobj.objtype.name == _ELEM_STR then
				-- infer type of argument to field topological type
				argobj.objtype        = callobj.topotype
				argobj.defn.node_type = argobj.objtype.name

				-- return object that is field data type
				self.node_type = callobj.elemtype:new()
				return self.node_type

			else
				diag:reporterror(self,
				"Field over ", callobj.topotype.name,
				"s is indexed by a ", argobj.objtype.name)
				return nil 
			end
		else
			-- TODO: How should function calls be allowed??
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
		DEBUG_PRINT("lua to liszt type(luav): " .. type(luav) .. "value: " .. tostring(luav))
		if (type(luav) == 'table') then
			DEBUG_PRINT("  luav.kind: " .. tostring(luav.kind))
			DEBUG_PRINT("  luav.data_type: " .. tostring(luav.data_type))
		end
		nameobj.scope = _LUA_STR
        nameobj.luaval = luav
		if type(luav) == _TAB_STR then

			-- terra globals
			if luav.isglobal then
				if luaToVectorType[luav.type] then
					nameobj.objtype  = luaToVectorType[luav.type]
					nameobj.elemtype = luaToVectorType[luav.type]
					nameobj.size     = 1
					return true
				else
					-- TODO:
					-- want to use diag:reporterror here, but don't have line # info
					-- print("Liszt does not yet support terra type " .. tostring(luav.type))
					return false
				end

			-- vectors
			elseif luav.kind == _VECTOR_STR then
				if luaToVectorType[luav.data_type] then
					nameobj.objtype  = _VECTOR
					nameobj.elemtype = luaToVectorType[luav.data_type]
					nameobj.size     = luav.size
					return true
				else
					return false
				end

			-- mesh, cell, face, edge, vertex
            elseif strToTopoType[luav.kind] then
                nameobj.objtype = strToTopoType[luav.kind]
                return true

			-- topological set
            elseif luav.kind == _TOPOSET_STR then
                nameobj.objtype = _TOPOSET
                -- cell, face, edge, vertex
                if strToElemType[luav.data_type] then
                    nameobj.topotype = strToElemType[luav.data_type]
                    return true
                end
				return false

			-- field
            elseif luav.kind == _FIELD_STR then
                nameobj.objtype = _FIELD
                local dobj    = luav.data_type
				local elemobj = ObjType:new()
				elemobj.scope = _LUA_STR

				-- determine field element type:
				if strToIntegralFieldType[dobj.obj_type] then
					DEBUG_PRINT("  field is of integral type " .. tostring(dobj.obj_type))
					elemobj.objtype  = strToIntegralFieldType[dobj.obj_type]
					elemobj.elemtype = strToIntegralFieldType[dobj.obj_type]
					elemobj.size     = 1
				elseif dobj.obj_type == _VECTOR_STR then
					elemobj.objtype = _VECTOR
					if strToVectorType[dobj.elem_type] then
						elemobj.elemtype = strToVectorType[dobj.elem_type]
						elemobj.size = dobj.size
						DEBUG_PRINT("  field is a vector of type " .. tostring(dobj.elem_type) .. ", size " .. tostring(elemobj.size))
					else
						return false
					end
				else -- field element type must be an integral type or a vector!
					return false
				end
				nameobj.elemtype = elemobj

				-- determine field topo type:
				if strToElemType[luav.topo_type] then
					nameobj.topotype = strToElemType[luav.topo_type]
				else
					return false
				end
				return true

			-- table does not represent a liszt type
            else
                return false
            end

        -- for referenced lua values that may be used as an int or a float
        elseif (type(luav) == _NUM_STR) then
        	nameobj.objtype  = _NUM
        	nameobj.elemtype = _NUM
        	nameobj.size     = 1
        	return true
        else
        	-- not number / table
        	return false
		end
	end

	function ast.Name:check()
		local locv = env:localenv()[self.children[1]]
		if locv then
			-- if liszt local variable, type stored in environment
			self.node_type = locv
			return locv
		end
		
		local luav = env:luaenv()[self.children[1]]
		if not luav then
			diag:reporterror(self, "Variable \'" .. 
				self.children[1] .. "\' is not defined")
			return nil
		end

		local nameobj = ObjType:new()
    	nameobj.defn  = self

		if not lua_to_liszt(luav, nameobj) then
			diag:reporterror(self,
			"Cannot convert the lua value to a liszt value")
			return nil
		end

    	self.node_type = nameobj
    	DEBUG_PRINT(self.children[1] .. " (node type: " .. self.kind .. ") is of type " .. self.node_type.objtype.name)
		return nameobj
	end

	function ast.Number:check()
		local numobj    = ObjType:new()
		-- These numbers are stored in lua as floats, so we might as well use them that way
		-- until the terra lexer gets the ability to separately parse ints and floats
		numobj.objtype  = _FLOAT
		numobj.elemtype = _FLOAT
		numobj.size     = 1
		self.node_type  = numobj
		DEBUG_PRINT(self.kind .. " of type " .. self.node_type.objtype.name .. ", value: " .. self.children[1])
		return numobj
	end

	function ast.Bool:check()
		local boolobj    = ObjType:new()
		boolobj.objtype  = _BOOL
		boolobj.elemtype = _NUM
		boolobj.size     = 1
		self.node_type   = boolobj
		return boolobj
	end

	------------------------------------------------------------------------------

	-- begin actual typechecking
	diag:begin()
	env:enterblock()
	local param = kernel_ast.children[1]
	local block = kernel_ast.children[2]

	local paramobj    = ObjType:new()
	paramobj.objtype  = _ELEM
	paramobj.elemtype = _ELEM
	paramobj.size     = 1
	paramobj.scope    = _LISZT_STR
	paramobj.defn     = param
	param.node_type   = paramobj

	env:localenv()[param.children[1]] = paramobj

	for id, node in ipairs(kernel_ast.children) do
		if (node.kind ~= nil) then
			node:check()
		end
	end
	env:leaveblock()

--	print("**** Typed AST")
--	terralib.tree.printraw(kernel_ast)

	diag:finishandabortiferrors("Errors during typechecking liszt", 1)

end
