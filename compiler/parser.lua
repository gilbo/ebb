--[[ Module defines all of the parsing functions used to generate
an AST for Liszt code 
]]--
local exports = {}

-- Imports
package.path = package.path .. ";./compiler/?.lua;./compiler/?.t"

-- Import ast nodes, keeping out of global scope
local ast = require "ast"

local pratt = terralib.require('compiler/pratt')

-- Global precedence table
local precedence = {

	["or"]  = 0,
	["and"] = 1,

	["<"]   = 2,
	[">"]   = 2,
	["<="]  = 2,
	[">="]  = 2,
	["=="]  = 2,
	["~="]  = 2,

	["+"]   = 3,
	["-"]   = 3,

	["*"]   = 4,
	["/"]   = 4,
	["%"]   = 4,

	unary   = 5,
	--	uminus  = 5,
	--	["not"] = 5,
	--	["#"]   = 5,

	["^"]   = 6,
}

local block_terminators = {
	['end']    = 1,
	['else']   = 1,
	['elseif'] = 1,
	['until']  = 1,
	['break']  = 1,
}

--[[ Generic expression operator parsing functions: ]]--
local function leftbinaryimpl(P, lhs, isreductionop)
	local node = ast.BinaryOp:New(P)
	local op   = P:next().type
	if isreductionop and P:matches("=") then 
		node = ast.Reduce:New(P)
		node.op, node.exp = op, lhs
		return node
	else
		local rhs = P:exp(op)
		node.lhs, node.op, node.rhs = lhs, op, rhs
		return node
	end
end

local function leftbinary(P, lhs)
	return leftbinaryimpl(P, lhs, false)
end

local function leftbinaryred(P, lhs)
	return leftbinaryimpl(P, lhs, true)
end

local function rightbinary(P, lhs)
	local node = ast.BinaryOp:New(P)
	local op   = P:next().type
	local rhs  = P:exp(op, "right")
	node.lhs, node.op, node.rhs = lhs, op, rhs
	return node
end

local function unary (P)
	local node = ast.UnaryOp:New(P)
	local op = P:next().type
	local exp = P:exp(precedence.unary)
	node.op, node.exp = op, exp
	return node
end	

----------------------------------
--[[ Build Liszt Pratt parser ]]--
----------------------------------
local lang = { }

--[[ Expression parsing ]]--
lang.exp = pratt.Pratt() -- returns a pratt parser
:prefix("-",   unary)
:prefix("not", unary)

:infix("or",  precedence["or"],  leftbinaryred)
:infix("and", precedence["and"], leftbinaryred)

:infix("<",   precedence["<"],   leftbinary)
:infix(">",   precedence[">"],   leftbinary)
:infix("<=",  precedence["<="],  leftbinary)
:infix(">=",  precedence[">="],  leftbinary)
:infix("==",  precedence["=="],  leftbinary)
:infix("~=",  precedence["~="],  leftbinary)

:infix("*",   precedence['*'],   leftbinaryred)
:infix('/',   precedence['/'],   leftbinaryred)
:infix('%',   precedence['%'],   leftbinary)

:infix("+",   precedence["+"],   leftbinaryred)
:infix("-",   precedence["-"],   leftbinaryred)
:infix('^',   precedence['^'],   rightbinary)
:prefix(pratt.default, function(P) return P:simpleexp() end)

-- tuples are used to represent a sequence of comma-seperated expressions that can
-- be found in function calls and will perhaps be used in for statements
lang.tuple = function (P, exprs)
	exprs = exprs or { }
	exprs[#exprs + 1] = P:exp()
	if P:nextif(",") then
		return P:tuple(exprs)
	else 
		local tuple    = ast.Tuple:New(P)
		tuple.children = exprs
		tuple:copy_location(exprs[1])
		return tuple
	end
end

--[[ recursively checks to see if lhs is part of a field index 
or a table lookup. lhs parameter is already an LValue 
--]]
lang.lvaluehelper = function (P, lhs)
	local cur = P:cur()
	-- table indexing:
	if cur.type == '.' then
		local node = ast.TableLookup:New(P)
		local op = P:next().type
		-- check to make sure the table is being indexed by a valid name
		if not P:matches(P.name) then P:error("expected name after '.'") end
		local nodename = ast.Name:New(P)
		nodename.name = P:next().value
		node.table, node.member = lhs, nodename
		return P:lvaluehelper(node)
	end
		
	-- field index / function call?
	local open_parens = P:nextif('(')
	if open_parens then
		local node = ast.Call:New(P)
		local args = P:tuple()
		P:expectmatch(')', '(', open_parens.linenumber)
		node.func, node.params = lhs, args
		return P:lvaluehelper(node)
	end

	local open_sq_bracket = P:nextif('[')
	if open_sq_bracket then
		local node = ast.VectorIndex:New(P)
		local exp = P:exp()
		P:expectmatch(']', '[', open_sq_bracket.linenumber)
		node.vector, node.index = lhs, exp
		return P:lvaluehelper(node)
	end

	return lhs
end

--[[ This function finds the first name of the lValue, then calls
lValueHelper to recursively check to table or field lookups. 
--]]
lang.lvalue = function (P)
	if not P:matches(P.name) then
		local token = P:next()
		P:error("Expected name at " .. token.linenumber .. ":" .. token.offset)
	end
	local node = ast.Name:New(P)
	node.name = P:next().value
	P:ref(node.name)
	return P:lvaluehelper(node)
end

lang.lvalname = function (P)
	if not P:matches(P.name) then
		local token = P:next()
		P:error("Expected name at " .. token.linenumber .. ":" .. token.offset)
	end
	local node = ast.Name:New(P)
	node.name  = P:next().value
	P:ref(node.name)
	return node
end

lang.vectorliteral = function (P)
	local node = ast.VectorLiteral:New(P)
	local start = P:expect('{')
	local elems = { }

	repeat
		elems[#elems + 1] = P:exp()
	until not P:nextif(",")
	P:expectmatch("}", "{", start.linenumber)
	node.elems = elems
	return node
end

--[[  simpleexp is called when exp cannot parse the next token...
we could be looking at an LValue, Value, or parenthetically
enclosed expression  
--]]
lang.simpleexp = function(P)
	-- catch values
	-- TODO: This does something weird with integers
	if P:matches(P.number) then
		local node = ast.Number:New(P)
		node.value = P:next().value
		return node

		-- catch bools
	elseif P:matches('true') or P:matches('false') then
		local node = ast.Bool:New(P)
		node.value = P:next().type
		return node
	end

	-- catch parenthesized expressions
	local open = P:nextif('(')
	if open then
		local v = P:exp()
		P:expectmatch(")", "(", open.linenumber)
		return v
	end

	if P:matches("{") then
		return P:vectorliteral()
	end
	-- expect LValue
	return P:lvalue()
end

lang.liszt_kernel = function (P)
	-- parse liszt_kernel keyword and argument:
	P:expect("liszt_kernel")
	local kernel_node = ast.LisztKernel:New(P)

	-- parse parameter
	local open  = P:expect("(")
	local iter  = P:lvalname()
    P:expect("in")
    local set   = P:lvalue()
	P:expectmatch(")", "(", open.linenumber)

	-- parse block
	local block = P:block()
	P:expect("end")

	kernel_node.iter, kernel_node.set, kernel_node.body = iter, set, block
	return kernel_node
end

--[[ Statement Parsing ]]--
--[[ Supported statements:
- if statement
- while statement
- repeat statement
- do statement
- variable declaration
- assignment
- variable initialization
- expression statement
- TODO: for statement
--]]
lang.statement = function (P)
	-- check for initialization/declaration
	if (P:nextif("var")) then
		local name = P:lvalname()
		-- differentiate between initialization and declaration
		if (P:nextif("=")) then
			local node_init = ast.InitStatement:New(P)
			local exp = P:exp()
			node_init.ref, node_init.exp = name, exp
			return node_init
		else
			local node_decl = ast.DeclStatement:New(P)
			node_decl.ref = name
			return node_decl
		end

		--[[ if statement ]]--
	elseif P:nextif("if") then
		local node_if = ast.IfStatement:New(P)
		local node_ifc = ast.CondBlock:New(P)

		local if_blocks = { }
		local else_block = nil

		local cond = P:exp()
		P:expect("then")
		local body = P:block()
		node_ifc.cond, node_ifc.body = cond, body
		if_blocks[#if_blocks+1] = node_ifc

		-- parse all elseif clauses
		while (P:nextif("elseif")) do
			local node_elseif = ast.CondBlock:New(P)
			local cond = P:exp()
			P:expect("then")
			local body = P:block()
			node_elseif.cond, node_elseif.body = cond, body
			if_blocks[#if_blocks+1] = node_elseif
		end

		if (P:nextif('else')) then
			else_block = P:block()
		end
		P:expect("end")
		node_if.if_blocks, node_if.else_block = if_blocks, else_block
		return node_if

		--[[ while statement ]]--
	elseif P:nextif("while") then
		local node_while = ast.WhileStatement:New(P)
		local condition = P:exp()
		P:expect("do")
		local body = P:block()
		P:expect("end")
		node_while.cond, node_while.body = condition, body
		return node_while

		-- do block end
	elseif P:nextif("do") then
		local node_do = ast.DoStatement:New(P)
		local body = P:block()
		P:expect("end")
		node_do.body = body
		return node_do

		-- repeat block until condition
	elseif P:nextif("repeat") then
		local node_repeat = ast.RepeatStatement:New(P)
		local body = P:block()
		P:expect("until")
		local condition = P:exp()
		node_repeat.cond, node_repeat.body = condition, body
		return node_repeat

		-- TODO: implement for statement
		-- Just a skeleton. NumericFor loops should be of just one type.
		-- GenericFor loops may be of different types.
		-- What for loops to support within the DSL?
	elseif P:nextif("for") then
		local iterator = P:lvalname()
		if (P:nextif("in")) then
			local node_gf = ast.GenericFor:New(P)
			local set = P:lvalue()
			P:expect("do")
			local body = P:block()
			P:expect("end")
			-- ?? what kinds should these be
			node_gf.iter, node_gf.set, node_gf.body = iterator, set, body
			return node_gf
		else
			P:expect("=")
			local node_nf = ast.NumericFor:New(P)
			node_nf.iter = iterator
			node_nf.lower = P:exp()
			P:expect(',')
			node_nf.upper = P:exp()
			if P:nextif(',') then
				node_nf.step = P:exp()
				P:expect("do")
				node_nf.body = P:block()
				P:expect("end")
			else
				P:expect("do")
				node_nf.body = P:block()
				P:expect("end")
			end
			return node_nf
		end

		--[[ expression statement / assignment statement ]]--
	else
		local expr = P:exp()
		if (P:nextif('=')) then
			local node_asgn = ast.Assignment:New(P)
			-- fix line # info for assignment statement
			node_asgn:copy_location(expr)
			
			node_asgn.lvalue = expr
			node_asgn.exp    = P:exp()
			return node_asgn
		else
			local e = ast.ExprStatement:New(P)
			e.exp   = expr
			return e
		end
	end
end

lang.block = function (P)
	local node_block = ast.Block:New(P)
	local statements = { }
	local first = P:cur().type
	while not block_terminators[first] do
		statements[#statements+1] = lang.statement(P)
		first = P:cur().type
	end

	if P:nextif('break') then
		--TODO: fix line number
		statements[#statements+1] = ast.Break:New(P)
		-- check to make sure break is the last statement of the block
		local key = P:cur().type
		if (not block_terminators[key]) or key == 'break' then
			P:error("block should terminate after the break statement")
		end
	end

	--TODO: fix line number
	node_block.statements = statements
	return node_block
end


exports.lang = lang
return exports

