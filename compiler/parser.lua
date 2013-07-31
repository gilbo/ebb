--[[ Module defines all of the parsing functions used to generate
an AST for Liszt code 
]]--

module(... or 'liszt', package.seeall)

-- Imports
package.path = package.path .. ";./compiler/?.lua;./compiler/?.t"

-- Import ast nodes, keeping out of global scope
ast = require "ast"
_G['ast'] = nil

local Parser = terralib.require('terra/tests/lib/parsing')

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
local function leftbinary(P, lhs)
	local node = ast.BinaryOp:New(P)
	local op  = P:next().type
	local rhs = P:exp(op)
	node.children = {lhs, op, rhs}
	return node
end

local function rightbinary(P, lhs)
	local node = ast.BinaryOp:New(P)
	local op  = P:next().type
	local rhs = P:exp(op, "right")
	node.children = {lhs, op, rhs}
	return node
end

local function unary (P)
	local node = ast.UnaryOp:New(P)
	local op = P:next().type
	local exp = P:exp(precedence.unary)
	node.children = {op, exp}
	return node
end	

----------------------------------
--[[ Build Liszt Pratt parser ]]--
----------------------------------
lang = { }

--[[ Expression parsing ]]--
lang.exp = Parser.Pratt() -- returns a pratt parser
:prefix("-",   unary)
:prefix("not", unary)

:infix("or",  precedence["or"],  leftbinary)
:infix("and", precedence["and"], leftbinary)

:infix("<",   precedence["<"],   leftbinary)
:infix(">",   precedence[">"],   leftbinary)
:infix("<=",  precedence["<="],  leftbinary)
:infix(">=",  precedence[">="],  leftbinary)
:infix("==",  precedence["=="],  leftbinary)
:infix("~=",  precedence["~="],  leftbinary)

:infix("*",   precedence['*'],   leftbinary)
:infix('/',   precedence['/'],   leftbinary)
:infix('%',   precedence['%'],   leftbinary)

:infix("+",   precedence["+"],   leftbinary)
:infix("-",   precedence["-"],   leftbinary)
:infix('^',   precedence['^'],   rightbinary)
:prefix(Parser.default, function(P) return P:simpleexp() end)

-- tuples are used to represent a sequence of comma-seperated expressions that can
-- be found in function calls and will perhaps be used in for statements
lang.tuple = function (P, exprs)
	exprs = exprs or { }
	exprs[#exprs + 1] = P:exp()
	if P:nextif(",") then
		return P:tuple(exprs)
	else 
		--TODO: check line number
		return ast.Tuple:New(P, unpack(exprs))
	end
end

--[[ recursively checks to see if lhs is part of a field index 
or a table lookup. lhs parameter is already an LValue 
--]]
lang.lvaluehelper = function (P, lhs)
	local cur = P:cur()
	-- table indexing:
	if cur.type == '.' or cur.type == ':' then
		local node = ast.TableLookup:New(P)
		local op = P:next().type
		-- check to make sure the table is being indexed by a valid name
		if not P:matches(P.name) then P:error("expected name after '.'") end
		local nodename = ast.Name:New(P)
		nodename.children = {P:next().value}
		node.children = {lhs, op, nodename}
		return P:lvaluehelper(node)

		-- field index / function call?
	elseif P:nextif('(') then
		local node = ast.Call:New(P)
		local args = P:tuple()
		P:expect(')')
		node.children = {lhs, args}
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
	local symname = P:next().value
	node.children = {symname}
	P:ref(symname)
	return P:lvaluehelper(node)
end

lang.lvalname = function (P)
	if not P:matches(P.name) then
		local token = P:next()
		P:error("Expected name at " .. token.linenumber .. ":" .. token.offset)
	end
	local node = ast.Name:New(P)
	local symname = P:next().value
	node.children = {symname}
	P:ref(symname)
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
		node.children = {P:next().value}
		return node

		-- catch bools
	elseif P:matches('true') or P:matches('false') then
		local node = ast.Bool:New(P)
		node.children = {P:next().type}
		return node

		-- catch parenthesized expressions
	elseif P:nextif("(") then
		local v = P:exp()
		P:expect(")")
		return v
	end

	-- expect LValue
	return P:lvalue()
end

lang.liszt_kernel = function (P)
	-- parse liszt_kernel keyword and argument:
	P:expect("liszt_kernel")
	local kernel_node = ast.LisztKernel:New(P)

	-- parse parameter
	P:expect("(")
	local param = P:expect(P.name).value
	P:expect(")")

	-- parse block
	local block = P:block()
	P:expect("end")

	kernel_node.children = {param, block}
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
		local name = P:lvalue()
		-- differentiate between initialization and declaration
		if (P:nextif("=")) then
			local node_init = ast.InitStatement:New(P)
			local expr = P:exp()
			node_init.children = {name, expr}
			return node_init
		else
			local node_decl = ast.DeclStatement:New(P)
			node_decl.children = {name}
			return node_decl
		end

		--[[ if statement ]]--
	elseif P:nextif("if") then
		local node_if = ast.IfStatement:New(P)
		local node_ifc = ast.CondBlock:New(P)
		local blocks = { }
		local cond = P:exp()
		P:expect("then")
		local body = P:block()
		node_ifc.children = {cond, body}
		blocks[#blocks+1] = node_ifc
		-- parse all elseif clauses
		while (P:nextif("elseif")) do
			local node_elseif = ast.CondBlock:New(P)
			local cond = P:exp()
			P:expect("then")
			local body = P:block()
			node_elseif.children = {cond, body}
			blocks[#blocks+1] = node_elseif
		end
		if (P:nextif('else')) then
			blocks[#blocks+1]=P:block()
		end
		P:expect("end")
		node_if.children = {unpack(blocks)}
		return node_if

		--[[ while statement ]]--
	elseif P:nextif("while") then
		local node_while = ast.WhileStatement:New(P)
		local condition = P:exp()
		P:expect("do")
		local body = P:block()
		P:expect("end")
		node_while.children = {condition, body}
		return node_while

		-- do block end
	elseif P:nextif("do") then
		local node_do = ast.DoStatement:New(P)
		local body = P:block()
		P:expect("end")
		node_do.children = {body}
		return node_do

		-- repeat block until condition
	elseif P:nextif("repeat") then
		local node_repeat = ast.RepeatStatement:New(P)
		local body = P:block()
		P:expect("until")
		local condition = P:exp()
		node_repeat.children = {condition, body}
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
			node_gf.children = {iterator, set, body}
			return node_gf
		else
			P:expect("=")
			local node_nf = ast.NumericFor:New(P)
			local exprs = { }
			exprs[1] = P:exp()
			P:expect(',')
			exprs[2] = P:exp()
			if P:nextif(',') then
				exprs[3] = P:exp()
			end
			P:expect("do")
			local body = P:block()
			P:expect("end")
			node_nf.children = {iterator, unpack(exprs), body}
			return node_nf
		end

		--[[ expression statement / assignment statement ]]--
	else
		local expr = P:exp()
		if (P:nextif('=')) then
			local node_asgn = ast.Assignment:New(P)
			-- check to make sure lhs is an LValue
			if not expr.isLValue() then P:error("expected LValue before '='") end
			local rhs = P:exp()
			node_asgn.children = {expr, rhs}
			return node_asgn
		else
			return expr
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
	node_block.children = {unpack(statements)}
	return node_block
end
