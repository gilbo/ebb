-- Imports
package.path = package.path .. ";./compiler/?.lua;./compiler/?.t"
require "ast"
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
	local op  = P:next().type
	local rhs = P:exp(op)
	return BinaryOp:New(lhs, op, rhs)
end

local function rightbinary(P, lhs)
	local op  = P:next().type
	local rhs = P:exp(op, "right")
	return BinaryOp:New(lhs, op, rhs)
end

local function unary (P)
	local op = P:next().type
	local exp = P:exp(precedence.unary)
	return UnaryOp:New(op, exp)
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
		return Tuple:New(unpack(exprs))
	end
end

--[[ recursively checks to see if lhs is part of a field index 
     or a table lookup. lhs parameter is already an LValue 
--]]
lang.lvaluehelper = function (P, lhs)
	local cur = P:cur()
	-- table indexing:
	if cur.type == '.' or cur.type == ':' then
		local op = P:next().type
		-- check to make sure the table is being indexed by a valid name
		if not P:matches(P.name) then P:error("expected name after '.'") end
		return P:lvaluehelper(TableLookup:New(lhs, op, Name:New(P:next().value)))

	-- field index / function call?
	elseif P:nextif('(') then
		local args = P:tuple()
		P:expect(')')
		return P:lvaluehelper(Call:New(lhs, args))
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

	return P:lvaluehelper(Name:New(P:next().value))
end

--[[  simpleexp is called when exp cannot parse the next token...
      we could be looking at an LValue, Value, or parenthetically
      enclosed expression  
--]]
lang.simpleexp = function(P)
	-- catch values
	if P:matches(P.number) then
		return Number:New(P:next().value)		

	-- catch bools
	elseif P:matches('true') or P:matches('false') then
		return Bool:New(P:next().type)

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

	-- parse parameter
	P:expect("(")
	local param = P:expect(P.name).value
	P:expect(")")

	-- parse block
	local block = P:block()
	P:expect("end")

	return LisztKernel:New(param, block)
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
			local expr = P:exp()
			return InitStatement:New(name, expr)
		else
			return DeclStatement:New(name)
		end

	--[[ if statement ]]--
	elseif P:nextif("if") then
		local blocks = { }

		local cond = P:exp()

		P:expect("then")
		local body = P:block()
		blocks[#blocks+1] = CondBlock:New(cond, body)

		-- parse all elseif clauses
		while (P:nextif("elseif")) do
			local cond = P:exp()
			P:expect("then")
			local body = P:block()
			blocks[#blocks+1] = CondBlock:New(cond, body)							
			end

		if (P:nextif('else')) then
			blocks[#blocks+1]=P:block()
		end

		P:expect("end")
		return IfStatement:New(unpack(blocks))

	--[[ while statement ]]--
	elseif P:nextif("while") then
		local condition = P:exp()
		P:expect("do")
		local body = P:block()
		P:expect("end")
		return WhileStatement:New(condition, body)

	-- do block end
	elseif P:nextif("do") then
		local body = P:block()
		P:expect("end")
		return DoStatement:New(body)

	-- repeat block until condition
	elseif P:nextif("repeat") then
		local body = P:block()
		P:expect("until")
		local condition = P:exp()
		return RepeatStatement:New(condition, body)

	-- TODO: implement for statement
	-- Just a skeleton. NumericFor loops should be of just one type.
	-- GenericFor loops may be of different types.
	-- What for loops to support within the DSL?
    elseif P:nextif("for") then
		local iterator = P:expect(P.name).value
		if (P:nextif("in")) then
                        local set = P:lvalue()
                        P:expect("do")
                        local body = P:block()
                        P:expect("end")
                        -- ?? what kinds should these be
			return GenericFor:New(iterator, set, body)
		else
			P:expect("=")
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
			return NumericFor:New(iterator, unpack(exprs), body)
		end

    elseif P:nextif("foreach") then
		local iterator = P:expect(P.name).value
                P:expect("in")
                local set = P:lvalue()
                P:expect("do")
                local body = P:block()
                P:expect("end")
		return GenericFor:New(iterator, set, body)

	--[[ expression statement / assignment statement ]]--
	else
		local expr = P:exp()
		if (P:nextif('=')) then
			-- check to make sure lhs is an LValue
			if not expr.isLValue() then P:error("expected LValue before '='") end

			local rhs = P:exp()
			return Assignment:New(expr, rhs)
		else
			return expr
		end
	end
end

lang.block = function (P)
	local statements = { }
	local first = P:cur().type
	while not block_terminators[first] do
		statements[#statements+1] = lang.statement(P)
		first = P:cur().type
	end

	if P:nextif('break') then
		statements[#statements+1] = Break:New()
		-- check to make sure break is the last statement of the block
		local key = P:cur().type
		if not block_terminators[key] then
			P:error("block should terminate after the break statement")
		end

		-- make sure we didn't just hit another break statement
		if key == 'break' then P:error("two consecutive break statements") end

	end

	return Block:New(unpack(statements))
end
