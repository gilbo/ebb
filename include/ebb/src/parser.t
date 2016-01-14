-- The MIT License (MIT)
-- 
-- Copyright (c) 2015 Stanford University.
-- All rights reserved.
-- 
-- Permission is hereby granted, free of charge, to any person obtaining a
-- copy of this software and associated documentation files (the "Software"),
-- to deal in the Software without restriction, including without limitation
-- the rights to use, copy, modify, merge, publish, distribute, sublicense,
-- and/or sell copies of the Software, and to permit persons to whom the
-- Software is furnished to do so, subject to the following conditions:
-- 
-- The above copyright notice and this permission notice shall be included
-- in all copies or substantial portions of the Software.
-- 
-- THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
-- IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
-- FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
-- AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
-- LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
-- FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
-- DEALINGS IN THE SOFTWARE.

--
-- Module defines all of the parsing functions used to generate
--  an AST for Ebb code 
--
local P = {}
package.loaded["ebb.src.parser"] = P

-- Import ast nodes, keeping out of global scope
local ast = require "ebb.src.ast"
local pratt = require "ebb.src.pratt"

local block_terminators = {
  ['end']    = 1,
  ['else']   = 1,
  ['elseif'] = 1,
  ['until']  = 1,
  ['break']  = 1,
  ['in']     = 1, -- for let-quotes i.e. quote ... in ... end
  ['return'] = 1, -- for functions only allowed as last statement
}

local unary_prec = 5

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
  local exp = P:exp(unary_prec)
  node.op, node.exp = op, exp
  return node
end 

----------------------------------
--[[ Build Ebb Pratt parser ]]--
----------------------------------
local lang = { }

--[[ Expression parsing ]]--
lang.exp = pratt.Pratt() -- returns a pratt parser
:prefix("-",   unary)
:prefix("not", unary)
:infix("or",  0, leftbinaryred)
:infix("and", 1, leftbinaryred)

:infix("<",   2, leftbinary)
:infix(">",   2, leftbinary)
:infix("<=",  2, leftbinary)
:infix(">=",  2, leftbinary)
:infix("==",  2, leftbinary)
:infix("~=",  2, leftbinary)

:infix("+",   3,   leftbinaryred)
:infix("-",   3,   leftbinaryred)
:infix("*",   4,   leftbinaryred)
:infix('/',   4,   leftbinaryred)
:infix('%',   4,   leftbinary)

:infix('max', 5,   leftbinaryred)
:infix('min', 5,   leftbinaryred)

:infix('^',   6,   rightbinary)
:infix('[',   7, function(P,lhs)
  local begin = P:next().linenumber
  local node = ast.SquareIndex:New(P)
  local exp = P:exp()
  local exp2 = nil
  if not P:matches(']') then -- kinda a kludge, but easier for now
    P:expect(',')
    exp2 = P:exp()
  else
    exp2 = nil
  end
  P:expectmatch(']', '[', begin)
  node.base, node.index = lhs, exp
  if exp2 then node.index2 = exp2 end
  return node
end)
:infix('.',   8, function(P,lhs)
  local node = ast.TableLookup:New(P)
  node.table = lhs
  local op = P:next().type
  local start = P:cur().linenumber
  if P:nextif('[') then --allow an escape to determine a field expression
    node.member = P:luaexpr()
    P:expectmatch(']', '[', start)
  else
    node.member = P:expect(P.name).value
  end
  return node
end)
:infix('(',   9, function(P,lhs)
  local begin = P:next().linenumber
  local node = ast.Call:New(P)
  local params = {}
  if not P:matches(')') then
    repeat
      table.insert(params,P:exp())
    until not P:nextif(',')
  end
  P:expectmatch(')', '(', begin)
  node.func, node.params = lhs, params
  return node
end)
:prefix(pratt.default, function(P) return P:simpleexp() end)

--[[  simpleexp is called when exp cannot parse the next token...
we could be looking at an LValue, Value, or parenthetically
enclosed expression  
--]]
lang.simpleexp = function(P)
  local start = P:cur().linenumber
  if P:matches(P.name) then
    local node = ast.Name:New(P)
    node.name = P:next().value
    P:ref(node.name)
    return node
  elseif P:matches(P.string) then
    local node     = ast.String:New(P)
    node.value     = P:next().value
    return node
  elseif P:matches(P.number) then
    local node     = ast.Number:New(P)
    local token    = P:next()
    node.value     = token.value
    node.valuetype = token.valuetype
    return node
  elseif P:matches('true') or P:matches('false') then
    local node = ast.Bool:New(P)
    node.value = (P:next().type == 'true')
    return node
  elseif P:nextif('(') then
    local v = P:exp()
    P:expectmatch(")", "(", start)
    return v
  elseif P:nextif("{") then
    local node = ast.VectorLiteral:New(P)
    local elems = { }
    repeat
      elems[#elems + 1] = P:exp()
    until not P:nextif(",")
    P:expectmatch("}", "{", start)
    node.elems = elems
    return node
  end
  P:error("unexpected symbol")
end


lang.func_name = function (P)
	local name = P:expect(P.name).value 
	local assign_tuple = { name }
	while P:nextif('.') do
		local select_str = P:expect(P.name).value
		assign_tuple[#assign_tuple+1] = select_str
		name = name .. '.' .. select_str
	end
	return { assign_tuple }, name
end

lang.generate_name_from_entry = function (P)
	local lz = P:nextif('ebb')
	if lz then
		local anon = "anon_" .. tostring(P.source)  .. '_' .. tostring(lz.linenumber)
		-- remove characters that will be escaped out by llvm to make the
		-- generated name more readable
		anon = string.gsub(anon, '[^%a%d_]','_')
		return anon
	else
		P:errorexpected("'ebb'")
	end
end

lang.user_function = function (P, id)
  local function_node = ast.UserFunction:New(P)

  -- parse params
  local arg_pos = 1
  local params  = {}
  local ptypes = {}
  local open    = P:expect("(")
  if not P:matches(')') then
    repeat
      params[arg_pos] = P:expect(P.name).value
      if P:nextif(':') then
        ptypes[arg_pos] = P:luaexpr()
      end
      arg_pos = arg_pos + 1
    until not P:nextif(',')
  end
  P:expectmatch(')', '(', open.linenumber)

  -- parse block
  local block = P:block()
  -- optional terminating return statement
  local exp   = nil
  if P:nextif("return") then
    exp = P:exp()
  end
  P:expect("end")

  function_node.id      = id
  function_node.params  = params
  function_node.ptypes  = ptypes
  function_node.body    = block
  function_node.exp     = exp
  return function_node
end

lang.ebbExpression = function (P)
    local code_type
    local anon_name = P:generate_name_from_entry()
    if P:nextif('`') then
        code_type = 'simp_quote'
    elseif P:nextif('quote') then
        code_type = 'let_quote'
    else
        code_type = 'function'
    end

    if code_type == 'function' then
      return P:user_function(anon_name)

    elseif code_type == 'let_quote' then
      local let_exp = ast.LetExpr:New(P)
      local block   = P:block()
                      P:expect('in')
      local exp     = P:exp()
                      P:expect('end')
      let_exp.block, let_exp.exp = block, exp

      local q = ast.Quote:New(P)
      q.code = let_exp
      return q

    else -- code_type == 'simp_quote'
      local q       = ast.Quote:New(P)
      local exp     = P:exp()
      
      q.code = exp
      return q

    end
end

lang.ebbStatement = function (P)
	local code_type
	if P:nextif("ebb") then
    code_type = 'function'
	else
		P:errorexpected("'ebb'")
	end

	local assign_tuple, name = P:func_name()
  return P:user_function(name), assign_tuple
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
    local node_decl = ast.DeclStatement:New(P)
    node_decl.name = P:expect(P.name).value
    if P:nextif(":") then
      node_decl.typeexpression = P:luaexpr()
      if (P:nextif("=")) then
        node_decl.initializer = P:exp()
      end
    else
      P:expect("=")
      node_decl.initializer = P:exp()
    end
    return node_decl

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
    local iterator = P:expect(P.name).value
    if (P:nextif("in")) then
      local node_gf = ast.GenericFor:New(P)
      local set = P:exp()
      P:expect("do")
      local body = P:block()
      P:expect("end")
      -- ?? what kinds should these be
      node_gf.name, node_gf.set, node_gf.body = iterator, set, body
      return node_gf
    else
      P:expect("=")
      local node_nf = ast.NumericFor:New(P)
      node_nf.name = iterator
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

    --[[ insert statement ]]--
  elseif P:nextif("insert") then
    local insert    = ast.InsertStatement:New(P)
    local record    = P:record()
                      P:expect("into")
    local relation  = P:exp()
    insert.record, insert.relation = record, relation
    return insert

    --[[ delete statement ]]--
  elseif P:nextif("delete") then
    local delete = ast.DeleteStatement:New(P)
    local key    = P:exp()
    delete.key   = key
    return delete

    --[[ expression statement / assignment statement ]]--
  else
    local expr = P:exp()
    if (P:nextif('=')) then
      local node_asgn = ast.Assignment:New(P)
      -- fix line # info for assignment statement
      node_asgn:copy_location(expr)
      
      if expr:is(ast.Reduce) then
        node_asgn.reduceop, node_asgn.lvalue = expr.op, expr.exp
      else
        node_asgn.lvalue = expr
      end
      node_asgn.exp    = P:exp()
      return node_asgn
    else
      local e = ast.ExprStatement:New(P)
      e.exp   = expr
      return e
    end
  end
end

lang.record = function (P)
  local open_curly = P:expect('{')

  local record = ast.RecordLiteral:New(P)
  local names = {}
  local exprs = {}
  repeat
    local field_name  = P:expect(P.name).value
                        P:expect('=')
    local field_value = P:exp()
    table.insert(names, field_name)
    table.insert(exprs, field_value)
  until not P:nextif(',')
  P:expectmatch('}','{', open_curly.linenumber)

  record.names, record.exprs = names, exprs
  return record
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

function P.ParseExpression(lexer)
	return pratt.Parse(lang, lexer, "ebbExpression")
end

function P.ParseStatement(lexer)
	return pratt.Parse(lang, lexer, "ebbStatement")
end
