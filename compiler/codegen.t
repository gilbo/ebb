local exports = {}

local ast     = require 'ast'
local semant  = require 'semant'
local types   = terralib.require 'compiler/types'
local Type    = types.Type
local t       = types.t

function ast.AST:codegen (env)
	print(debug.traceback())
	error("Codegen not implemented for AST node " .. self.kind)
end

function ast.ExprStatement:codegen (env)
	return self.exp:codegen(env)
end

function ast.LisztKernel:codegen (env)
	local param = symbol(t.addr:terraType())
	env:localenv()[self.iter.name] = param

	local set  = self.set:codegen(env)
	local body = self.body:codegen(env)

	return quote
		for [param] = 0, [set]._size do
			[body]
		end
	end
end

function ast.Relation:codegen(env)
	local rel = self.relation
	return `rel
end

function ast.Block:codegen (env)
	-- start with an empty ast node, or we'll get an error when appending new quotes below
	local code = quote end
	for i = 1, #self.statements do
		local stmt = self.statements[i]:codegen(env)
		code = quote code stmt end
	end
	return code
end

function ast.CondBlock:codegen(env, cond_blocks, else_block, index)
	index = index or 1

	local cond  = self.cond:codegen(env)
	env:enterblock()
	local body = self.body:codegen(env)
	env:leaveblock()

	if index == #cond_blocks then
		if else_block then
			return quote if [cond] then [body] else [else_block:codegen(env)] end end
		else
			return quote if [cond] then [body] end end
		end
	else
		env:enterblock()
		local nested = cond_blocks[index + 1]:codegen(env, cond_blocks, else_block, index + 1)
		env:leaveblock()
		return quote if [cond] then [body] else [nested] end end
	end
end

function ast.IfStatement:codegen (env)
	return self.if_blocks[1]:codegen(env, self.if_blocks, self.else_block)
end

function ast.WhileStatement:codegen (env)
	local cond = self.cond:codegen(env)
	env:enterblock()
	local body = self.body:codegen(env)
	env:leaveblock()
	return quote while [cond] do [body] end end
end

function ast.DoStatement:codegen (env)
	env:enterblock()
	local body = self.body:codegen(env)
	env:leaveblock()
	return quote do [body] end end
end

function ast.RepeatStatement:codegen (env)
	env:enterblock()
	local body = self.body:codegen(env)
	local cond = self.cond:codegen(env)
	env:leaveblock()

	return quote repeat [body] until [cond] end
end

function ast.NumericFor:codegen (env)
	-- min and max expression should be evaluated in current scope,
	-- iter expression should be in a nested scope, and for block
	-- should be nested again -- that way the loop var is reset every
	-- time the loop runs.
	local minexp  = self.lower:codegen(env)
	local maxexp  = self.upper:codegen(env)
	local stepexp = self.step and self.step:codegen(env) or nil

	env:enterblock()
	local iterstr = self.iter.name
	local itersym = symbol()
	env:localenv()[iterstr] = itersym

	env:enterblock()
	local body = self.body:codegen(env)
	env:leaveblock()
	env:leaveblock()

	if stepexp then
		return quote for [itersym] = [minexp], [maxexp], [stepexp] do [body] end end
	end

	return quote for [itersym] = [minexp], [maxexp] do [body] end end
end

function ast.Break:codegen(env)
	return quote break end
end

function ast.LocalVar:codegen(env)
	local s = env:localenv()[self.name]
	return `[s]
end

local function bin_exp (op, lhe, rhe)
	if     op == '+'   then return `[lhe] +   [rhe]
	elseif op == '-'   then return `[lhe] -   [rhe]
	elseif op == '/'   then return `[lhe] /   [rhe]
	elseif op == '*'   then return `[lhe] *   [rhe]
	elseif op == '%'   then return `[lhe] %   [rhe]
	elseif op == '^'   then return `[lhe] ^   [rhe]
	elseif op == 'or'  then return `[lhe] or  [rhe]
	elseif op == 'and' then return `[lhe] and [rhe]
	elseif op == '<'   then return `[lhe] <   [rhe]
	elseif op == '>'   then return `[lhe] >   [rhe]
	elseif op == '<='  then return `[lhe] <=  [rhe]
	elseif op == '>='  then return `[lhe] >=  [rhe]
	elseif op == '=='  then return `[lhe] ==  [rhe]
	elseif op == '~='  then return `[lhe] ~=  [rhe]
	end
end

function ast.Assignment:codegen (env)
	-- if lhs is local, there will be a symbol stored in env
	-- If it is global, we will not have a symbol stored,
	-- and we will need to codegen to get the reference
	local lhs   = env:localenv()[self.lvalue.name] or self.lvalue:codegen(env)
	local ttype = self.lvalue.node_type:terraType()
	local rhs   = self.exp:codegen(env)

	if self.reduceop then
		rhs = bin_exp(self.reduceop, lhs, rhs)
	end
	return quote [lhs] = rhs end
end

function ast.Row:codegen (env)
	local e = env:localenv()[self.name]
	return `[e]
end

function ast.FieldAccess:codegen (env)
	local field = self.field
	local index = self.row:codegen(env)
	return `@(field.data + [index])

	--[[
	local typ   = self.node_type:terraType()
	local el_type, el_len = self.node_type:runtimeType()

	return quote
		var [read] : typ
		runtime.lkFieldRead([field.__lkfield], [topo], el_type, el_len, 0, el_len, &[read])
			in
		[read]
	end
	]]
end

-- By the time we make it to codegen, Call nodes are only used to represent builtin function calls.
function ast.Call:codegen (env)
	return self.func.func.codegen(self, env)
end

function ast.DeclStatement:codegen (env)
	local varname = self.ref.name
	local tp      = self.ref.node_type:terraType()
	local varsym  = symbol(tp)
	env:localenv()[varname] = varsym

	if self.initializer then
		local exp = self.initializer:codegen(env)
		return quote var [varsym] = [exp] end
	else
		return quote var [varsym] end
	end
end

function ast.VectorLiteral:codegen (env)
	local ct = { }
	local v = symbol()
	local tp = self.node_type:terraBaseType()
	for i = 1, #self.elems do
		ct[i] = self.elems[i]:codegen(env)
	end

   -- These quotes give terra the opportunity to generate optimized assembly via the vectorof call
   -- when I dissassembled terra functions at the console, they seemed more likely to use vector
   -- ops for loading values if vectors are initialized this way.
   local v1, v2, v3, v4, v5, v6 = ct[1], ct[2], ct[3], ct[4], ct[5], ct[6]
	if #ct == 2 then
		return `vectorof(tp, v1, v2)
	elseif #ct == 3 then
		return `vectorof(tp, v1, v2, v3)
	elseif #ct == 4 then
		return `vectorof(tp, v1, v2, v3, v4)
	elseif #ct == 5 then
		return `vectorof(tp, v1, v2, v3, v4, v5)
	elseif #ct == 6 then
		return `vectorof(tp, v1, v2, v3, v4, v5, v6)

	else
		local s = symbol(self.node_type:terraType())
		local t = symbol()
		local q = quote
			var [s]
			var [t] = [&tp](&s)
		end

		for i = 1, #ct do
			local val = ct[i]
			q = quote 
				[q] 
				@[t] = [val]
				t = t + 1
			end
		end
		return quote [q] in [s] end
	end
end

function ast.Scalar:codegen (env)
	local d = self.scalar.data
	local s = symbol(&self.scalar.type:terraType())
	return quote var [s] = d in @[s] end
end

-- Name:codegen only has to worry about returning r-values
-- Name:codegen_lhs returns l-values
function ast.Name:codegen (env)
	local val = env:combinedenv()[self.name]

	return `[val]
end

function ast.VectorIndex:codegen (env)
	local vector = self.vector:codegen(env)
	local index  = self.index:codegen(env)

	return `vector[index]
end

function ast.Number:codegen (env)
	return `[self.value]
end

function ast.Bool:codegen (env)
	if self.value == 'true' then
		return `true
	else 
		return `false
	end
end

function ast.UnaryOp:codegen (env)
	local expr = self.exp:codegen(env)
	if (self.op == '-') then return `-[expr]
	else return `not [expr]
	end
end


function ast.BinaryOp:codegen (env)
	local lhe = self.lhs:codegen(env)
	local rhe = self.rhs:codegen(env)
	return bin_exp(self.op, lhe, rhe)
end

--[[
function ast.GenericFor:codegen (env)
	env:enterblock()

	local varname = self.children[1].children[1]
	local varsym = symbol()
	local ctx    = env.context
	env:localenv()[varname] = varsym

	local code = quote
		var [varsym] : runtime.lkElement
		if (runtime.lkGetActiveElement(&[ctx], [varsym]) > 0) then
			-- run block!
		end
	end


	env:leaveblock()
	return code
end
]]--

function exports.codegen (luaenv, kernel_ast)
	local env = terralib.newenvironment(luaenv)

	env:enterblock()
	local kernel_body = kernel_ast:codegen(env)
	env:leaveblock()

	return terra ()
		[kernel_body]
	end
end

return exports

