local exports = {}

local ast    = require 'ast'
local semant = require 'semant'
terralib.require 'runtime/liszt'
local types = terralib.require 'compiler/types'
local Type = types.Type
local t    = types.t
local runtime = package.loaded.runtime

function ast.AST:codegen (env)
	error("Codegen not implemented for AST node " .. self.kind)
end

function ast.ExprStatement:codegen (env)
	return self.exp:codegen(env)
end

function ast.LisztKernel:codegen (env)
	env.param   = symbol()
	env:localenv()[self.param.name] = env.param -- symbol for kernel parameter
	env.context = symbol() -- lkContext* argument for kernel function
	return self.body:codegen(env)
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

function ast.DeclStatement:codegen (env)
	-- if this var is never used, don't bother declaring it
	-- since we don't know the type, we couldn't anyway.
	if self.ref.node_type == t.unknown then return quote end end

	local typ  = self.ref.node_type:terraType()
	local name = self.ref.name
	local sym  = symbol(typ)
	env:localenv()[name] = sym
	return quote var [sym] end
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

local c = terralib.includecstring([[
#include <stdio.h>
#include <stdlib.h>

FILE *get_stderr () { return stderr; }
]])

local terra lisztAssert(test : bool, file : rawstring, line : int)
    if not test then
        c.fprintf(c.get_stderr(), "%s:%d: assertion failed!\n", file, line)
        c.exit(1)
    end
end

local phaseMap = {
	['+']   	= runtime.L_PLUS,
	['-']   	= runtime.L_MINUS,
	['*']   	= runtime.L_MULTIPLY,
	['/']   	= runtime.L_DIVIDE,
	['and'] 	= runtime.L_BAND,
	['or']  	= runtime.L_BOR,
	['land']    = runtime.L_AND,
	['lor']     = runtime.L_OR,
}

function ast.Assignment:codegen (env)
	--local lval = self.lvalue.luaval
	local lvalue = self.lvalue
	local ttype  = self.lvalue.node_type:terraType()

	if lvalue:is(ast.FieldAccess) then
		local exp     = self.exp:codegen(env)
		local field   = lvalue.field
		local topo    = lvalue.topo:codegen(env)
		local phase   = self.reduceop and phaseMap[self.reduceop]
		             or runtime.L_ASSIGN
		local element_type, element_length = lvalue.node_type:runtimeType()

		return quote
			var tmp : ttype = [exp]
			runtime.lkFieldWrite([field.__lkfield], topo, phase,
													 element_type, element_length,
													 0, element_length, &tmp)
		end

	elseif lvalue:is(ast.Scalar) then
		local exp     = self.exp:codegen(env)
		local scalar  = lvalue.scalar
		local phase   = phaseMap[self.reduceop]
		local el_type, el_len = lvalue.node_type:runtimeType()
		return quote
			var tmp : ttype = [exp]
			runtime.lkScalarWrite([env.context], [scalar.__lkscalar],
														phase, el_type, el_len, 0, el_len, &tmp)
		end

	else
		local lhs = lvalue:codegen(env)
		local rhs = self.exp:codegen(env)
		return quote lhs = rhs end
	end
end

function ast.FieldAccess:codegen (env)
	local field = self.field
	local topo  = self.topo:codegen(env)
	local read  = symbol()

	local typ   = self.node_type:terraType()
	local el_type, el_len = self.node_type:runtimeType()

	return quote
		var [read] : typ
		runtime.lkFieldRead([field.__lkfield], [topo], el_type, el_len, 0, el_len, &[read])
			in
		[read]
	end
end

-- By the time we make it to codegen, Call nodes are only used to represent builtin function calls.
function ast.Call:codegen (env)
	return self.func.func.codegen(self, env)
end

function ast.InitStatement:codegen (env)
	local varname = self.ref.name
	local tp      = self.ref.node_type:terraType()
	local varsym  = symbol(tp)

	env:localenv()[varname] = varsym
	local exp = self.exp:codegen(env)
	return quote var [varsym] = [exp] end
end

--function ast.Name:codegen_lhs (env)
--	return `[env:combinedenv()[self.name]]
--end

function ast.VectorLiteral:codegen (env)
	local ct = { }
	local v = symbol()
	local tp = self.node_type:terraBaseType()
	for i = 1, #self.elems do
		ct[i] = self.elems[i]:codegen()
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

local function codegen_scalar(node, env)
	local read = symbol()
	local el_type, el_len = node.node_type:runtimeType()
	local lks = node.luaval.__lkscalar
	local typ = node.node_type:terraType()

	return quote
		var [read] : typ
		runtime.lkScalarRead([env.context], [lks], el_type, el_len, 0, el_len, &[read])
		in
		[read]
	end
end

function ast.Scalar:codegen (env)
	local read = symbol()
	local el_type, el_len = self.node_type:runtimeType()
	local lks  = self.scalar.__lkscalar
	local typ  = self.node_type:terraType()

	return quote
		var [read] : typ
		runtime.lkScalarRead([env.context], [lks], el_type, el_len, 0, el_len, &[read])
		in
		[read]
	end
	--return codegen_scalar(self, env)
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

	-- TODO: special case equality, inequality operators for vectors!

	if     self.op == '+'   then return `lhe +   rhe
	elseif self.op == '-'   then return `lhe -   rhe
	elseif self.op == '/'   then return `lhe /   rhe
	elseif self.op == '*'   then return `lhe *   rhe
	elseif self.op == '%'   then return `lhe %   rhe
	elseif self.op == '^'   then return `lhe ^   rhe
	elseif self.op == 'or'  then return `lhe or  rhe
	elseif self.op == 'and' then return `lhe and rhe
	elseif self.op == '<'   then return `lhe <   rhe
	elseif self.op == '>'   then return `lhe >   rhe
	elseif self.op == '<='  then return `lhe <=  rhe
	elseif self.op == '>='  then return `lhe >=  rhe
	elseif self.op == '=='  then return `lhe ==  rhe
	elseif self.op == '~='  then return `lhe ~=  rhe
	end

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

	local program = terra (ctx : runtime.lkContext)
		var [env.context] = &ctx
		var [env.param] : runtime.lkElement
		if runtime.lkGetActiveElement([env.context], &[env.param]) > 0 then
			kernel_body
		end
	end
	env:leaveblock()

	return program
end


return exports

