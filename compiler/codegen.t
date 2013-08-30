module(... or 'codegen', package.seeall)

ast    = require 'ast'
semant = require 'semant'
terralib.require 'runtime/liszt'
local runtime = runtime

function ast.AST:codegen (env)
	error("Codegen not implemented for AST node " .. self.kind)
end

function ast.LisztKernel:codegen (env)
	env:localenv()[self.children[1].children[1]] = symbol() -- symbol for kernel parameter
	env.context = symbol() -- lkContext* argument for kernel function
	return self.children[2]:codegen(env)

end

function ast.Block:codegen (env)
	-- start with an empty ast node, or we'll get an error when appending new quotes below
	local code = quote end
	for i = 1, #self.children do
		local stmt = self.children[i]:codegen(env)
		code = quote code stmt end
	end
	return code
end

-- for now, just assume all assignments are to locally-scoped variables
-- assignments to variables from lua scope will require liszt runtime calls and extra information from semantic type checking
function ast.Assignment:codegen (env)
	local lhs = self.children[1]:codegen_lhs(env)
	local rhs = self.children[2]:codegen(env)

	return quote lhs = rhs end
end

function ast.InitStatement:codegen (env)
	local varname = self.children[1].children[1]
	local varsym  = symbol()
	env:localenv()[varname] = varsym

	return quote var [varsym] = [self.children[2]:codegen(env)] end
end

function ast.Name:codegen_lhs (env)
	local name = self.children[1]

	-- if declared in local scope, then we should have a 
	-- symbol in the environment table
	if self.node_type.scope == semant._LISZT_STR then
		if self.node_type.objtype == semant._VECTOR then
			return `env.localenv()[name].__data
		else
			return `env:localenv()[name]
		end

	else
		if self.node_type.objtype == semant._VECTOR then
			return `[self.children[1] .. '.__data']
		else
			return `[self.children[1]]
		end
	end
end

-- Name:codegen only has to worry about returning r-values,
-- so it can just look stuff up in the environment and return
-- it, since all variables in the liszt scope will be
-- in the environment as a symbol
function ast.Name:codegen (env)
	local str = self.children[1]
	val = env:combinedenv()[str]
	if (type(val) == 'number' or type(val) == 'boolean') then
		return `[val]
	elseif (val.isglobal) then
		local terra get_global () return val end
		-- store this global in the environment table so we won't have to look it up again
		env:luaenv()[str] = get_global()
		val = env:luaenv()[str]
		return `val
	elseif Vector.isVector(val) then
		return `val.__data
	-- symbols
	else
		return `[val]
	end
end

function ast.Number:codegen (env)
	return `[self.children[1]]
end

function ast.Bool:codegen (env)
	if self.children[1] == 'true' then return `true
	else return `false
	end
end

function ast.UnaryOp:codegen (env)
	local expr = self.children[2]:codegen(env)
	local op   = self.children[1]

	if (op == '-') then return `-[expr]
	else return `not [expr]
	end
end

function ast.BinaryOp:codegen (env)
	local lhe = self.children[1]:codegen(env)
	local rhe = self.children[3]:codegen(env)

	op = self.children[2]

	if     op == '+'   then return `lhe +   rhe
	elseif op == '-'   then return `lhe -   rhe
	elseif op == '/'   then return `lhe /   rhe
	elseif op == '*'   then return `lhe *   rhe
	elseif op == '%'   then return `lhe %   rhe
	elseif op == '^'   then return `lhe ^   rhe
	elseif op == 'or'  then return `lhe or  rhe
	elseif op == 'and' then return `lhe and rhe
	elseif op == '<'   then return `lhe <   rhe
	elseif op == '>'   then return `lhe >   rhe
	elseif op == '<='  then return `lhe <=  rhe
	elseif op == '>='  then return `lhe >=  rhe
	elseif op == '=='  then return `lhe ==  rhe
	elseif op == '~='  then return `lhe ~=  rhe
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
			[self.chidren]
		end
	end


	env:leaveblock()
	return code
end
]]--

function codegen (luaenv, kernel_ast)
	local env = terralib.newenvironment(luaenv)
	env:enterblock()
	local kernel_code = kernel_ast:codegen(env)
--[[
	local kernel_fn = terra ([env.context] : runtime.lkContext) : {}
		[kernel_code]
	end
]]

	return terra ()
		kernel_code
	end
	-- return kernel_fn
end