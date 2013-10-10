module(... or 'codegen', package.seeall)

local ast    = require 'ast'
local semant = require 'semant'
terralib.require 'runtime/liszt'
local runtime = package.loaded.runtime

function ast.AST:codegen (env)
	error("Codegen not implemented for AST node " .. self.kind)
end

function ast.ExprStatement:codegen (env)
	return quote end
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
	if self.node_type.objtype == semant._NOTYPE then return quote end end

	local lisztToTerraTypes = {
		[semant._NUM]   = double,
		[semant._INT]   = int,
		[semant._FLOAT] = float,
		[semant._BOOL]  = bool
	}

	local typ = lisztToTerraTypes[self.node_type.objtype]
	if not typ then
		local elemtype = lisztToTerraTypes[self.node_type.elemtype]
		typ = vector(elemtype, self.node_type.size)
	end

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

-- for now, just assume all assignments are to locally-scoped variables
-- assignments to variables from lua scope will require liszt runtime 
-- calls and extra information from semantic type checking

local simpleTypeMap = {
	[semant._FLOAT] = runtime.L_FLOAT,
	[semant._INT]   = runtime.L_INT,
	[semant._BOOL]  = runtime.L_BOOL,
}

local terraTypeMap = {
	[semant._FLOAT] = float,
	[semant._INT]   = int,
	[semant._BOOL]  = bool,
}

local function objTypeToLiszt (obj)
	if simpleTypeMap[obj.objtype] then
		return simpleTypeMap[obj.objtype], 1, 0, 1
	else -- objtype should be a vector!
		return simpleTypeMap[obj.elemtype], obj.size, 0, obj.size
	end
end

local function objTypeToTerra (obj)
	if obj.size == 1 then
		return terraTypeMap[obj.objtype]
	else
		return vector(terraTypeMap[obj.elemtype], obj.size)
	end
end

local elemTypeMap = {
	[semant._VERTEX] = runtime.L_VERTEX,
	[semant._CELL]   = runtime.L_CELL,
	[semant._FACE]   = runtime.L_FACE,
	[semant._EDGE]   = runtime.L_EDGE
}

local mPhaseMap = {
	['+']   = runtime.L_PLUS,
	['-']   = runtime.L_MINUS,
	['*']   = runtime.L_MULTIPLY,
	['/']   = runtime.L_DIVIDE,
	['and'] = runtime.L_BAND,
	['or']  = runtime.L_BOR
}

local bPhaseMap = {
	['and'] = runtime.L_AND,
	['or']  = runtime.L_OR
}

local function getPhase (binop)
	-- for boolean types, return boolean reduction phases
	if binop.node_type.objtype == semant._BOOL or
		(binop.node_type.objtype == semant._VECTOR and binop.node_type.elemtype == semant._BOOL) then
		return bPhaseMap[binop.op] end

	return mPhaseMap[binop.op]
end


function ast.Assignment:codegen (env)
	local tp = objTypeToTerra(self.node_type)
	if self.fieldop == semant._FIELD_WRITE then
		local exp     = self.exp:codegen(env)
		local field   = self.field
		local topo    = self.topo:codegen(env)
		local element_type, element_length, val_offset, val_length = objTypeToLiszt(self.lvalue.node_type)

		return quote
			var tmp : tp = [exp]
			runtime.lkFieldWrite([field.lkfield], [topo], runtime.L_ASSIGN, element_type, element_length, val_offset, val_length, &tmp)
		end
	elseif self.fieldop == semant._FIELD_REDUCE then
		local exp     = self.rexp:codegen(env)
		local field   = self.field
		local topo    = self.topo:codegen(env)
		local phase   = getPhase(self.exp)
		local element_type, element_length, val_offset, val_length = objTypeToLiszt(self.lvalue.node_type)

		return quote
			var tmp : tp = [exp]
			runtime.lkFieldWrite([field.lkfield], [topo], [phase], element_type, element_length, val_offset, val_length, &tmp)
		end
	elseif self.fieldop == semant._SCALAR_REDUCE then
		local exp   = self.rexp:codegen(env)
		local lsc   = self.scalar
		local phase = getPhase(self.exp)
		local el_type, el_len, val_offset, val_len = objTypeToLiszt(self.lvalue.node_type)
		return quote
			var tmp : tp = [exp]
			runtime.lkScalarWrite([env.context], [lsc.__lkscalar], [phase], [el_type], [el_len], [val_offset], [val_len], &tmp)
		end

	else
		local lhs = self.lvalue:codegen_lhs(env)
		local rhs = self.exp:codegen(env)
		return quote lhs = rhs end
	end
end

-- Call:codegen is called for field reads, when the field appears
-- in an expression, or on the rhs of an assignment.
function ast.Call:codegen (env)
	local field = self.func.node_type.luaval
	local topo  = self.params.children[1]:codegen(env) -- Should return an lkElement
	local read  = symbol()

	local typ = objTypeToTerra(self.node_type)
	local el_type, el_len, val_off, val_len = objTypeToLiszt(self.node_type)

	return quote
		var [read] : typ
		runtime.lkFieldRead([field.lkfield], [topo], el_type, el_len, val_off, val_len, &[read])
		in
		[read]
	end
end

function ast.InitStatement:codegen (env)
	local varname = self.ref.name
	local varsym  = symbol()
	env:localenv()[varname] = varsym
	return quote var [varsym] = [self.exp:codegen(env)] end
end

function ast.Name:codegen_lhs (env)
	return `[env:combinedenv()[self.name]]
end

function ast.VectorLiteral:codegen (env)
	local ct = { }
	local v = symbol()
	local tp = terraTypeMap[self.node_type.elemtype]
	for i = 1, #self.elems do
		ct[i] = self.elems[i]:codegen()
	end

   -- These quotes give terra the opportunity to generate optimized assembly via the vectorof call
   -- when I dissassembled terra functions at the console, they seemed more likely to use vector
   -- ops for loading values if vectors are initialized this way.
	if #ct == 3 then
		return quote var [v] = vectorof([tp], [ct[1]], [ct[2]], [ct[3]]) in [v] end
	elseif #ct == 4 then
		return quote var [v] = vectorof([tp], [ct[1]], [ct[2]], [ct[3]], [ct[4]]) in [v] end
	elseif #ct == 5 then
		return quote var [v] = vectorof([tp], [ct[1]], [ct[2]], [ct[3]], [ct[4]], [ct[5]]) in [v] end
	else
		local t = symbol()
		local q = quote
			var [v] : vector(tp, #ct)
			var [t] = &[tp](&s)
		end
		for i = 1, #ct do
			q = quote [q] @[t] = [ct[i]] t = t + 1 end
		end
		return quote [q] in [s] end
	end
end

local function codegen_scalar(node, env)
	local read = symbol()
	local el_type, el_len, val_off, val_len = objTypeToLiszt(node.node_type)
	local lks = node.node_type.luaval.__lkscalar
	local typ = objTypeToTerra(node.node_type)

	return quote
		var [read] : typ
		runtime.lkScalarRead([env.context], [lks], el_type, el_len, val_off, val_len, &[read])
		in
		[read]
	end
end
-- Name:codegen only has to worry about returning r-values,
-- so it can just look stuff up in the environment and return
-- it, since all variables in the liszt scope will be
-- in the environment as a symbol
function ast.Name:codegen (env)
	if type(self.node_type.luaval) == 'table' and self.node_type.luaval.kind == semant._SCALAR_STR then
		return codegen_scalar(self, env)
	end

	local val = env:combinedenv()[self.name]
	if type(val) == 'table' and (val.isglobal) then
		local terra get_global () return val end
		-- store this global in the environment table so we won't have to look it up again
		env:luaenv()[self.name] = get_global()
		val = env:luaenv()[self.name]
		return `val
	-- if we've encountered a Liszt vector, extract a terra vector and store it in the local environment

	elseif type(val) == 'table' and Vector.isVector(val) then
		return val:__codegen()
	end

	return `[val]
end

function ast.TableLookup:codegen (env)
	if type(self.node_type.luaval) == 'table' and self.node_type.luaval.kind == semant._SCALAR_STR then
		return codegen_scalar(self, env)
	else
		return `[self.node_type.luaval]
	end
end

function ast.Number:codegen (env)
	return `[self.value]
end

function ast.Bool:codegen (env)
	if self.value == 'true' then
		return quote in true end 
	else 
		return quote in false end
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

function codegen (luaenv, kernel_ast)
	local env = terralib.newenvironment(luaenv)
	env:enterblock()
	local kernel_body = kernel_ast:codegen(env)

	return terra (ctx : runtime.lkContext)
		var [env.context] = &ctx
		var [env.param] : runtime.lkElement
		if runtime.lkGetActiveElement([env.context], &[env.param]) > 0 then
			kernel_body
		end
	end
end