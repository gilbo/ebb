local C = {}
package.loaded["compiler.codegen"] = C

local ast = require "compiler.ast"

function ast.AST:codegen (ctxt)
	print(debug.traceback())
	error("Codegen not implemented for AST node " .. self.kind)
end

function ast.ExprStatement:codegen (ctxt)
	return self.exp:codegen(ctxt)
end

function ast.LisztKernel:codegen (ctxt)
	local param = symbol(self.node_type:terraType())
	ctxt:localenv()[self.name] = param
	return ctxt:runtime_codegen_kernel_body(self)
end

function ast.Block:codegen (ctxt)
	-- start with an empty ast node, or we'll get an error when appending new quotes below
	local code = quote end
	for i = 1, #self.statements do
		local stmt = self.statements[i]:codegen(ctxt)
		code = quote code stmt end
	end
	return code
end

function ast.CondBlock:codegen(ctxt, cond_blocks, else_block, index)
	index = index or 1

	local cond  = self.cond:codegen(ctxt)
	ctxt:enterblock()
	local body = self.body:codegen(ctxt)
	ctxt:leaveblock()

	if index == #cond_blocks then
		if else_block then
			return quote if [cond] then [body] else [else_block:codegen(ctxt)] end end
		else
			return quote if [cond] then [body] end end
		end
	else
		ctxt:enterblock()
		local nested = cond_blocks[index + 1]:codegen(ctxt, cond_blocks, else_block, index + 1)
		ctxt:leaveblock()
		return quote if [cond] then [body] else [nested] end end
	end
end

function ast.IfStatement:codegen (ctxt)
	return self.if_blocks[1]:codegen(ctxt, self.if_blocks, self.else_block)
end

function ast.WhileStatement:codegen (ctxt)
	local cond = self.cond:codegen(ctxt)
	ctxt:enterblock()
	local body = self.body:codegen(ctxt)
	ctxt:leaveblock()
	return quote while [cond] do [body] end end
end

function ast.DoStatement:codegen (ctxt)
	ctxt:enterblock()
	local body = self.body:codegen(ctxt)
	ctxt:leaveblock()
	return quote do [body] end end
end

function ast.RepeatStatement:codegen (ctxt)
	ctxt:enterblock()
	local body = self.body:codegen(ctxt)
	local cond = self.cond:codegen(ctxt)
	ctxt:leaveblock()

	return quote repeat [body] until [cond] end
end

function ast.NumericFor:codegen (ctxt)
	-- min and max expression should be evaluated in current scope,
	-- iter expression should be in a nested scope, and for block
	-- should be nested again -- that way the loop var is reset every
	-- time the loop runs.
	local minexp  = self.lower:codegen(ctxt)
	local maxexp  = self.upper:codegen(ctxt)
	local stepexp = self.step and self.step:codegen(ctxt) or nil

	ctxt:enterblock()
	local iterstr = self.name
	local itersym = symbol()
	ctxt:localenv()[iterstr] = itersym

	ctxt:enterblock()
	local body = self.body:codegen(ctxt)
	ctxt:leaveblock()
	ctxt:leaveblock()

	if stepexp then
		return quote for [itersym] = [minexp], [maxexp], [stepexp] do [body] end end
	end

	return quote for [itersym] = [minexp], [maxexp] do [body] end end
end

function ast.Break:codegen(ctxt)
	return quote break end
end

function ast.Name:codegen(ctxt)
	local s = ctxt:localenv()[self.name]
	assert(terralib.issymbol(s))
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

function ast.Assignment:codegen (ctxt)
	local lhs   = self.lvalue:codegen(ctxt)
	local ttype = self.lvalue.node_type:terraType()
	local rhs   = self.exp:codegen(ctxt)

	if self.reduceop then
		rhs = bin_exp(self.reduceop, lhs, rhs)
	end
	return quote [lhs] = rhs end
end

function ast.FieldWrite:codegen (ctxt)
	return ctxt:runtime_codegen_field_write(self)
end

function ast.FieldAccess:codegen (ctxt)
	return ctxt:runtime_codegen_field_read(self)
end

function ast.Cast:codegen(ctxt)
    local valuecode = self.value:codegen(ctxt)
    local casttype = self.node_type:terraType()
    return `[casttype](valuecode)
end

-- By the time we make it to codegen, Call nodes are only used to represent builtin function calls.
function ast.Call:codegen (ctxt)
    return self.func.codegen(self, ctxt)
end

function ast.DeclStatement:codegen (ctxt)
	local varname = self.name
	local tp      = self.node_type:terraType()
	local varsym  = symbol(tp)
	ctxt:localenv()[varname] = varsym

	if self.initializer then
		local exp = self.initializer:codegen(ctxt)
		return quote var [varsym] = [exp] end
	else
		return quote var [varsym] end
	end
end

function ast.VectorLiteral:codegen (ctxt)
	local ct = { }
	local v = symbol()
	local tp = self.node_type:terraBaseType()
	for i = 1, #self.elems do
		ct[i] = self.elems[i]:codegen(ctxt)
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

function ast.Scalar:codegen (ctxt)
	local d = self.scalar.data
	local s = symbol(&self.scalar.type:terraType())
	return `@d
end

function ast.VectorIndex:codegen (ctxt)
	local vector = self.vector:codegen(ctxt)
	local index  = self.index:codegen(ctxt)

	return `vector[index]
end

function ast.Number:codegen (ctxt)
	return `[self.value]
end

function ast.Bool:codegen (ctxt)
	if self.value == 'true' then
		return `true
	else 
		return `false
	end
end

function ast.UnaryOp:codegen (ctxt)
	local expr = self.exp:codegen(ctxt)
	if (self.op == '-') then return `-[expr]
	else return `not [expr]
	end
end

function ast.BinaryOp:codegen (ctxt)
	local lhe = self.lhs:codegen(ctxt)
	local rhe = self.rhs:codegen(ctxt)
	return bin_exp(self.op, lhe, rhe)
end

function ast.LuaObject:codegen (ctxt)
    return `{}
end
function ast.Where:codegen(ctxt)
    local key = self.key:codegen(ctxt)
    local sType = self.node_type:terraType()
    local index = self.relation._indexdata
    local v = quote
        var k = [key]
        var idx = [index]
    in 
        sType { idx[k], idx[k+1] }
    end
    return v
end

local function doProjection(obj,field)
    assert(L.is_field(field))
    return `field.data[obj]
end

function ast.GenericFor:codegen (ctxt)
	local set = self.set:codegen(ctxt)
	local iter = symbol("iter")
    local rel = self.set.node_type.relation
    local projected = iter
    for i,p in ipairs(self.set.node_type.projections) do
        local field = rel[p]
        projected = doProjection(projected,field)
        rel = field.type.relation
        assert(rel)
    end
    local sym = symbol(L.row(rel):terraType())
    ctxt:enterblock()
	ctxt:localenv()[self.name] = sym
	local body = self.body:codegen(ctxt)
    ctxt:leaveblock()
    local code = quote
	    var s = [set]
	    for [iter] = s.start,s.finish do
	        var [sym] = [projected]
	        [body]
	    end
	end
	return code
end

local Context = {}
Context.__index = Context

function Context.new(env, runtime)
    local ctxt = setmetatable({
        env     = env,
        runtime = runtime
    }, Context)
    return ctxt
end
function Context:localenv()
	return self.env:localenv()
end
function Context:enterblock()
	self.env:enterblock()
end
function Context:leaveblock()
	self.env:leaveblock()
end
function Context:runtime_codegen_kernel_body (kernel_node)
	return self.runtime:codegen_kernel_body(self, kernel_node)
end
function Context:runtime_codegen_field_write (fw_node)
	return self.runtime:codegen_field_write(self, fw_node)
end
function Context:runtime_codegen_field_read (fa_node)
	return self.runtime:codegen_field_read(self, fa_node)
end


function C.codegen (runtime, luaenv, kernel_ast)
	local env = terralib.newenvironment(luaenv)
	local ctxt = Context.new(env, runtime)

	ctxt:enterblock()
	local kernel_body = kernel_ast:codegen(ctxt)
	ctxt:leaveblock()

	local r = terra ()
		[kernel_body]
	end
	return r
end
