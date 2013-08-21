module(... or 'codegen', package.seeall)

ast = require('ast')

function ast.AST:codegen_kernel (env)
	error("Codegen not implemented for AST node " .. self.kind)
end

function ast.AST:codegen_prologue (env)
	error("Kernel prologue codegen not implemented for AST node " .. self.kind)
end

function ast.AST:codegen_epilogue (env)
	error("Kernel epilogue codegen not implemented for AST node " .. self.kind)
end

function codegen (luaenv, kernel_ast)
	local env = terralib.newenvironment(luaenv)

	--[[
	prologue = kernel_ast:codegen_prologue(env)
	kernel   = kernel_ast:codegen_kernel(env)
	epilogue = kernel_ast:codegen_epilogue(env)

	return prologue, kernel, epilogue
	]]--

	return nil
end