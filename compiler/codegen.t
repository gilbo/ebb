module(... or 'codegen', package.seeall)

ast = require 'ast'
terralib.require 'runtime/liszt'
local runtime = runtime

function ast.AST:codegen (env)
	error("Codegen not implemented for AST node " .. self.kind)
end

function ast.LisztKernel:codegen (env)
	env:localenv()[self.children[1].children[1]] = symbol()
	env.context = symbol() -- lkContext* argument for kernel function
	return self.children[2]:codegen()

end

function ast.Block:codegen (env)
	

end

-- terralib.tree.printraw(ast.LisztKernel)

function codegen (luaenv, kernel_ast)
	local env   = terralib.newenvironment(luaenv)
	env:enterblock()
	local kernel_code = kernel_ast:codegen(env)
	local kernel_fn = terra ([env.context] : runtime.lkContext) : {}

	end

	return kernel_fn
end