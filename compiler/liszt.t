package.path = package.path .. ";./compiler/?.lua;./compiler/?.t;./?.lua"

local parser  = require "parser"
terralib.require "compiler/kernel"
terralib.require "include/liszt"

local Parser = terralib.require('terra/tests/lib/parsing')

local lisztlanguage = {
	name        = "liszt", -- name for debugging
	entrypoints = {"liszt_kernel"},
	keywords    = {"var"},

	expression = function(self, lexer)
		local kernel_ast = Parser.Parse(parser.lang, lexer, "liszt_kernel")

		return function (env_fn) 
			local env = env_fn()
			return kernel.Kernel.new(kernel_ast, env)
		end
	end
}

return lisztlanguage
