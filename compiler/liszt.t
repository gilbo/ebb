local parser  = require "compiler/parser"
local kernel  = terralib.require "compiler/kernel"

-- include liszt library for programmer
L = terralib.require "include/liszt"

-- export builtins into L namespace
local builtins = terralib.require "include/builtins"
builtins.addBuiltinsToNamespace(L)

local pratt = terralib.require('compiler/pratt')

local lisztlanguage = {
	name        = "liszt", -- name for debugging
	entrypoints = {"liszt_kernel"},
	keywords    = {"var"},

	expression = function(self, lexer)
		local kernel_ast = pratt.Parse(parser.lang, lexer, "liszt_kernel")

		return function (env_fn) 
			local env = env_fn()
			return kernel.Kernel.new(kernel_ast, env)
		end
	end
}

return lisztlanguage
