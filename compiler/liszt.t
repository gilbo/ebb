local parser  = require "compiler/parser"
local kernel  = terralib.require "compiler/kernel"
terralib.require "include/liszt" -- included for liszt programmer


local pratt = terralib.require('compiler/pratt')

local lisztlanguage = {
	name        = "liszt", -- name for debugging
	entrypoints = {"liszt_kernel"},
	keywords    = {"var", "assert", "print"},

	expression = function(self, lexer)
		local kernel_ast = pratt.Parse(parser.lang, lexer, "liszt_kernel")

		return function (env_fn) 
			local env = env_fn()
			return kernel.Kernel.new(kernel_ast, env)
		end
	end
}

return lisztlanguage
