local parser  = require "compiler/parser"
terralib.require "compiler/kernel"
terralib.require "include/liszt" -- included for liszt programmer

-- Keep kernel out of global scope for liszt programmer
local kernel = kernel
_G.kernel    = nil


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
