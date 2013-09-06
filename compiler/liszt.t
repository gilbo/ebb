package.path = package.path .. ";./compiler/?.lua;./compiler/?.t;./?.lua"

local parser  = require "parser"
local semant  = require "semant"
terralib.require "compiler/codegen"
require "include/liszt"

local Parser = terralib.require('terra/tests/lib/parsing')

local lisztlanguage = {
	name        = "liszt", -- name for debugging
	entrypoints = {"liszt_kernel"},
	keywords    = {"var"},

	expression = function(self, lexer)
		local kernel_ast = Parser.Parse(parser.lang, lexer, "liszt_kernel")
		--[[ this function is called in place of executing the code that 
			 we parsed 
		--]]

		return function (env_fn) 
			local env = env_fn()
			local success = semant.check(env, kernel_ast)
			local kernel  = codegen.codegen(env, kernel_ast)
			return kernel
		end
	end
}

return lisztlanguage
