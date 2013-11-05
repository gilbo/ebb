local parser  = require "compiler/parser"
local semant  = require "compiler/semant"
local kernel  = terralib.require "compiler/kernel"

-- include liszt library for programmer
L = terralib.require "include/liszt"

-- export builtins into L namespace
local builtins = terralib.require "include/builtins"
builtins.addBuiltinsToNamespace(L)

local pratt = terralib.require('compiler/pratt')

local lisztlanguage = {
	name        = "liszt", -- name for debugging
	entrypoints = {"liszt_kernel", "liszt"},
	keywords    = {"var", "kernel"},

	expression = function(self, lexer)
		local ast = pratt.Parse(parser.lang, lexer, "liszt")

        if ast.kind == 'kernel' then
            return function (env_fn) 
                local env = env_fn()
                return kernel.Kernel.new(ast, env)
            end
        else
            return function (env_fn)
                local env = env_fn()
                return semant.check(env, ast)
            end
        end
	end
}

return lisztlanguage
