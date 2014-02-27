--this file is used to extend the terra parser with
--the liszt language
--use it like: import "compiler.liszt"

local P = require "compiler.parser"

local lisztlib = terralib.require "compiler.lisztlib"
local semant = terralib.require "compiler.semant"
-- include liszt library for programmer
L = lisztlib

local lisztlanguage = {
	name        = "liszt", -- name for debugging
	entrypoints = {"liszt_kernel", "liszt"},
	keywords    = {"var", "kernel", "quote"},

	expression = function(self, lexer)
		local ast = P.Parse(lexer)
        if ast.kind == 'kernel' then
            return function (env_fn) 
                local env = env_fn()
                return lisztlib.NewKernel(ast, env)
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
