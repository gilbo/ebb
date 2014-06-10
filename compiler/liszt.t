--this file is used to extend the terra parser with
--the liszt language
--use it like: import "compiler.liszt"

local P = require "compiler.parser"

local lisztlib = terralib.require "compiler.lisztlib"
local specialization = terralib.require "compiler.specialization"
local semant = terralib.require "compiler.semant"
-- include liszt library for programmer
L = lisztlib

local lisztlanguage = {
    name        = "liszt", -- name for debugging
    entrypoints = {"liszt"},
    -- including max and min as keywords is necessary to get
    -- the parser to interpret them as operators.  This has the
    -- unfortunate affect of not allowing anyone to use 'min' or 'max
    -- as variable names within Liszt code.
    keywords    = {
        "kernel", "quote",
        "max", "min",
        "var",
        "insert", "into", "delete",
    },

    expression = function(self, lexer)
        local ast = P.Parse(lexer)
        if ast.kind == 'LisztKernel' then
            return function (env_fn) 
                local env = env_fn()
                return lisztlib.NewKernel(ast, env)
            end
        elseif ast.kind == 'UserFunction' then
            return function (env_fn)
                local env = env_fn()
                return lisztlib.NewUserFunc(ast, env)
            end
        else -- quote
            return function (env_fn)
                local env = env_fn()

                local specialized   = specialization.specialize(env, ast)
                local checked       = semant.check(env, specialized)

                return checked
            end
        end
    end
}

return lisztlanguage
