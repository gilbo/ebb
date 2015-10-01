--this file is used to extend the terra parser with
--the liszt language
--use it like: import "ebb.src.liszt"

local P = require "ebb.src.parser"

local lisztlib = require "ebb.src.lisztlib"
local specialization = require "ebb.src.specialization"
local semant = require "ebb.src.semant"
-- include liszt library for programmer
L = lisztlib

local statement = function(self, lexer)
    local ast, assign = P.ParseStatement(lexer)
    local constructor
    if ast.kind == 'UserFunction' then
        constructor = function (env_fn)
            local env = env_fn()
            return lisztlib.NewUserFunc(ast, env)
        end
    else -- quote
        error("Expected liszt function!", 2)
    end
    return constructor, assign
end

local lisztlanguage = {
    name        = "liszt", -- name for debugging
    entrypoints = {"liszt"},
    -- including max and min as keywords is necessary to get
    -- the parser to interpret them as operators.  This has the
    -- unfortunate affect of not allowing anyone to use 'min' or 'max
    -- as variable names within Liszt code.
    keywords    = {
        "quote",
        "max", "min",
        "var",
        "insert", "into", "delete",
        "_", -- want to hold onto this token in case it's useful
    },

    -- Liszt quotes and anonymous functions
    expression = function (self, lexer)
        local ast = P.ParseExpression(lexer)
        if ast.kind == 'UserFunction' then
            return function (env_fn)
                local env = env_fn()
                return lisztlib.NewUserFunc(ast, env)
            end
        else -- quote
            return function (env_fn)
                local env = env_fn()
                local specialized = specialization.specialize(env, ast)
                local checked     = semant.check(specialized)
                return checked
            end
        end
    end,

    -- named Liszt functions
    statement = statement,
    localstatement = statement,
}

return lisztlanguage
