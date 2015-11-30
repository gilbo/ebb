--this file is used to extend the terra parser with
--the ebb language
--use it like: import "ebb"

-- shim in the coverage analysis
--require 'ebb.src.coverage'

local P = require "ebb.src.parser"

local ebblib            = require "ebblib"
local specialization    = require "ebb.src.specialization"
local semant            = require "ebb.src.semant"
local F                 = require "ebb.src.functions"

local statement = function(self, lexer)
    local ast, assign = P.ParseStatement(lexer)
    local constructor
    if ast.kind == 'UserFunction' then
        constructor = function (env_fn)
            local env = env_fn()
            return F.NewFunction(ast, env)
        end
    else -- quote
        error("Expected ebb function!", 2)
    end
    return constructor, assign
end

local ebblanguage = {
    name        = "ebb", -- name for debugging
    entrypoints = {"ebb"},
    -- including max and min as keywords is necessary to get
    -- the parser to interpret them as operators.  This has the
    -- unfortunate affect of not allowing anyone to use 'min' or 'max
    -- as variable names within Ebb code.
    keywords    = {
        "quote",
        "max", "min",
        "var",
        "insert", "into", "delete",
        "_", -- want to hold onto this token in case it's useful
    },

    -- Ebb quotes and anonymous functions
    expression = function (self, lexer)
        local ast = P.ParseExpression(lexer)
        if ast.kind == 'UserFunction' then
            return function (env_fn)
                local env = env_fn()
                return F.NewFunction(ast, env)
            end
        elseif ast.kind == 'Quote' then -- quote
            return function (env_fn)
                local env = env_fn()
                local specialized = specialization.specialize(env, ast)
                local checked     = semant.check_quote(specialized)
                return checked
            end
        else
            error('INTERNAL: unexpected ebb code type')
        end
    end,

    -- named Ebb functions
    statement = statement,
    localstatement = statement,
}

return ebblanguage
