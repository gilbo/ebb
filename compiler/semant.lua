module(... or 'semant', package.seeall)

symbols = {}

ast = require("ast")

------------------------------------------------------------------------------
--[[ Useful functions
--]]

function table.contains(t, element)
    for key, value in pairs(t) do
        if key == element then
            return true
        else
            return false
        end
    end
end

------------------------------------------------------------------------------

--[[ TODO: Implement semantic checking here
--]]
function check(kernel_ast, env)
    -- print ("Semantic checking")
    -- environment for checking variables and scopes
    local lisztenv = terralib.newenvironment(env)
    lisztenv:enterblock()
    for key, value in pairs(_G) do
        if type(value) == "table" and table.contains(value, "kind") then
            -- print("Global:", key, value.kind)
        end
    end
    -- block
    for id, node in ipairs(kernel_ast.children) do
        if (node.kind ~= nil) then
            node:check()
        end
    end
    lisztenv:leaveblock()
    return true
end

------------------------------------------------------------------------------

--[[ Cases not handled
--]]
function ast.AST:check()
    print("To implement semantic checking for", self.kind)
end

------------------------------------------------------------------------------

--[[ TODO: Block
--]]
function ast.Block:check()
    -- statements
    for id, node in ipairs(self.children) do
        node:check()
    end
end

------------------------------------------------------------------------------

--[[ TODO: Statements
--]]
function ast.Statement:check()
    -- not needed
    for id, node in ipairs(self.children) do
        node:check()
    end
end

function ast.IfStatement:check()
    -- condblock and block
    for id, node in ipairs(self.children) do
        node:check()
    end
end

function ast.WhileStatement:check()
    -- condition expression and block
    for id, node in ipairs(self.children) do
        node:check()
    end
end

function ast.DoStatement:check()
    -- block
    for id, node in ipairs(self.children) do
        node:check()
    end
end

function ast.RepeatStatement:check()
    -- condition expression, block
end

function ast.ExprStatement:check()
    -- expression
end

function ast.Assignment:check()
    -- lhs, rhs expressions
end

function ast.InitStatement:check()
    -- name, expression
end

function ast.DeclStatement:check()
    -- name
end

function ast.NumericFor:check()
end

function ast.GenericFor:check()
end

function ast.Break:check()
    -- nothing needed
end

function ast.CondBlock:check()
    -- condition expression, block
end

------------------------------------------------------------------------------

--[[ TODO: Expressions
--]]
function ast.Expression:check()
end

------------------------------------------------------------------------------

--[[ TODO: Misc
--]]
function ast.LValue:check()
end

function ast.BinaryOp:check()
end

function ast.UnaryOp:check()
end

function ast.Tuple:check()
end

function ast.TableLookup:check()
end

function ast.Call:check()
end

------------------------------------------------------------------------------

--[[ TODO: Variables
--]]
function ast.Name:check()
end
