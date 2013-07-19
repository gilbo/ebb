module(... or 'semant', package.seeall)

symbols = {}

ast = require("ast")

------------------------------------------------------------------------------

--[[ TODO: Implement semantic checking here
--]]
function check(luaenv, kernel_ast)
	--print("AST")
	--terralib.tree.printraw(kernel_ast)
    -- environment for checking variables and scopes
    local env = terralib.newenvironment(luaenv)
    env:enterblock()
    -- block
    for id, node in ipairs(kernel_ast.children) do
        if (node.kind ~= nil) then
            node:check(env)
        end
    end
    env:leaveblock()
    return true
end

------------------------------------------------------------------------------

--[[ Cases not handled
--]]
function ast.AST:check(env)
    print("To implement semantic checking for", self.kind)
end

------------------------------------------------------------------------------

--[[ TODO: Block
--]]
function ast.Block:check(env)
    -- statements
    for id, node in ipairs(self.children) do
        node:check(env)
    end
end

------------------------------------------------------------------------------

--[[ TODO: Statements
--]]
function ast.Statement:check(env)
    -- not needed
    for id, node in ipairs(self.children) do
        node:check(env)
    end
end

function ast.IfStatement:check(env)
    -- condblock and block
    for id, node in ipairs(self.children) do
        node:check(env)
    end
end

function ast.WhileStatement:check(env)
    -- condition expression and block
    for id, node in ipairs(self.children) do
        node:check(env)
    end
end

function ast.DoStatement:check(env)
    -- block
    for id, node in ipairs(self.children) do
        node:check(env)
    end
end

function ast.RepeatStatement:check(env)
    -- condition expression, block
end

function ast.ExprStatement:check(env)
    -- expression
end

function ast.Assignment:check(env)
    -- lhs, rhs expressions
	self.children[1].check(env)
end

function ast.InitStatement:check(env)
    -- name, expression
end

function ast.DeclStatement:check(env)
    -- name
end

function ast.NumericFor:check(env)
end

function ast.GenericFor:check(env)
end

function ast.Break:check(env)
    -- nothing needed
end

function ast.CondBlock:check(env)
    -- condition expression, block
end

------------------------------------------------------------------------------

--[[ TODO: Expressions
--]]
function ast.Expression:check(env)
end

------------------------------------------------------------------------------

--[[ TODO: Misc
--]]
function ast.LValue:check(env)
end

function ast.BinaryOp:check(env)
end

function ast.UnaryOp:check(env)
end

function ast.Tuple:check(env)
end

function ast.TableLookup:check(env)
end

function ast.Call:check(env)
end

------------------------------------------------------------------------------

--[[ TODO: Variables
--]]
function ast.Name:check(env)
end
