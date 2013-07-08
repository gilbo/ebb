module(... or 'semant', package.seeall)

ast = require("ast")

--[[ TODO: Implement semantic checking here
--]]
function check(kernel_ast)
	for id, node in ipairs(kernel_ast.children) do
		if (node.kind ~= nil) then
			node:check()
		end
	end
	return true
end

--[[ Cases not handled
--]]
function ast.AST:check()
	print("To implement semantic checking for", self.kind)
end

--[[ Block
--]]
function ast.Block:check()
	for id, node in ipairs(self.children) do
		node:check()
	end
end

--[[ Statements
--]]
function ast.Statement:check()
end

function ast.IfStatement:check()
end

function ast.WhileStatement:check()
end

function ast.DoStatement:check()
end

function ast.RepeatStatement:check()
end

function ast.ExprStatement:check()
end

function ast.Assignment:check()
end

function ast.InitStatement:check()
end

function ast.DeclStatement:check()
end

function ast.NumericFor:check()
end

function ast.GenericFor:check()
end

function ast.Break:check()
end

function ast.CondBlock:check()
end

--[[ Expressions
--]]
function ast.Expression:check()
end

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
