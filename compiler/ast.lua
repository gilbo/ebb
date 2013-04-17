---------------------------
--[[ Declare AST types ]]--
---------------------------
AST            = { kind = 'ast' }
AST.__index    = AST

LisztKernel    = { kind = 'kernel' }
Block          = { kind = 'block'  } -- Statement*

-- Expressions:
Expression     = { kind = 'expr'   } -- abstract
LValue         = { kind = 'lvalue' } -- abstract
BinaryOp       = { kind = 'binop'  }
UnaryOp        = { kind = 'unop'   }
Tuple          = { kind = 'tuple'  }

TableLookup    = { kind = 'lookup' }
Call           = { kind = 'call'   }

Name           = { kind = 'name'   }
Number         = { kind = 'number' }
String         = { kind = 'string' }

-- Statements:
Statement      = { kind = 'statement'  }  -- abstract
IfStatement    = { kind = 'ifstmt'     }  -- if expr then block (elseif cond then block)* (else block)? end
WhileStatement = { kind = 'whilestmt'  }  -- while expr do block end
ExprStatement  = { kind = 'exprstmt'   }  -- e;
Assignment     = { kind = 'assnstmt'   }  -- "blah     = expr"
InitStatement  = { kind = 'initstmt'   }  -- "var blah = expr"
NumericFor     = { kind = 'numericfor' }
GenericFor     = { kind = 'genericfor' }

CondBlock = { kind = 'condblock' } -- store condition and block to be executed...


----------------------------
--[[ Set up inheritance ]]--
----------------------------
local function inherit (child, parent)
	child.__index = child -- readies child as metatable for inheritance
	setmetatable(child, parent)
end

inherit(LisztKernel, AST)
inherit(Expression,  AST)
inherit(Statement,   AST)
inherit(Block,       AST)
inherit(CondBlock,   AST)

inherit(LValue,   Expression)
inherit(BinaryOp, Expression)
inherit(UnaryOp,  Expression)
inherit(Number,   Expression)
inherit(String,   Expression)
inherit(Tuple,    Expression)

inherit(Call,        LValue)
inherit(TableLookup, LValue)
inherit(Name,        LValue)

inherit(IfStatement,    Statement)
inherit(WhileStatement, Statement)
inherit(ExprStatement,  Statement)
inherit(Assignment,     Statement)
inherit(InitStatement,  Assignment)
inherit(NumericFor,     Statement)
inherit(GenericFor,     Statement)


---------------------
--[[ General AST Methods ]]--
---------------------
function AST:New (...)
	return setmetatable({ children = {...}, kind = self.kind }, {__index = self})
end

function AST.isLValue    ( ) return false end
function LValue.isLValue ( ) return true  end

function Tuple:size ( ) return #self.children end


---------------------
--[[ AST tree printing ]]--
---------------------
local indent_delta = '   '

function AST:pretty_print (indent)
	indent = indent or ''
	print(indent .. self.kind .. ":")
	for _, child in ipairs(self.children) do
		child:pretty_print(indent .. indent_delta)
	end
end

function LisztKernel:pretty_print (indent)
	indent = indent or ''
	print(indent .. self.kind .. ":")
	print(indent .. indent_delta .. "(param): " .. self.children[1])
	self.children[2]:pretty_print(indent .. indent_delta)
end

function BinaryOp:pretty_print (indent)
	indent = indent or ''
	print(indent .. self.kind .. ": " .. self.children[2])
	self.children[1]:pretty_print(indent .. indent_delta)
	self.children[3]:pretty_print(indent .. indent_delta)
end

function UnaryOp:pretty_print (indent)
	indent = indent or ''
	print(indent .. self.kind .. ": " .. self.children[1])
	self.children[2]:pretty_print(indent .. indent_delta)
end

function TableLookup:pretty_print (indent)
	indent = indent or ''
	print(indent .. self.kind .. ", op:", self.children[2])
	self.children[1]:pretty_print(indent .. indent_delta)
	self.children[3]:pretty_print(indent .. indent_delta)
end

function Name:pretty_print (indent)
	indent = indent or ''
	print(indent .. self.kind .. ": " .. self.children[1])
end

function Number:pretty_print (indent)
	indent = indent or ''
	print(indent .. self.kind .. ": " .. self.children[1])
end

function String:pretty_print (indent)
	indent = indent or ''
	print(indent .. self.kind .. ": \"" .. self.children[1] .. "\"")
end

function NumericFor:pretty_print (indent)	
	indent = indent or ''
	print(indent .. self.kind .. ":")
end

function GenericFor:pretty_print (indent)
	indent = indent or ''
	print(indent .. self.kind .. ":")
end
