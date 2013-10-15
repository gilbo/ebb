--[[ Module defines all of the AST nodes used to represent the Liszt 
     language.
]]--
module(... or 'ast', package.seeall)


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

Name           = { kind = 'name'       }
Number         = { kind = 'number'     }
String         = { kind = 'string'     }
Bool           = { kind = 'bool'       }
VectorLiteral  = { kind = 'vecliteral' } -- {0, 4, 3} {true, true, false}

-- Statements:
Statement       = { kind = 'statement'  }  -- abstract
IfStatement     = { kind = 'ifstmt'     }  -- if expr then block (elseif cond then block)* (else block)? end
WhileStatement  = { kind = 'whilestmt'  }  -- while expr do block end
DoStatement     = { kind = 'dostmt'     }  -- do block end
RepeatStatement = { kind = 'repeatstmt' }  -- repeat block until cond
ExprStatement   = { kind = 'exprstmt'   }  -- e;
Assignment      = { kind = 'assnstmt'   }  -- "lvalue   = expr" 
InitStatement   = { kind = 'initstmt'   }  -- "var name = expr"
DeclStatement   = { kind = 'declstmt'   }  -- "var name"
AssertStatement = { kind = 'assertstmt' }  -- "assert(expr)"
PrintStatement  = { kind = 'printstmt'  }  -- "print(expr)" 
NumericFor      = { kind = 'numericfor' }
GenericFor      = { kind = 'genericfor' }
Break           = { kind = 'break'      }

CondBlock = { kind = 'condblock' } -- store condition and block to be executed for if/elseif clauses

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

inherit(LValue,        Expression)
inherit(BinaryOp,      Expression)
inherit(UnaryOp,       Expression)
inherit(Number,        Expression)
inherit(String,        Expression)
inherit(Bool,          Expression)
inherit(Tuple,         Expression)
inherit(VectorLiteral, Expression)

inherit(Call,        LValue)
inherit(TableLookup, LValue)
inherit(Name,        LValue)

inherit(IfStatement,     Statement)
inherit(WhileStatement,  Statement)
inherit(DoStatement,     Statement)
inherit(RepeatStatement, Statement)
inherit(ExprStatement,   Statement)
inherit(Assignment,      Statement)
inherit(DeclStatement,   Statement)
inherit(InitStatement,   Statement)
inherit(AssertStatement, Statement)
inherit(PrintStatement,  Statement)
inherit(NumericFor,      Statement)
inherit(GenericFor,      Statement)
inherit(Break,           Statement)


-----------------------------
--[[ General AST Methods ]]--
-----------------------------
function AST:New (P)
	local newnode = 
	{ 
		kind       = self.kind, 
		linenumber = P:cur().linenumber,
		filename   = P.source,
		offset     = P:cur().offset,
	}
	return setmetatable(newnode, self)
end

function AST:copy_location (node)
	linenumber = node.linenumber
	filename   = node.filename
	offset     = node.offset
end

function AST.isLValue    ( ) return false end
function LValue.isLValue ( ) return true  end

function Tuple:size ( ) return #self.children end

function AST:is (obj)
	return self == getmetatable(obj)
end

---------------------------
--[[ AST tree printing ]]--
---------------------------
local indent_delta = '   '

function LisztKernel:pretty_print (indent)
	indent = indent or ''
	print(indent .. self.kind .. ": (param, body)")
	indent = indent .. indent_delta
	self.param:pretty_print(indent)
	self.body:pretty_print(indent)
end

function Block:pretty_print (indent)
	indent = indent or ''
	print(indent .. self.kind)
	for i = 1, #self.statements do
		self.statements[i]:pretty_print(indent .. indent_delta)
	end
end

function CondBlock:pretty_print (indent)
	print(indent .. self.kind .. ": (cond, block)")
	self.cond:pretty_print(indent .. indent_delta)
	self.body:pretty_print(indent .. indent_delta)
end

function BinaryOp:pretty_print (indent)
	indent = indent or ''
	print(indent .. self.kind .. ": " .. self.op)
	self.lhs:pretty_print(indent .. indent_delta)
	self.rhs:pretty_print(indent .. indent_delta)
end

function UnaryOp:pretty_print (indent)
	indent = indent or ''
	print(indent .. self.kind .. ": " .. self.op)
	self.exp:pretty_print(indent .. indent_delta)
end

function Tuple:pretty_print(indent)
	indent = indent or ''
	print(indent .. self.kind)
	for i = 1, #self.children do
		self.children[i]:pretty_print(indent .. indent_delta)
	end
end

function VectorLiteral:pretty_print(indent)
	indent = indent or ''
	print(indent .. self.kind)
	for i = 1, #self.elems do
		self.elems[i]:pretty_print(indent .. indent_delta)
	end
end

function Call:pretty_print (indent)
	indent = indent or ''
	print(indent .. self.kind .. ": (func, params)")
	self.func:pretty_print(indent .. indent_delta)
	self.params:pretty_print(indent .. indent_delta)
end

function TableLookup:pretty_print (indent)
	indent = indent or ''
	self.table:pretty_print(indent .. indent_delta)
	self.member:pretty_print(indent .. indent_delta)
end

function Name:pretty_print (indent)
	indent = indent or ''
	print(indent .. self.kind .. ": " .. self.name)
end

function Number:pretty_print (indent)
	indent = indent or ''
	print(indent .. self.kind .. ": " .. self.value)
end

function Bool:pretty_print (indent)
	indent = indent or ''
	print(indent .. self.kind .. ':' .. self.value)
end

function String:pretty_print (indent)
	indent = indent or ''
	print(indent .. self.kind .. ": \"" .. self.value .. "\"")
end

function IfStatement:pretty_print (indent)
	indent = indent or ''
	print(indent .. self.kind)
	for i = 1, #self.if_blocks do
		self.if_blocks[i]:pretty_print(indent .. indent_delta)
	end
	if self.else_block then
		self.else_block:pretty_print(indent .. indent_delta)
	end
end

function WhileStatement:pretty_print (indent)
	indent = indent or ''
	print(indent .. self.kind .. ": (condition, body)")
	self.cond:pretty_print(indent .. indent_delta)
	self.body:pretty_print(indent .. indent_delta)
end

function DoStatement:pretty_print (indent)
	indent = indent or ''
	print(indent .. self.kind)
	self.body:pretty_print(indent .. indent_delta)
end

function RepeatStatement:pretty_print (indent)
	indent = indent or ''
	print(indent .. self.kind .. ": (body, condition)")
	self.body:pretty_print(indent .. indent_delta)
	self.cond:pretty_print(indent .. indent_delta)
end

function InitStatement:pretty_print (indent)
	indent = indent or ''
	print(indent .. self.kind .. ": (ref, exp)")
	self.ref:pretty_print(indent .. indent_delta)
	self.exp:pretty_print(indent .. indent_delta)
end

function DeclStatement:pretty_print (indent)
	indent = indent or ''
	print(indent .. self.kind)
	self.ref:pretty_print(indent .. indent_delta)
end

function ExprStatement:pretty_print (indent)
	indent = indent or ''
	print(indent .. self.kind)
	self.exp:pretty_print(indent .. indent_delta)
end

function Assignment:pretty_print (indent)
	indent = indent or ''
	print(indent .. self.kind .. ': (lvalue, exp)')
	self.lvalue:pretty_print(indent .. indent_delta)
	self.exp:pretty_print(indent .. indent_delta)
end

function Break:pretty_print (indent)
	indent = indent or ''
	print(indent .. self.kind)
end

function AssertStatement:pretty_print (indent)
	indent = indent or ''
	print(indent .. self.kind)
	self.test:pretty_print(indent .. indent_delta)
end

function PrintStatement:pretty_print (indent)
	indent = indent or ''
	print(indent .. self.kind)
	self.output:pretty_print(indent .. indent_delta)
end

function NumericFor:pretty_print (indent)	
	indent = indent or ''
	if self.step then
		print(indent .. self.kind .. ": (iter, lower, upper, step, body)")
	else
		print(indent .. self.kind .. ": (iter, lower, upper, body)")
	end
	self.iter:pretty_print(indent .. indent_delta)
	self.lower:pretty_print(indent .. indent_delta)
	self.upper:pretty_print(indent .. indent_delta)
	if self.step then self.step:pretty_print(indent .. indent_delta) end
	self.body:pretty_print(indent .. indent_delta)
end

function GenericFor:pretty_print (indent)
	indent = indent or ''
	print(indent .. self.kind .. ": (iter, set, body)")
	self.iter:pretty_print(indent .. indent_delta)
	self.set:pretty_print(indent .. indent_delta)
	self.body:pretty_print(indent .. indent_delta)
end

function VectorLiteral:pretty_print (indent)
	indent = indent or ''
	print(indent .. self.kind .. ":")
	for i = 1, #self.elems do
		self.elems[i]:pretty_print(indent .. indent_delta)
	end
end
