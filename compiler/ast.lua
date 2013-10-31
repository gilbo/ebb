--[[ Module defines all of the AST nodes used to represent the Liszt 
     language.
]]--

local exports = {}

---------------------------
--[[ Declare AST types ]]--
---------------------------
local AST             = { kind = 'ast', is_liszt_ast = true }
AST.__index           = AST

local LisztKernel     = { kind = 'kernel' }
local Block           = { kind = 'block'  } -- Statement*
  -- store condition and block to be executed for if/elseif clauses
local CondBlock       = { kind = 'condblock' } 

-- Expressions:
local Expression      = { kind = 'expr'   } -- abstract
local BinaryOp        = { kind = 'binop'  }
local Reduce          = { kind = 'reduce' }
local UnaryOp         = { kind = 'unop'   }
local Tuple           = { kind = 'tuple'  }

local TableLookup     = { kind = 'lookup' }
local VectorIndex     = { kind = 'index'  }
local Call            = { kind = 'call'   }

local Name            = { kind = 'name'   }
local Number          = { kind = 'number' }
local String          = { kind = 'string' }
local Bool            = { kind = 'bool'   }
  -- {0, 4, 3} {true, true, false}
local VectorLiteral   = { kind = 'vecliteral' } 

-- non-syntactic expressions (i.e. these will not be generated by the parser)
local Scalar          = { kind = 'scalar'      }
local Field           = { kind = 'field'       }
local FieldAccess     = { kind = 'fieldaccess' }
local Function        = { kind = 'function'    }
local Table           = { kind = 'table'       }

-- Statements:
local Statement       = { kind = 'statement'  }  -- abstract
  -- if expr then block (elseif cond then block)* (else block)? end
local IfStatement     = { kind = 'ifstmt'     }
local WhileStatement  = { kind = 'whilestmt'  }  -- while expr do block end
local DoStatement     = { kind = 'dostmt'     }  -- do block end
local RepeatStatement = { kind = 'repeatstmt' }  -- repeat block until cond
local ExprStatement   = { kind = 'exprstmt'   }  -- e;
local Assignment      = { kind = 'assnstmt'   }  -- "lvalue   = expr" 
local InitStatement   = { kind = 'initstmt'   }  -- "var name = expr"
local DeclStatement   = { kind = 'declstmt'   }  -- "var name"
local NumericFor      = { kind = 'numericfor' }
local GenericFor      = { kind = 'genericfor' }
local Break           = { kind = 'break'      }


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

inherit(BinaryOp,      Expression)
inherit(UnaryOp,       Expression)
inherit(Number,        Expression)
inherit(String,        Expression)
inherit(Bool,          Expression)
inherit(Tuple,         Expression)
inherit(VectorLiteral, Expression)

inherit(Call,          Expression)
inherit(TableLookup,   Expression)
inherit(VectorIndex,   Expression)
inherit(Name,          Expression)
inherit(Reduce,        Expression)

inherit(Scalar,        Expression)
inherit(Field, 	       Expression)
inherit(FieldAccess,   Expression)
inherit(Function,      Expression)
inherit(Table,         Expression)

inherit(IfStatement,     Statement)
inherit(WhileStatement,  Statement)
inherit(DoStatement,     Statement)
inherit(RepeatStatement, Statement)
inherit(ExprStatement,   Statement)
inherit(Assignment,      Statement)
inherit(DeclStatement,   Statement)
inherit(InitStatement,   Statement)
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
	self.linenumber = node.linenumber
	self.filename   = node.filename
	self.offset     = node.offset
end

function AST:DeriveFrom (ast)
	local newnode = setmetatable({kind=self.kind}, self)
	newnode:copy_location(ast)
	return newnode
end

function AST:clone ()
	local copy =
	{
		kind       = self.kind,
		linenumber = self.linenumber,
		filename   = self.filename,
		offset     = self.offset,
	}
	return setmetatable(copy, getmetatable(self))
end

function Tuple:size ( ) return #self.children end

function AST:is (obj)
	return obj == getmetatable(self)
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

function Reduce:pretty_print(indent)
	indent = indent or ''
	print(indent .. self.kind .. ": " .. self.op)
	self.exp:pretty_print(indent .. indent_delta)
end

function Scalar:pretty_print(indent)
	indent = indent or ''
	local name = self.name or ""
	print(indent .. self.kind .. ": " .. name)
end

function Field:pretty_print(indent)
	indent = indent or ''
	local name = self.name or ""
	print(indent .. self.kind .. ": " .. name)
end

function Table:pretty_print (indent)
	indent = indent or ''
	print(indent .. self.kind .. ": " .. tostring(self.table))
end

function Function:pretty_print (indent)
	indent = indent or ''
	print(indent .. self.kind .. ": " .. tostring(self.func))
end

function FieldAccess:pretty_print(indent)
	indent = indent or ''
	print(indent .. self.kind)
	self.field:pretty_print(indent .. indent_delta)
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
	print(indent .. self.kind .. ": (table, member)")
	self.table:pretty_print(indent .. indent_delta)
	self.member:pretty_print(indent .. indent_delta)
end

function VectorIndex:pretty_print (indent)
	indent = indent or ''
	print(indent .. self.kind .. ": (vector, index)")
	self.vector:pretty_print(indent .. indent_delta)
	self.index:pretty_print(indent .. indent_delta)
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
	self.set:pretty_print(indent  .. indent_delta)
	self.body:pretty_print(indent .. indent_delta)
end

function VectorLiteral:pretty_print (indent)
	indent = indent or ''
	print(indent .. self.kind .. ":")
	for i = 1, #self.elems do
		self.elems[i]:pretty_print(indent .. indent_delta)
	end
end

-- declare other exports
for k,v in pairs({
	AST             = AST,
	LisztKernel     = LisztKernel,
	Block           = Block,
	Expression      = Expression,
	BinaryOp        = BinaryOp,
	UnaryOp         = UnaryOp,
	Reduce 	        = Reduce,
	Tuple           = Tuple,
	TableLookup     = TableLookup,
	VectorIndex     = VectorIndex,
	Scalar 	        = Scalar,
	Field           = Field,
	FieldAccess     = FieldAccess,
	Function        = Function,
	Table           = Table,
	Call            = Call,
	Name            = Name,
	Number          = Number,
	String          = String,
	Bool            = Bool,
	VectorLiteral   = VectorLiteral,
	Statement       = Statement,
	IfStatement     = IfStatement,
	WhileStatement  = WhileStatement,
	DoStatement     = DoStatement,
	RepeatStatement = RepeatStatement,
	ExprStatement   = ExprStatement,
	Assignment      = Assignment,
	InitStatement   = InitStatement,
	DeclStatement   = DeclStatement,
	NumericFor      = NumericFor,
	GenericFor      = GenericFor,
	Break           = Break,
	CondBlock       = CondBlock,

}) do exports[k] = v end

return exports

