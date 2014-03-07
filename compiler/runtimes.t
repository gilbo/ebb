local Runtime = {}
local ast = terralib.require('compiler.ast')

function Runtime:New (str)
	return setmetatable({__string=str}, self)
end

function Runtime:__tostring ()
	return self.__string
end

local singleCore  = Runtime:New('Single Core Runtime')
local gpu         = Runtime:New('GPU Runtime')

local function is_valid_runtime(rt)
	return getmetatable(rt) == Runtime
end

--[[ kernel_body:
    -- ctxt: codegen Context
	-- param: terra symbol representing the kernel parameter
	-- relation: LRelation
	-- body: recursively generated AST that iterates over a single element from the relation
--]]
function singleCore:codegen_kernel_body (ctxt, liszt_kernel, relation)
	local param = ctxt:localenv()[liszt_kernel.name]
	local body  = liszt_kernel.body:codegen(ctxt)

	return quote
		for [param] = 0, [relation]._size do
			[body]
		end
	end
end

function singleCore:codegen_field_write (ctxt, fw)
	-- just re-direct to an assignment statement for now.
	local assign = ast.Assignment:DeriveFrom(fw)
	assign.lvalue = fw.fieldaccess
	assign.exp    = fw.exp
	if fw.reduceop then assign.reduceop = fw.reduceop end

	return assign:codegen(ctxt)
end

function singleCore:codegen_field_read (ctxt, fa)
	local field = fa.field
	local index = fa.row:codegen(ctxt)
	return `@(field.data + [index])
end

return {
	singleCore       = singleCore,
	gpu              = gpu,
	is_valid_runtime = is_valid_runtime
}
