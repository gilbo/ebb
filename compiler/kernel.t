local exports = {}

local semant = require "semant"
terralib.require "include/liszt"
local codegen = terralib.require "compiler/codegen"
terralib.require "runtime/liszt"

-- Keep imports from polluting global scope of any file that includes this module
local runtime = runtime
_G.runtime = nil

local Kernel = { }
Kernel.__index = Kernel

function Kernel.isKernel (obj)
	return getmetatable(obj) == Kernel
end

function Kernel.new (kernel_ast, env)
	return setmetatable({ast=kernel_ast,env=env}, Kernel)
end

function Kernel:acceptsType (param_type)
	return true
end

function Kernel:generate (param_type)
	local new_ast = semant.check(self.env, self.ast)
  -- Typechecker goal: get this following line to not break
  --                   any tests
  -- self.ast = new_ast

	if not self.__kernel then
		self.__kernel = codegen.codegen(self.env, self.ast)
	end
	return self.__kernel
end


exports.Kernel = Kernel

return exports

