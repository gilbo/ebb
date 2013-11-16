local exports = {}

local semant  = require "semant"
local codegen = terralib.require "compiler/codegen"

-- Keep imports from polluting global scope of any file that includes this module

local Kernel = { }
Kernel.__index = Kernel
Kernel.__call  = function (kobj)
	if not kobj.__kernel then kobj:generate() end
	kobj.__kernel()
end

function Kernel.isKernel (obj)
	return getmetatable(obj) == Kernel
end

function Kernel.new (kernel_ast, env)
	return setmetatable({ast=kernel_ast,env=env}, Kernel)
end

function Kernel:generate (param_type)
  self.typed_ast = semant.check(self.env, self.ast)

	if not self.__kernel then
		self.__kernel = codegen.codegen(self.env, self.typed_ast)
	end
end

exports.Kernel = Kernel

return exports

