local Codegen = {}
package.loaded["compiler.codegen_common"] = Codegen

-- state should be stored in ctxt.gpu and ctxt.reduce
local Context = {}
Codegen.Context = Context
Context.__index = Context

function Context.new(env, bran)
    local ctxt = setmetatable({
        env  = env,
        bran = bran,
    }, Context)
    return ctxt
end
function Context:initializeGPUState(block_size)
  self.gpu = GPUContext.New(self, block_size)
  self.reduce = ReductionCtx.New(self, self.gpu:blockSize())
end

function Context:localenv()
  return self.env:localenv()
end
function Context:enterblock()
  self.env:enterblock()
end
function Context:leaveblock()
  self.env:leaveblock()
end

function Context:onGPU()
  return self.bran.location == L.GPU
end

function Context:fieldPhase(field)
  return self.bran.kernel.field_use[field]
end

function Context:globalPhase(global)
  return self.bran.kernel.global_use[global]
end
