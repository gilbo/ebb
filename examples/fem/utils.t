import "compiler.liszt"
local PN = L.require 'lib.pathname'

local U = {}
U.__index = U
package.loaded["examples.fem.utils"] = U


--------------------------------------------------------------------------------
--[[                    Timer for timing execution time                     ]]--
--------------------------------------------------------------------------------

local Timer = {}
Timer.__index = Timer
U.Timer = Timer

function Timer.New()
  local timer = { start = 0, finish = 0, total = 0 }
  setmetatable(timer, Timer)
  return timer
end

function Timer:Reset()
  self.start = 0
  self.finish = 0
  self.total = 0
end

function Timer:Start()
  self.start = terralib.currenttimeinseconds()
end

function Timer:Pause()
  self.finish = terralib.currenttimeinseconds()
  self.total = self.total + self.finish - self.start
  self.start = 0
end

function Timer:Stop()
  self.finish = terralib.currenttimeinseconds()
  local total = self.total + self.finish - self.start
  self.start = 0
  self.finish = 0
  self.total = 0
  return total
end

function Timer:GetTime()
  return self.total
end


--------------------------------------------------------------------------------
--[[                     Helper functions and kernels                       ]]--
--------------------------------------------------------------------------------

-- Compute absolute value for a given variable
liszt U.fabs(num)
  var result = num
  if num < 0 then result = -num end
  return result
end

-- Identity matrix
liszt U.getId3()
  return { { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 } }
end

-- Matrix with all entries equal to value v
liszt U.constantMatrix3(v)
  return { { v, v, v }, { v, v, v }, { v, v, v } }
end

-- Tensor product of 2 vectors
liszt U.tensor3(a, b)
  var result = { { a[0] * b[0], a[0] * b[1], a[0] * b[2] },
                 { a[1] * b[0], a[1] * b[1], a[1] * b[2] },
                 { a[2] * b[0], a[2] * b[1], a[2] * b[2] } }
  return result
end

-- Matrix vector product
liszt U.multiplyMatVec3(M, x)
  return  { M[0, 0]*x[0] + M[0, 1]*x[1] + M[0, 2]*x[2],
            M[1, 0]*x[0] + M[1, 1]*x[1] + M[1, 2]*x[2],
            M[2, 0]*x[0] + M[2, 1]*x[1] + M[2, 2]*x[2] }
end
liszt U.multiplyVectors(x, y)
  return { x[0]*y[0], x[1]*y[1], x[2]*y[2]  }
end

-- Diagonals
liszt U.diagonalMatrix(a)
    return { { a, 0, 0 }, { 0, a, 0 }, { 0, 0, a } }
end
