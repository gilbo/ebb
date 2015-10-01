import "ebb.liszt"
local PN = require 'ebb.lib.pathname'

local U = {}
U.__index = U
package.loaded["devapps.fem.utils"] = U


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

-- Identity matrix
liszt U.getId3()
  return { { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 } }
end

-- Diagonals
liszt U.diagonalMatrix(a)
    return { { a, 0, 0 }, { 0, a, 0 }, { 0, 0, a } }
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

-- Matrix matrix product
liszt U.multiplyMatrices3d(A, B)
  var res : L.mat3d = { { 0, 0, 0 }, { 0, 0, 0 }, { 0, 0, 0 } }
  for i = 0, 3 do
    for j = 0, 3 do
      res[i,j] += A[i,0] * B[0,j] + A[i,1] * B[1,j] + A[i,2] * B[2,j]
    end
  end
  return res
end

-- Determinant of 3X3 matrix
liszt U.detMatrix3d(M)
  var res = ( M[0,0] * ( M[1,1] * M[2,2] - M[1,2] * M[2,1] ) +
              M[0,1] * ( M[1,2] * M[2,0] - M[2,2] * M[1,0] ) +
              M[0,2] * ( M[1,0] * M[2,1] - M[1,1] * M[2,0] )
            )
  return res
end

-- Invert 3X3 matrix
liszt U.invertMatrix3d(M)
  var det  = U.detMatrix3d(M)
  var invdet = 1.0/det
  var res  = { { (M[1,1] * M[2,2] - M[1,2] * M[2,1]) * invdet,
                 (M[2,1] * M[0,2] - M[2,2] * M[0,1]) * invdet,
                 (M[0,1] * M[1,2] - M[0,2] * M[1,1]) * invdet},
               { (M[1,2] * M[2,0] - M[1,0] * M[2,2]) * invdet,
                 (M[2,2] * M[0,0] - M[2,0] * M[0,2]) * invdet,
                 (M[0,2] * M[1,0] - M[0,0] * M[1,2]) * invdet},
               { (M[1,0] * M[2,1] - M[1,1] * M[2,0]) * invdet,
                 (M[2,0] * M[0,1] - M[2,1] * M[0,0]) * invdet,
                 (M[0,0] * M[1,1] - M[0,1] * M[1,0]) * invdet} }
  return res
end

-- Transpose of a matrix
liszt U.transposeMatrix3(M)
  var res = { { M[0,0], M[1,0], M[2,0] },
              { M[0,1], M[1,1], M[2,1] },
              { M[0,2], M[1,2], M[2,2] } } 
  return res
end
