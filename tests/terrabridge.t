--DISABLE-ON-GPU  (b/c standard lib functions can't be embedded in CUDA code)

-- The MIT License (MIT)
-- 
-- Copyright (c) 2015 Stanford University.
-- All rights reserved.
-- 
-- Permission is hereby granted, free of charge, to any person obtaining a
-- copy of this software and associated documentation files (the "Software"),
-- to deal in the Software without restriction, including without limitation
-- the rights to use, copy, modify, merge, publish, distribute, sublicense,
-- and/or sell copies of the Software, and to permit persons to whom the
-- Software is furnished to do so, subject to the following conditions:
-- 
-- The above copyright notice and this permission notice shall be included
-- in all copies or substantial portions of the Software.
-- 
-- THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
-- IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
-- FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
-- AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
-- LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
-- FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
-- DEALINGS IN THE SOFTWARE.
import 'ebb'
local L = require 'ebblib'

local assert = L.assert
local R = L.NewRelation { name="R", size=6 }

local sqrt   = terralib.includec('math.h').sqrt
local srand  = terralib.includec('stdlib.h').srand -- just looking for a void function somewhere
local printf = terralib.includec('stdio.h').printf

local ans    = sqrt(5)

local terra print_int(val : int)
  printf('%d\n', val)
end

local terra say_hi()
  printf('Hi!\n')
end

local terra square(val : int)
  return val * val
end

local test_terra = ebb(r : R)
  assert(square(5) == 25) -- call a user-defined Terra function
  assert(sqrt(5) == ans) -- call a built-in C function

  var sq5 = square(5) -- participate in assignment
  assert(sq5 == 25)

  print_int(3) -- correctly handle a user-defined Terra function
               -- with no return value
  say_hi() -- correctly handle a Terra function with no parameters

  srand(2) -- correctly handle a built-in C function with void return type
end
R:foreach(test_terra)
