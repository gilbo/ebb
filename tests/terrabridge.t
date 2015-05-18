--DISABLE-ON-GPU  (b/c standard lib functions can't be embedded in CUDA code)
import "compiler.liszt"

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

local test_terra = liszt(r : R)
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
