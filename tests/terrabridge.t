import "compiler/liszt"

local assert = L.assert
mesh = LoadMesh("examples/mesh.lmesh")

local sqrt = terralib.includec('math.h').sqrt
local ans = sqrt(5)
local srand = terralib.includec('stdlib.h').srand -- just looking for a void function somewhere
local printf = terralib.includec('stdio.h').printf

local terra print_int(val : int)
    printf('%d\n', val)
end

local terra square(val : int)
    return val * val
end

local test_sqrt = liszt_kernel(f)
    assert(square(5) == 25) -- call a user-defined Terra function
    assert(sqrt(5) == ans) -- call a built-in C function

    var sq5 = square(5) -- participate in assignment
    assert(sq5 == 25)

    print_int(3) -- correctly handle a user-defined Terra function with no return value
    srand(2) -- correctly handle a built-in C function with void return type
end

mesh.faces:map(test_sqrt)
