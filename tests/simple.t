import "compiler.liszt"

local LMesh = L.require "domains.lmesh"
local mesh = LMesh.Load("examples/mesh.lmesh")

mesh.faces:NewField('field', L.float)
mesh.faces.field:LoadConstant(0)

local lassert, lprint, length = L.assert, L.print, L.length

local a     = 43
local com   = L.Global(L.vector(L.float, 3), {0, 0, 0})--Vector.new(float, {0.0, 0.0, 0.0})
local upval = 5
local vv = L.Constant(L.vec3i, {1,2,3})

local luavalue_true = true

local liszt kernel test_bool (v : mesh.vertices)
    var q = true
    var x = q  -- Also, test re-declaring variables (symbols for 'x' should now be different)
    var z = not q
    var t = not not q
    var y = z == false
    lassert(x == true)
    lassert(q == true)
    lassert(z == false)
    lassert(y == true)
    lassert(t == q)
    -- make sure that specialization preserves booleans correctly
    -- had to hunt down this bug once already
    lassert(luavalue_true)
end
test_bool(mesh.vertices)

local liszt kernel test_decls (v : mesh.vertices)
    -- DeclStatement tests --
    var c : L.int
    c = 12

    var x : L.bool
    x = true

    var z : L.bool
    do z = true end
    lassert(z == true)

    var z : L.int
    do z = 4 end
    lassert(z == 4)

    -- this should be fine
    var y : L.vector(L.float, 4)
    var y : L.int

    -- should be able to assign w/an expression after declaring,
    -- checking with var e to make sure expressions are the same.
    var zip = 43.3
    var doo : L.double
    doo = zip * c
    var dah = zip * c
    var x = doo == dah
    lassert(doo == dah)
end
test_decls(mesh.vertices)


local liszt kernel test_conditionals (v : mesh.vertices)
    -- IfStatement tests
    var q = true
    var x = 3
    var y = 4

    if q then
        var x = 5
        lassert(x == 5)
        y = x
        lassert(y == 5)
    else
        y = 9
        lassert(y == 9)
    end

    lassert(x == 3)
    lassert(y == 5)
    y = x
    lassert(y == 3)

    if y == x * 2 then
        x = 4
    elseif y == x then
        x = 5
    end
    lassert(x == 5)

    if y == x * 2 then
        x = 4
    end
    lassert(x == 5)

    var a = 3
    if y == x * 2 then
        x = 4
        lassert(false)
    elseif y == x then
        x = 5
        lassert(false)
    else
        var a = true
        lassert(a == true)
    end
    lassert(a == 3)
end

test_conditionals(mesh.vertices)

local liszt kernel test_arith(v : mesh.vertices)
    -- BinaryOp, UnaryOp, InitStatement, Number, Bool, and RValue codegen tests
    var x = 9
    lassert(x == 9)
    var xx = x - 4
    lassert(xx == 5)
    var y = x + -(6 * 3)
    lassert(y == -9)
    var z = upval
    lassert(z == 5)
    var b = a
    lassert(b == 43)
    b += 5
    lassert(b == 48)
    var q = true
    lassert(q == true)
    var x = q  -- Also, test re-declaring variables (symbols for 'x' should now be different)
    lassert(x == true)
    var z = not x
    lassert(z == false)
    var y = not z or x
    lassert(y == true)
    var z = not true and false or true
    lassert(z == true)

    -- Codegen for vectors (do types propagate correctly?)
    var x = 3 * vv
    lassert(x == {3,6,9})

    var y = vv / 4.2
    var z = x + y
    var a = y - x

    var a = L.float(43.3)
    var d : L.vector(L.float, 3)
    d = a * vv
    var e = a * vv
    lassert(length(d - e) < 1e-04) -- d is float, e is double, so they won't be exact

    var f : L.vector(L.double, 3)
    f = a * vv
    lassert(f == e)
end
test_arith(mesh.vertices)

local liszt kernel test_while (v : mesh.vertices)
    -- While Statement tests --
    -- if either of these while statements doesn't terminate, then our codegen scoping is wrong!
    var a = true
    while a do
        a = false
    end

    var b = true
    while b ~= a do
        a = true
        var b = false
    end
end
test_while(mesh.vertices)


local liszt kernel test_do (v : mesh.vertices)
    var b = false
    var x = true
    var y = 3
    do
        var x = false
        lassert(x == false)
        if x then
            y = 5
            lassert(false)
        else
            y = 4
            lassert(y == 4)
        end
    end
    lassert(x == true)
    lassert(y == 4)
end
test_do(mesh.vertices)


local liszt kernel test_repeat (v : mesh.vertices)
    -- RepeatStatement tests -- 
    var x = 0
    var y = 0
    -- again, if this doesn't terminate, then our scoping is wrong
    repeat
        y = y + 1
        var x = 5
    until x == 5
    lassert(x == 0)

    y = 0
    repeat
        y = y + 1
    until y == 5
    lassert(y == 5)
end
test_repeat(mesh.vertices)


local liszt kernel test_for (v : mesh.vertices)
    -- Numeric for tests: --
    var x = true
    for i = 1, 5 do
        var x = i
        if x == 3 then break end
    end
    lassert(x == true)
end
test_for(mesh.vertices)
