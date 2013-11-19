import "compiler.liszt"
local LMesh = terralib.require "compiler.lmesh"
local mesh = LMesh.Load("examples/mesh.lmesh")

local assert, length, print, dot = L.assert, L.length, L.print, L.dot

---------------------------------------
--Test ordinary (function-like) macro--
---------------------------------------
local square = L.NewMacro(function(x)
    return liszt `x*x
end)

(liszt kernel(v in mesh.vertices)
	assert(square(7) == 49)
end)()

----------------------------
--Test stacked macro calls--
----------------------------
local sum = L.NewMacro(function(x, y, z)
    return liszt `x + y + z
end)
(liszt kernel(v in mesh.vertices)
    assert(sum(square(1), square(2), square(3)) == 14)
end)()

----------------------------------------
--Test macro that behaves like a field--
----------------------------------------
mesh.vertices:NewFieldMacro('scaledposition', L.NewMacro(function(v)
    return liszt `2*v.position 
end))
(liszt kernel(v in mesh.vertices)
    assert(v.scaledposition == 2 * v.position)
end)()

-----------------------------------
--Combine normal and field macros--
-----------------------------------
local norm = L.NewMacro(function(v)
    return liszt `dot(v, v)
end)
(liszt kernel(v in mesh.vertices)
    var lensq = norm(v.scaledposition)
    var expected = 4.0 * length(v.position) * length(v.position)
    assert(square(lensq - expected) < 0.00005)
end)()
