import 'ebb'
local L = require 'ebblib'

local ioOff = require 'ebb.domains.ioOff'
local mesh  = ioOff.LoadTrimesh('tests/octa.off')

local assert, length, print, dot = L.assert, L.length, L.print, L.dot

---------------------------------------
--Test ordinary (function-like) macro--
---------------------------------------
local square = L.Macro(function(x)
    return ebb `x*x
end)

mesh.vertices:foreach(ebb(v : mesh.vertices)
	assert(square(7) == 49)
end)

----------------------------
--Test stacked macro calls--
----------------------------
local sum = L.Macro(function(x, y, z)
    return ebb `x + y + z
end)
mesh.vertices:foreach(ebb(v : mesh.vertices)
    assert(sum(square(1), square(2), square(3)) == 14)
end)

----------------------------------------
--Test macro that behaves like a field--
----------------------------------------
mesh.vertices:NewFieldMacro('scaledpos', L.Macro(function(v)
    return ebb `2*v.pos 
end))
mesh.vertices:foreach(ebb(v : mesh.vertices)
    assert(v.scaledpos == 2 * v.pos)
end)

-----------------------------------
--Combine normal and field macros--
-----------------------------------
local norm = L.Macro(function(v)
    return ebb `dot(v, v)
end)
mesh.vertices:foreach(ebb(v : mesh.vertices)
    var lensq = norm(v.scaledpos)
    var expected = 4.0 * length(v.pos) * length(v.pos)
    assert(square(lensq - expected) < 0.00005)
end)

--------------------------------------
--Test Macros Using Let-Style Quotes--
--------------------------------------
local sub1_but_non_neg = L.Macro(function(num)
    return ebb quote
        var result = num - 1
        if result < 0 then result = 0 end
    in
        result
    end
end)
mesh.vertices:foreach(ebb (v : mesh.vertices)
    assert(sub1_but_non_neg(2) == 1)
    assert(sub1_but_non_neg(0) == 0)
end)


-----------------------------
--Test special Apply Macro --
-----------------------------
mesh.vertices:NewField('temperature', L.double):Load(1.0)
mesh.vertices:NewFieldMacro('__apply_macro', L.Macro(function(v, scale)
    return ebb `scale * v.temperature
end))
mesh.vertices:foreach(ebb (v : mesh.vertices)
    assert(v(1) == 1)
    assert(v(5) == 5)
end)




