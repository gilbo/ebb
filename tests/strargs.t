import 'ebb'
local L = require 'ebblib'
require "tests/test"

local cells = L.NewRelation { size = 10, name = 'cells' }
cells:NewField('position', L.vec2d):Load(function(i)
  local x = i%5
  local y = math.floor(i/5)
  return {x,y}
end)
cells:NewField('temp',L.double):Load(0)
cells:NewField('pres',L.double):Load(0)


local assert, length, print, dot = L.assert, L.length, L.print, L.dot


----------------------
-- Some Valid cases --
----------------------

local ebb check0(c : cells, field)
  var fval = c[field]
  L.assert(fval == 0)
end
local ebb check1(c : cells, field)
  var fval = c[field]
  L.assert(fval == 1)
end

local ebb copyfield(c : cells, fdst, fsrc)
  c[fdst] = c[fsrc]
end

local ebb noargs(c : cells) end

cells:foreach(check0, 'temp')
cells:foreach(check0, 'pres')

-- write 1 to temp and then copy back over the 0s
-- check that each operation worked correctly
cells.temp:Load(1)
cells:foreach(check1, 'temp')
cells:foreach(copyfield, 'temp', 'pres')
cells:foreach(check0, 'temp')

-----------------------------------------
-- Checks for the right # of arguments --
-----------------------------------------
test.fail_function(function()
  cells:foreach(check0)
end, "Function was expecting 1 arguments, but got 0")

test.fail_function(function()
  cells:foreach(check0, 'temp', 'pres')
end, "Function was expecting 1 arguments, but got 2")

test.fail_function(function()
  cells:foreach(copyfield, 'temp')
end, "Function was expecting 2 arguments, but got 1")

test.fail_function(function()
  cells:foreach(noargs, 'temp')
end, "Function was expecting 0 arguments, but got 1")

-----------------------------------
-- Disallow non-string arguments --
-----------------------------------
test.fail_function(function()
  local ebb test(c : cells, val : L.double)
    c.temp = val
  end
  cells:foreach(test, 3)
end, "Argument .* was expected to be a string; Secondary arguments to "..
     "functions mapped over relations must be strings.")

test.fail_function(function()
  local ebb test(c : cells, val : L.double)
    c.temp = val
  end
  cells:foreach(test, 'temp')
end, "Secondary string arguments to functions should be untyped arguments")


---------------------------------
-- Invalid indexing into a key --
---------------------------------
test.fail_function(function()
  local ebb test(c : cells, field)
    c[0] = 2
  end
  cells:foreach(test, 'temp')
end, "Expecting string literal to index key")


-----------------------------------
-- Indexing with string literals --
-----------------------------------
local ebb checktemp0(c : cells)
  var fval = c['temp']
  L.assert(fval == 0)
end
cells:foreach(checktemp0)

local pres_str = 'pres'
local ebb checkpres0(c : cells)
  var fval = c[pres_str]
  L.assert(fval == 0)
end
cells:foreach(checkpres0)

--------------------------------------------------
-- Assignment and nested calls with string args --
--------------------------------------------------
test.fail_function(function()
  local ebb test(c : cells, field)
    var fcopy = field
  end
  cells:foreach(test, 'temp')
end,'can only assign numbers, bools, or keys to local temporaries')

local ebb passthrough(c : cells, field)
  check0(c, field)
end
cells:foreach(passthrough, 'temp')

local m0check = L.Macro(function(c, f)
  return ebb quote
    L.assert(c[f] == 0)
  in 0 end
end)

local ebb macropassthrough(c : cells, field)
  m0check(c, field)
end
cells:foreach(macropassthrough, 'temp')







