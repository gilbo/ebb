import "ebb"
require "tests/test"

local cells = L.NewRelation { size = 10, name = 'cells' }

cells:NewField('f1', L.double):Load(0)
cells:NewField('f2', L.double):Load(0)

local setup = ebb ( c : cells )
  c.f1 = 5
end
cells:foreach(setup)

cells:Swap('f1','f2')

local check1 = ebb ( c : cells )
  L.assert(c.f2 == 5)
  L.assert(c.f1 == 0)
end
cells:foreach(check1)

cells:Copy{to='f1',from='f2'}

local check2 = ebb ( c : cells )
  L.assert(c.f2 == 5)
  L.assert(c.f1 == 5)
end
cells:foreach(check2)

-- Check for errors
cells:NewField('ftype', L.float):Load(0)

-- Swap Failures
test.fail_function(function()
  cells:Swap('f1','noname')
end, 'Could not find a field named "noname"')
test.fail_function(function()
  cells:Swap('f1','ftype')
end, 'Cannot Swap%(%) fields of different type')

-- Copy Failures
test.fail_function(function()
  cells:Copy{from='f1',to='noname'}
end, 'Could not find a field named "noname"')
test.fail_function(function()
  cells:Copy{from='ftype',to='f1'}
end, 'Cannot Copy%(%) fields of different type')
test.fail_function(function()
  cells:Copy('f1','f2')
end, 'Copy%(%) should be called.*relation%:Copy%{from=\'f1\',to=\'f2\'%}')

-- Copy Success using auto-allocate
cells:Copy{from='f1',to='f2'}

