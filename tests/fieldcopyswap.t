import "compiler.liszt"
require "tests/test"

local cells = L.NewRelation(10, 'cells')

cells:NewField('f1', L.double):Load(0)
cells:NewField('f2', L.double):Load(0)

local setup = liszt kernel ( c : cells )
  c.f1 = 5
end
setup(cells)

cells:Swap('f1','f2')

local check1 = liszt kernel ( c : cells )
  L.assert(c.f2 == 5)
  L.assert(c.f1 == 0)
end
check1(cells)

cells:Copy{to='f1',from='f2'}

local check2 = liszt kernel ( c : cells )
  L.assert(c.f2 == 5)
  L.assert(c.f1 == 5)
end
check2(cells)

-- Check for errors
cells:NewField('uninitialized', L.double)
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
  cells:Copy{from='uninitialized',to='f1'}
end, 'Cannot Copy%(%) from field with no data')
test.fail_function(function()
  cells:Copy{from='f1',to='noname'}
end, 'Could not find a field named "noname"')
test.fail_function(function()
  cells:Copy{from='ftype',to='f1'}
end, 'Cannot Copy%(%) fields of different type')
test.fail_function(function()
  cells:Copy('f1','f2')
end, 'Copy%(%) should be called.*relation%:Copy%{from=\'f1\',to=\'f2\'%}')

-- Swap Success with null ptr
cells:Swap('f2','uninitialized')
-- Copy Success using auto-allocate
cells:Copy{from='f1',to='f2'}

