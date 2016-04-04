--DISABLE-PARTITIONED
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

