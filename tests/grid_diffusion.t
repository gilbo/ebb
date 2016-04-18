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

local N = 1024
local cells = L.NewRelation { name="cells",  dims={N,N},
                                             periodic={true,true} }
cells:SetPartitions{2,2}

cells:NewField('f', L.double):Load(0)
cells:NewField('f_new', L.double):Load(0)

local ebb init_checker( c : cells )
  var color = (L.xid(c) + L.yid(c)) % 2
  if color == 0 then c.f = 2.0
                else c.f = 1.0 end
end

cells:NewFieldMacro('__apply_macro', L.Macro(function(c,x,y)
  return ebb `L.Affine(cells, {{1,0,x},
                               {0,1,y}}, c)
end))

local ebb diffuse( c : cells )
  var avg = 0.25 * ( c(1,0).f + c(-1,0).f + c(0,1).f + c(0,-1).f )
  c.f_new = c.f + 0.25 * (avg - c.f)
end

local ebb check_vals( c : cells )
  var color = (L.xid(c) + L.yid(c)) % 2
  var goal  = 1.75
  if color == 1 then goal = 1.25 end
  L.assert(c.f_new == goal)
end

cells:foreach(init_checker)
cells:foreach(diffuse)
cells:foreach(check_vals)



