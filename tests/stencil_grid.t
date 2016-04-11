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

local Grid = require 'ebb.domains.grid'

local nX,nY,nZ = 40,60,80

local g2d  = Grid.NewGrid2d {
  size    = {nX,nY},
  origin  = {0,0},
  width   = {nX,nY},
}

local g3d  = Grid.NewGrid3d {
  size    = {nX,nY,nZ},
  origin  = {0,0,0},
  width   = {nX,nY,nZ},
}

g2d.cells:NewField('a', L.double):Load(0)
g2d.cells:NewField('b', L.double):Load(0)
g3d.cells:NewField('a', L.double):Load(0)
g3d.cells:NewField('b', L.double):Load(0)


local ebb shiftX2d( c : g2d.cells )
  c.a = c(-1,0).b
end
do
  local fa_s = shiftX2d:_TESTING_GetFieldAccesses(g2d.cells)
  local astencil = fa_s[g2d.cells.a]:getstencil()
  local bstencil = fa_s[g2d.cells.b]:getstencil()
  local arect = astencil:envelopeRect():getranges()
  local brect = bstencil:envelopeRect():getranges()
  test.rec_aeq(arect, {{0,0},{0,0}})
  test.rec_aeq(brect, {{-1,-1},{0,0}})
end

local ebb shiftXY3d( c : g3d.cells )
  c.a = c(-1,-2,0).b
end
do
  local fa_s = shiftXY3d:_TESTING_GetFieldAccesses(g3d.cells)
  local astencil = fa_s[g3d.cells.a]:getstencil()
  local bstencil = fa_s[g3d.cells.b]:getstencil()
  local arect = astencil:envelopeRect():getranges()
  local brect = bstencil:envelopeRect():getranges()
  test.rec_aeq(arect, {{0,0},{0,0},{0,0}})
  test.rec_aeq(brect, {{-1,-1},{-2,-2},{0,0}})
end

local ebb fourPoint( c : g2d.cells )
  c.a = -c.b + 0.25 * ( c(-1,0).b + c(1,0).b + c(0,-1).b + c(0,1).b )
end
do
  local fa_s = fourPoint:_TESTING_GetFieldAccesses(g2d.cells)
  local astencil = fa_s[g2d.cells.a]:getstencil()
  local bstencil = fa_s[g2d.cells.b]:getstencil()
  local arect = astencil:envelopeRect():getranges()
  local brect = bstencil:envelopeRect():getranges()
  test.rec_aeq(arect, {{0,0},{0,0}})
  test.rec_aeq(brect, {{-1,1},{-1,1}})
end

g2d.cells:NewField('field3',L.float):Load(6)
local ebb read1( c : g2d.cells )
  var tmp = c.field3 + 5
  L.assert(tmp == 11)
end
g2d.cells:foreach(read1)
do
  local fa_s = read1:_TESTING_GetFieldAccesses(g2d.cells)
  test.neq( next(fa_s), nil )
end












