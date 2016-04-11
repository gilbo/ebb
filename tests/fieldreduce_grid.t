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

-- includes
local Grid  = require 'ebb.domains.grid'

-- grid
local Nx = 12
local Ny = 18
local grid = Grid.NewGrid2d {
    size   = {Nx, Ny},
    origin = {0, 0},
    width  = {4, 4},
    periodic_boundary = {true, true},
}
local C = grid.cells
local V = grid.vertices
C:SetPartitions{2,3}
V:SetPartitions{2,3}

-----------------------------------
--  Uncentered Scalar reduction: --
-----------------------------------

V:NewField("sval", L.double)
V:NewField("sval1", L.double)
V:NewField("sval2", L.double)
V:NewField("sval3", L.double)
V:NewField("sval4", L.double)

local ebb s_set_v(v : V)
  var dx = L.double(L.xid(v))
  var dy = L.double(L.yid(v))
  var d = Ny*dx + dy
  v.sval = d
  v.sval1 = 0
  v.sval2 = 0
  v.sval3 = 0
  v.sval4 = 0
end
local ebb s_reduce_uncentered (c : C)
  c.vertex(-1,-1).sval += .25*.1
  c.vertex(-1, 1).sval += .25*.1
  c.vertex( 1,-1).sval += .25*.1
  c.vertex( 1, 1).sval += .25*.1
  var dx = L.double(L.xid(c))
  var dy = L.double(L.yid(c))
  var d = Ny*dx + dy
  c.vertex(-1,-1).sval1 += 0.01*d + 0.001
  c.vertex(-1, 1).sval2 += 0.01*d + 0.001
  c.vertex( 1,-1).sval3 += 0.01*d + 0.001
  c.vertex( 1, 1).sval4 += 0.01*d + 0.001
end

V:foreach(s_set_v)
V.sval:Print()

C:foreach(s_reduce_uncentered)

V.sval:Print()
V.sval1:Print()
V.sval2:Print()
V.sval3:Print()
V.sval4:Print()

-----------------------------------
--  Uncentered Vector reduction: --
-----------------------------------

V:NewField("vval", L.vec3d)

local ebb v_set_v(v : V)
  var dx = L.double(L.xid(v))
  var dy = L.double(L.yid(v))
  var d = Ny*dx + dy
  v.vval = {1*d, 2*d, 3*d}
end
local ebb v_reduce_uncentered (c : C)
  c.vertex(-1,-1).vval += .25*{ .1, .2, .3}
  c.vertex(-1, 1).vval += .25*{ .1, .2, .3}
  c.vertex( 1,-1).vval += .25*{ .1, .2, .3}
  c.vertex( 1, 1).vval += .25*{ .1, .2, .3}
end

V:foreach(v_set_v)

C:foreach(v_reduce_uncentered)

V.vval:Print()

-----------------------------------
--  Uncentered Matrix reduction: --
-----------------------------------

V:NewField("mat", L.mat3d)

local ebb m_set_v(v : V)
  var dx = L.double(L.xid(v))
  var dy = L.double(L.yid(v))
  var d = Ny*dx + dy
  v.mat = {{d, 0.0, 0.0},
             {0.0, d, 0.0},
             {0.0, 0.0, d}}
end
local ebb m_reduce_uncentered (c : C)
  c.vertex(-1,-1).mat += .25*{
    {.11, .11, .11},
    {.22, .22, .22},
    {.33, .33, .33}
  }
  c.vertex(-1,1).mat += .25*{
    {.11, .11, .11},
    {.22, .22, .22},
    {.33, .33, .33}
  }
  c.vertex(1,-1).mat += .25*{
    {.11, .11, .11},
    {.22, .22, .22},
    {.33, .33, .33}
  }
  c.vertex(1,1).mat += .25*{
    {.11, .11, .11},
    {.22, .22, .22},
    {.33, .33, .33}
  }
end

V:foreach(m_set_v)
C:foreach(m_reduce_uncentered)

V.mat:Print()
