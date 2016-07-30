--DISABLE-DISTRIBUTED
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

local ioOff = require 'ebb.domains.ioOff'
local M = ioOff.LoadTrimesh('tests/octa.off')

local V = M.vertices
local P = V.pos

local loc_data = P:Dump({})

function shift(x,y,z)
  local ebb shift_func (v : M.vertices)
      v.pos += {x,y,z}
  end
  M.vertices:foreach(shift_func)

  local Pdata = P:Dump({})
  for i = 1, V:Size() do
    local v = Pdata[i]
    local d = loc_data[i]

    d[1] = d[1] + x
    d[2] = d[2] + y
    d[3] = d[3] + z

    --print("Pos " .. tostring(i)  .. ': (' .. tostring(v[1]) .. ',' ..
    --                tostring(v[2]) .. ',' .. tostring(v[3]) .. ')')
    --print("Loc " .. tostring(i)  .. ': (' .. tostring(d[1]) .. ',' ..
    --                tostring(d[2]) .. ',' .. tostring(d[3]) .. ')')
    assert(v[1] == d[1])
    assert(v[2] == d[2])
    assert(v[3] == d[3])
  end
end

shift(0,0,0)
shift(5,5,5)
shift(-1,6,3)

---------------------------------
--  Centered Matrix reduction: --
---------------------------------
local T = M.triangles

T:NewField("mat", L.mat3d)

local ebb m_set(t : T)
  var d = L.double(L.id(t))
  t.mat = {{d, 0.0, 0.0},
             {0.0, d, 0.0},
             {0.0, 0.0, d}}
end

local ebb m_reduce_centered (t : T)
  t.mat += {
    {.11, .11, .11},
    {.22, .22, .22},
    {.33, .33, .33}
  }
end

T:foreach(m_set)
T:foreach(m_reduce_centered)

T.mat:Print()

-----------------------------------
--  Uncentered Matrix reduction: --
-----------------------------------
-- This will produce the invocation
-- of a second reduction kernel on the GPU runtime
local E = M.edges

V:NewField("mat", L.mat3d)

local ebb m_set_v(v : V)
  var d = L.double(L.id(v))
  v.mat = {{d, 0.0, 0.0},
             {0.0, d, 0.0},
             {0.0, 0.0, d}}
end
local ebb m_reduce_uncentered (e : E)
  e.head.mat += .5*{
    {.11, .11, .11},
    {.22, .22, .22},
    {.33, .33, .33}
  }
  e.tail.mat += .5*{
    {.11, .11, .11},
    {.22, .22, .22},
    {.33, .33, .33}
  }
end

V:foreach(m_set_v)
E:foreach(m_reduce_uncentered)

V.mat:Print()
