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
import "ebb"
local L = require 'ebblib'

local ioOff = {}
ioOff.__index = ioOff
package.loaded["ebb.domains.ioOff"] = ioOff

local Trimesh = require 'ebb.domains.trimesh'

------------------------------------------------------------------------------

-- OFF files have the following format
--  
--  OFF
--  #vertices #triangles 0
--  x0 y0 z0
--    ...
--    ...   #vertices rows of coordinate triples
--    ...
--  3 vertex_1 vertex_2 vertex_3
--    ...
--    ...   #triangles rows of vertex index triples
--    ...
--

------------------------------------------------------------------------------

function ioOff.LoadTrimesh(path)

  -- make sure we convert the path into a string before use
  path = tostring(path)

  -- In Lua, we can open files just like in C
  local off_in = io.open(path, "r")
  if not off_in then
    error('failed to open OFF file '..path)
  end

  -- we can read a line like so
  local OFF_SIG = off_in:read('*line')

  if OFF_SIG ~= 'OFF' then
    error('OFF file must begin with the first line "OFF"')
  end

  -- read the counts of vertices and triangles
  local n_verts = off_in:read('*number')
  local n_tris  = off_in:read('*number')
  local zero    = off_in:read('*number')

  -- now read in all the vertex coordinate data
  -- we pack each coordinate triple into a list to represent a vector value
  local position_data_array = {}
  for i = 1, n_verts do
    local vec = {
      off_in:read('*number'),
      off_in:read('*number'),
      off_in:read('*number')
    }
    position_data_array[i] = vec
  end

  -- Then read in all the vertex index arrays
  -- again, we pack these triples into lists to represent vector values
  local tri_data_array = {}
  for i = 1, n_tris do
    local three   = off_in:read('*number')
    if three ~= 3 then
      error('tried to read a triangle with '..three..' vertices')
    end
    tri_data_array[i] = {
      off_in:read('*number'),
      off_in:read('*number'),
      off_in:read('*number')
    }
  end

  -- don't forget to close the file when done
  off_in:close()

  -- Defer the construction of the mesh to the function we
  -- defined above
  return Trimesh.LoadFromLists(
    position_data_array,
    tri_data_array
  )
end






