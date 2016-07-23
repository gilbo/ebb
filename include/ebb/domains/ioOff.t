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

------------------------------------------------------------------------------

local C = terralib.includecstring [[
#include <stdio.h>

FILE* _io_off_get_stderr() { return stderr; }
]]
local DLD = require 'ebb.lib.dld'

-- some Terra macro programming
local fp = symbol(&C.FILE, 'fp')
-- error check
local ec = macro(function(test, msg, ...)
  local varargs = {...}
  if msg then
    msg = msg:asvalue()..'\n'
    msg = quote C.fprintf(C._io_off_get_stderr(), msg, varargs) end
  else
    msg = {}
  end
  return quote
    if not test then
      C.fclose(fp)
      [msg]
      return 1
    end
  end
end)

local terra DumpPositions( dld : &DLD.C_DLD, filename : rawstring ) : int
  var [fp] = C.fopen(filename, 'a')
  ec( fp ~= nil, "Cannot open CSV file for writing" )

  ec( dld.version[0] == 1 and dld.version[1] == 0, "bad DLD version" )
  ec( dld.base_type == DLD.DOUBLE, "expected DLD for doubles" )
  ec( dld.type_stride == 24, "bad DLD type_stride %d", dld.type_stride )
  ec( dld.type_dims[0] == 3 and dld.type_dims[1] == 1, "bad DLD type_dims" )
  ec( dld.location == DLD.CPU, "trying to dump DLD on GPU" )
  ec( dld.dim_size[1] == 1 and dld.dim_size[2] == 1, "bad DLD dim_size" )
  ec( dld.dim_stride[0] == 1, "bad DLD dim_stride" )

  var n_verts = dld.dim_size[0]
  var pos     = [&double](dld.address)

  for i = 0, n_verts do
    ec( C.fprintf(fp, "%f %f %f\n", pos[3*i+0], pos[3*i+1], pos[3*i+2]) > 0,
        "error writing entry\n" )
  end

  C.fclose(fp)
  return 0
end

local terra DumpTriangles( dld : &DLD.C_DLD, filename : rawstring ) : int
  var [fp] = C.fopen(filename, 'a')
  ec( fp ~= nil, "Cannot open CSV file for writing\n" )

  ec( dld.version[0] == 1 and dld.version[1] == 0, "bad DLD version" )
  var key_bytes = 1
      if dld.base_type == DLD.KEY_8  then key_bytes = 1
                                          ec( dld.type_stride == 3 )
  elseif dld.base_type == DLD.KEY_16 then key_bytes = 2
                                          ec( dld.type_stride == 6 )
  elseif dld.base_type == DLD.KEY_32 then key_bytes = 4
                                          ec( dld.type_stride == 12 )
  elseif dld.base_type == DLD.KEY_64 then key_bytes = 8
                                          ec( dld.type_stride == 24 )
  else ec( false, "unexpected key type in DLD" ) end

  ec( dld.type_dims[0] == 3 and dld.type_dims[1] == 1, "bad DLD type_dims" )
  ec( dld.location == DLD.CPU, "trying to dump DLD on GPU" )
  ec( dld.dim_size[1] == 1 and dld.dim_size[2] == 1, "bad DLD dim_size" )
  ec( dld.dim_stride[0] == 1, "bad DLD dim_stride" )

  var n_tris  = dld.dim_size[0]

  escape
    local function genloop(typ) return quote
      var verts   = [&typ](dld.address)
      for i = 0, n_tris do
        ec( C.fprintf(fp, "3 %d %d %d\n",
                          verts[3*i+0], verts[3*i+1], verts[3*i+2]) > 0,
            "error writing entry\n" )
      end
    end end
    emit quote
          if key_bytes == 1 then [genloop(uint8)]
      elseif key_bytes == 2 then [genloop(uint16)]
      elseif key_bytes == 4 then [genloop(uint32)]
                            else [genloop(uint64)] end
    end
  end

  C.fclose(fp)
  return 0
end

function ioOff.DumpTrimesh(mesh, path)
  -- make sure we convert the path into a string before use
  path = tostring(path)

  local off_out = io.open(path, "w")
  if not off_out then
    error('failed to open/create output OFF file '..path)
  end

  -- write out magic word
  off_out:write("OFF\n")

  -- write out number of elements of each type
  off_out:write(tostring(mesh.vertices:Size()).." "..
                tostring(mesh.triangles:Size()).." "..
                "0\n")

  off_out:close()

  if mesh.vertices.pos:Dump(DumpPositions, path) ~= 0 then
    error('Error while dumping positions '..path, 2)
  end
  if mesh.triangles.v:Dump(DumpTriangles, path) ~= 0 then
    error('Error while dumping triangles '..path, 2)
  end
end


