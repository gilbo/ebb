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

local vdb = {}
package.loaded["ebb.lib.vdb"] = vdb

local L = require "ebblib"
local T = require "ebb.src.types"

local Pathname  = (require "ebb.lib.pathname").Pathname
local libdir    = Pathname.scriptdir()
local VDB_C     = terralib.includecstring([[
#include <vdb.h>]], {"-I", tostring(libdir)})


local function check_vec(vec, errmsg)
  if not vec or
     not T.istype(vec.node_type) or
     not vec.node_type:isvector() or
     not vec.node_type:isnumeric() or
     vec.node_type.N < 3
  then
    error(errmsg)
  end
end

vdb.color = L.Macro(function(vec)
  check_vec(vec, "vdb.color() expects a vector argument of length 3 or more")
  if L.GetDefaultProcessor() == L.GPU then return ebb `0
  else return ebb `VDB_C.vdb_color(vec[0], vec[1], vec[2]) end
end)

vdb.normal = L.Macro(function(p0, p1)
  local err = "vdb.normal() expects 2 vector arguments of length 3 or more"
  check_vec(p0, err)
  check_vec(p1, err)
  if L.GetDefaultProcessor() == L.GPU then return ebb `0
  else return ebb `VDB_C.vdb_normal(p0[0], p0[1], p0[2],
                                    p1[0], p1[1], p1[2]) end
end)

vdb.point = L.Macro(function(vec)
  check_vec(vec, "vdb.point() expects a vector argument of length 3 or more")
  if L.GetDefaultProcessor() == L.GPU then return ebb `0
  else return ebb `VDB_C.vdb_point(vec[0], vec[1], vec[2]) end
end)

vdb.line = L.Macro(function(p0, p1)
  local err = "vdb.line() expects 2 vector arguments of length 3 or more"
  check_vec(p0, err)
  check_vec(p1, err)
  if L.GetDefaultProcessor() == L.GPU then return ebb `0
  else return ebb `VDB_C.vdb_line(p0[0], p0[1], p0[2],
                                  p1[0], p1[1], p1[2]) end
end)

vdb.triangle = L.Macro(function(p0, p1, p2)
  local err = "vdb.triangle() expects 3 vector arguments of length 3 or more"
  check_vec(p0, err)
  check_vec(p1, err)
  check_vec(p2, err)
  if L.GetDefaultProcessor() == L.GPU then return ebb `0
  else return ebb `VDB_C.vdb_triangle(p0[0], p0[1], p0[2],
                                      p1[0], p1[1], p1[2],
                                      p2[0], p2[1], p2[2]) end
end)

vdb.vbegin  = VDB_C.vdb_begin
vdb.vend    = VDB_C.vdb_end
vdb.flush  = VDB_C.vdb_flush
vdb.frame  = VDB_C.vdb_frame
--vdb.sample = VDB_C.vdb_sample



