import 'ebb.liszt'

local vdb = {}
package.loaded["ebb.lib.vdb"] = vdb

local L = require "ebb.src.lisztlib"
local T = require "ebb.src.types"

local Pathname  = (require "ebb.lib.pathname").Pathname
local libdir    = Pathname.scriptdir()
local VDB_C     = terralib.includecstring([[
#include <vdb.h>]], {"-I", tostring(libdir)})


local function check_vec(vec, errmsg)
  if not vec or
     not T.isLisztType(vec.node_type) or
     not vec.node_type:isVector() or
     not vec.node_type:isNumeric() or
     vec.node_type.N < 3
  then
    error(errmsg)
  end
end

vdb.color = L.NewMacro(function(vec)
  check_vec(vec, "vdb.color() expects a vector argument of length 3 or more")
  return liszt `VDB_C.vdb_color(vec[0], vec[1], vec[2])
end)

vdb.normal = L.NewMacro(function(p0, p1)
  local err = "vdb.normal() expects 2 vector arguments of length 3 or more"
  check_vec(p0, err)
  check_vec(p1, err)
  return liszt `VDB_C.vdb_normal(p0[0], p0[1], p0[2],
                                 p1[0], p1[1], p1[2])
end)

vdb.point = L.NewMacro(function(vec)
  check_vec(vec, "vdb.point() expects a vector argument of length 3 or more")
  return liszt `VDB_C.vdb_point(vec[0], vec[1], vec[2])
end)

vdb.line = L.NewMacro(function(p0, p1)
  local err = "vdb.line() expects 2 vector arguments of length 3 or more"
  check_vec(p0, err)
  check_vec(p1, err)
  return liszt `VDB_C.vdb_line(p0[0], p0[1], p0[2],
                               p1[0], p1[1], p1[2])
end)

vdb.triangle = L.NewMacro(function(p0, p1, p2)
  local err = "vdb.triangle() expects 3 vector arguments of length 3 or more"
  check_vec(p0, err)
  check_vec(p1, err)
  check_vec(p2, err)
  return liszt `VDB_C.vdb_triangle(p0[0], p0[1], p0[2],
                                   p1[0], p1[1], p1[2],
                                   p2[0], p2[1], p2[2])
end)

vdb.vbegin  = VDB_C.vdb_begin
vdb.vend    = VDB_C.vdb_end
vdb.flush  = VDB_C.vdb_flush
vdb.frame  = VDB_C.vdb_frame
--vdb.sample = VDB_C.vdb_sample



