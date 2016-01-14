import "ebb"
local L = require "ebblib"

function split_on_spaces(str)
  local a = {}
  while str:len() > 0 do
    -- trim any whitespace at the front
    str = str:sub(str:match("^%s*"):len()+1) -- 1 indexing
    -- read off the next token
    local token = str:match("[^%s]+")
    if token == nil then break end  -- break if there isn't a token
    -- save the token
    table.insert(a,token)
    -- advance past token
    str = str:sub(token:len()+1) -- 1 indexing
  end
  return a
end


function dump_list(t)
  print("size is "..tostring(#t))
  for i=1,#t do
    print(t[i])
  end
end

--dump_list(split_on_spaces("asdf"))

local Tetmesh = require 'INTERNAL_devapps.fem.tetmesh'
local VEGFileIO = require 'INTERNAL_devapps.fem.vegfileio'

local turtle = VEGFileIO.LoadTetmesh
  'INTERNAL_devapps/fem/turtle-volumetric-homogeneous.veg'

























-- let's visualize the mesh if we can?

local sqrt3 = math.sqrt(3)

local ebb tet_volume(p0,p1,p2,p3)
  var d1 = p1-p0
  var d2 = p2-p0
  var d3 = p3-p0

  -- triple product
  return L.dot(L.cross(d1,d2),d3)
end
local ebb trinorm(p0,p1,p2)
  var d1 = p1-p0
  var d2 = p2-p0
  var n  = L.cross(d1,d2)
  var len = L.length(n)
  if len < 1e-10 then len = L.float(1e-10) end
  return n/len
end
local ebb dot_to_color(d)
  var val = d * 0.5 + 0.5
  var col = L.vec3f({val,val,val})
  return col
end

local lightdir = L.Constant(L.vec3f,{sqrt3,sqrt3,sqrt3})

-- EXTRA: (optional.  It demonstrates the use of VDB, a visual debugger)
local vdb = require('ebb.lib.vdb')
local visualize = ebb ( t : turtle.tetrahedra )
  var p0 = L.vec3f(t.v0.pos)
  var p1 = L.vec3f(t.v1.pos)
  var p2 = L.vec3f(t.v2.pos)
  var p3 = L.vec3f(t.v3.pos)

  var flipped : L.double = 1.0
  if tet_volume(p0,p1,p2,p3) < 0 then flipped = -1.0 end

  var d0 = flipped * L.dot(lightdir, trinorm(p1,p2,p3))
  var d1 = flipped * L.dot(lightdir, trinorm(p0,p3,p2))
  var d2 = flipped * L.dot(lightdir, trinorm(p0,p1,p3))
  var d3 = flipped * L.dot(lightdir, trinorm(p1,p0,p2))

  vdb.color(dot_to_color(d0))
  vdb.triangle(p1, p2, p3)
  vdb.color(dot_to_color(d1))
  vdb.triangle(p0, p3, p2)
  vdb.color(dot_to_color(d2))
  vdb.triangle(p0, p1, p3)
  vdb.color(dot_to_color(d3))
  vdb.triangle(p1, p0, p2)
end

vdb.vbegin()
  vdb.frame() -- this call clears the canvas for a new frame
  turtle.tetrahedra:foreach(visualize)
vdb.vend()

-- pause before exit
print('pausing before exit... (hit enter to exit)')
io.read()




