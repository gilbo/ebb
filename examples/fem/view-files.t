import 'compiler.liszt'
local vdb = L.require 'lib.vdb'

local Tetmesh = L.require 'examples.fem.tetmesh'
local VEGFileIO = L.require 'examples.fem.vegfileio'
local PN = L.require 'lib.pathname'

-- Input can be liszt or vega
local input = 'vega'

print("Loading mesh ...")
local turtle = VEGFileIO.LoadTetmesh
  'examples/fem/turtle-volumetric-homogeneous.veg'

local mesh = turtle

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

-- let's visualize the mesh if we can?

local sqrt3 = math.sqrt(3)

local tet_volume = liszt function(p0,p1,p2,p3)
  var d1 = p0-p3
  var d2 = p1-p3
  var d3 = p2-p3

  -- triple product
  return L.dot(L.cross(d1,d2),d3)
end
local trinorm = liszt function(p0,p1,p2)
  var d1 = p1-p0
  var d2 = p2-p0
  var n  = L.cross(d1,d2)
  var len = L.length(n)
  if len < 1e-10 then len = L.float(1e-10) end
  return n/len
end
local dot_to_color = liszt function(d)
  var val = d * 0.5 + 0.5
  var col = L.vec3f({val,val,val})
  return col
end

local lightdir = L.NewVector(L.float,{sqrt3,sqrt3,sqrt3})

local visualizeDeformation = liszt kernel ( t : mesh.tetrahedra )
  var p0 = L.vec3f(t.v[0].pos)
  var p1 = L.vec3f(t.v[1].pos)
  var p2 = L.vec3f(t.v[2].pos)
  var p3 = L.vec3f(t.v[3].pos)

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

function visualize(mesh)
  vdb.vbegin()
  vdb.frame() -- this call clears the canvas for a new frame
  visualizeDeformation(mesh.tetrahedra)
  vdb.vend()
  print('Hit enter for next frame')
  io.read()
end

function loadPositions(filename, mesh)
  local positions = {}
  local infile = io.open(tostring(filename), "r")
  local line = infile:read()
  local toprint = true
  while line ~= nil do
    local p = {}
    positions[#positions + 1] = p
    local tokens = split_on_spaces(line)
    for i = 1,3 do
      p[i] = tonumber(tokens[i]:sub(1, tokens[i]:len()-1))
    end
    line = infile:read()
  end
  infile:close()
  mesh.vertices.pos:Load(positions)
end

visualize(mesh)
for i = 0,10 do
  loadPositions("examples/fem/out/mesh_" .. input .. "_" ..tostring(i), mesh)
  visualize(mesh)
end

-- pause before exit
print('pausing before exit... (hit enter to exit)')
io.read()
