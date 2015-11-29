import "ebb"
local L = require "ebblib" -- Every Ebb File should start with this command


-- Declare a table named Trimesh
local Trimesh = {}

-- We are going to use this table as a prototype object
-- see http://en.wikipedia.org/prototypes
Trimesh.__index = Trimesh

-- Finally, we declare that the Trimesh table should be returned
-- when this file is 'required' elsewhere
package.loaded["devapps.benchmarks.trimesh"] = Trimesh


------------------------------------------------------------------------------

-- edges are duplicated, one way for each direction
local function build_edges(mesh, vs) --v1s, v2s, v3s)
  local neighbors = {} -- vertex to vertex graph
  for k = 1,(mesh:nVerts()) do neighbors[k] = {} end

  -- record the set of edges that each triangle forces to exist
  -- redundancy is eliminated since multiple triangles just set the
  -- same entry to true multiple times
  for i = 1,(mesh:nTris()) do
    neighbors[vs[i][1]+1][vs[i][2]+1] = true
    neighbors[vs[i][1]+1][vs[i][3]+1] = true

    neighbors[vs[i][2]+1][vs[i][1]+1] = true
    neighbors[vs[i][2]+1][vs[i][3]+1] = true

    neighbors[vs[i][3]+1][vs[i][1]+1] = true
    neighbors[vs[i][3]+1][vs[i][2]+1] = true
  end

  local n_edges = 0
  local degrees = {}
  local e_tail = {}
  local e_head = {}
  for i = 1,(mesh:nVerts()) do
    degrees[i] = 0
    for j,_ in pairs(neighbors[i]) do
      table.insert(e_tail, i-1)
      table.insert(e_head, j-1)
      degrees[i] = degrees[i] + 1
    end
    n_edges = n_edges + degrees[i]
  end

  -- basic data
  mesh.edges = L.NewRelation { size = n_edges, name = 'edges' }
  mesh.edges:NewField('tail', mesh.vertices):Load(e_tail)
  mesh.edges:NewField('head', mesh.vertices):Load(e_head)

  mesh.vertices:NewField('degree', L.int):Load(degrees)

  -- index the edges
  mesh.edges:GroupBy('tail')
  mesh.vertices:NewFieldMacro('edges', L.Macro(function(v)
    return ebb ` L.Where(mesh.edges.tail, v)
  end))
  mesh.vertices:NewFieldMacro('neighbors', L.Macro(function(v)
    return ebb ` L.Where(mesh.edges.tail, v).head
  end))

  -- set up the pointers from triangles to edges
  mesh.triangles:NewField('e12', mesh.edges):Load(0)
  mesh.triangles:NewField('e21', mesh.edges):Load(0)
  mesh.triangles:NewField('e13', mesh.edges):Load(0)
  mesh.triangles:NewField('e31', mesh.edges):Load(0)
  mesh.triangles:NewField('e23', mesh.edges):Load(0)
  mesh.triangles:NewField('e32', mesh.edges):Load(0)
  local ebb compute_tri_pointers ( t : mesh.triangles )
    for e in t.v[0].edges do
      if e.head == t.v[1] then t.e12 = e end
      if e.head == t.v[2] then t.e13 = e end
    end
    for e in t.v[1].edges do
      if e.head == t.v[0] then t.e21 = e end
      if e.head == t.v[2] then t.e23 = e end
    end
    for e in t.v[2].edges do
      if e.head == t.v[0] then t.e31 = e end
      if e.head == t.v[1] then t.e32 = e end
    end
  end
  mesh.triangles:foreach(compute_tri_pointers)
end

-- Let's define a new function as an entry in the Trimesh table
-- This function is going to be responsible for constructing the
-- Relations representing a triangle mesh.
function Trimesh.LoadFromLists(positions, tri_verts) --v1s, v2s, v3s)
  -- We're going to pack everything into a new table encapsulating
  -- the triangle mesh.
  local mesh = {}

  -- First, we set Trimesh as the prototype of the new table
  setmetatable(mesh, Trimesh)

  local n_tris = #tri_verts--#v1s
  local n_verts = #positions

  -- Define two new relations and store them in the mesh
  mesh.triangles = L.NewRelation { size = n_tris, name = 'triangles' }
  mesh.vertices  = L.NewRelation { size = n_verts, name = 'vertices' }

  -- Define the fields
  mesh.vertices:NewField('pos', L.vec3d)
  mesh.triangles:NewField('v', L.vector(mesh.vertices, 3))
  --mesh.triangles:NewField('v1', mesh.vertices)
  --mesh.triangles:NewField('v2', mesh.vertices)
  --mesh.triangles:NewField('v3', mesh.vertices)

  -- Load the supplied data
  mesh.vertices.pos:Load(positions)
  mesh.triangles.v:Load(tri_verts)
  --mesh.triangles.v1:Load(v1s)
  --mesh.triangles.v2:Load(v2s)
  --mesh.triangles.v3:Load(v3s)

  build_edges(mesh, tri_verts) --v1s, v2s, v3s)

  -- and return the resulting mesh
  return mesh
end

-- We've also chosen to support loading from an OFF file. This routine
-- just loads the data in and then calls the other constructor function
function Trimesh.LoadFromOFF(path)
  -- OFF files have the following format
  --
  --[[
  OFF
  #vertices #triangles 0
  x0 y0 z0
    ...
    ...   #vertices rows of coordinate triples
    ...
  3 vertex_1 vertex_2 vertex_3
    ...
    ...   #triangles rows of vertex index triples
    ...
  ]]--

  -- make sure path is converted to a string before use
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
  local tri_data_array = {}
  --local v1_data_array = {}
  --local v2_data_array = {}
  --local v3_data_array = {}
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
    --v1_data_array[i] = off_in:read('*number')
    --v2_data_array[i] = off_in:read('*number')
    --v3_data_array[i] = off_in:read('*number')
  end

  -- don't forget to close the file when done
  off_in:close()

  return Trimesh.LoadFromLists(
    position_data_array,
    tri_data_array
    --v1_data_array,
    --v2_data_array,
    --v3_data_array
  )
end


------------------------------------------------------------------------------


-- We can supply convenience functions that will work on all
-- meshes by installing those functions on the prototype.
-- See a Lua tutorial for more information about the obj:f() colon syntax
function Trimesh:nTris()
  return self.triangles:Size()
end
function Trimesh:nVerts()
  return self.vertices:Size()
end
function Trimesh:nEdges()
  return self.edges:Size()
end


------------------------------------------------------------------------------


