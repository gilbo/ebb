import "compiler.liszt" -- Every Liszt File should start with this command


-- Declare a table named Trimesh
local Trimesh = {}

-- We are going to use this table as a prototype object
-- see http://en.wikipedia.org/prototypesesd;fnasd;flknawpeofina;lsdknf
Trimesh.__index = Trimesh

-- Finally, we declare that the Trimesh table should be returned
-- when this file is 'required' elsewhere
package.loaded["examples.tutorials.trimesh"] = Trimesh


------------------------------------------------------------------------------


-- Let's define a new function as an entry in the Trimesh table
-- This function is going to be responsible for constructing the
-- Relations representing a triangle mesh.
function Trimesh.LoadFromLists(positions, v1s, v2s, v3s)
  -- We're going to pack everything into a new table encapsulating
  -- the triangle mesh.
  local mesh = {}

  -- First, we set Trimesh as the prototype of the new table
  setmetatable(mesh, Trimesh)

  local n_tris = #v1s
  local n_verts = #positions

  -- Define two new relations and store them in the mesh
  mesh.triangles = L.NewRelation { size = n_tris, name = 'triangles' }
  mesh.vertices  = L.NewRelation { size = n_verts, name = 'vertices' }

  -- Define the fields
  mesh.vertices:NewField('pos', L.vec3d)
  mesh.triangles:NewField('v1', mesh.vertices)
  mesh.triangles:NewField('v2', mesh.vertices)
  mesh.triangles:NewField('v3', mesh.vertices)

  -- Load the supplied data
  mesh.vertices.pos:Load(positions)
  mesh.triangles.v1:Load(v1s)
  mesh.triangles.v2:Load(v2s)
  mesh.triangles.v3:Load(v3s)

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
  local v1_data_array = {}
  local v2_data_array = {}
  local v3_data_array = {}
  for i = 1, n_tris do
    local three   = off_in:read('*number')
    if three ~= 3 then
      error('tried to read a triangle with '..three..' vertices')
    end
    v1_data_array[i] = off_in:read('*number')
    v2_data_array[i] = off_in:read('*number')
    v3_data_array[i] = off_in:read('*number')
  end

  -- don't forget to close the file when done
  off_in:close()

  return Trimesh.LoadFromLists(
    position_data_array,
    v1_data_array,
    v2_data_array,
    v3_data_array
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


------------------------------------------------------------------------------


-- We can choose to abstract out certain computations that aren't
-- specific to a given simulation

function Trimesh:ComputeVertexDegree()
  self.vertices:NewField('degree', L.int)
  self.vertices.degree:Load(0)

  local degree_kernel = liszt kernel( tri : self.triangles )
    tri.v1.degree += 1
    tri.v2.degree += 1
    tri.v3.degree += 1
  end

  degree_kernel(self.triangles)
end




