import "compiler.liszt" -- Every Liszt File should start with this command

-- Declare a table named Trimesh for the Module
local Trimesh = {}

-- We are going to use this table as a prototype object
-- see http://en.wikipedia.org/prototypes
Trimesh.__index = Trimesh

-- Finally, we declare that the Trimesh table should be returned
-- when this file is 'required' elsewhere
package.loaded["domains.trimesh"] = Trimesh


------------------------------------------------------------------------------

-- Let's define a way to create a Trimesh from a list of vertex
-- positions and a list of triangles identified by triples of vertices
-- (We'll assume the vertex indices are 0-indexed, rather than 1-indexed)
function Trimesh.LoadFromLists(positions, tri_verts)
  -- First, we create a new "object" (Lua calls them tables)
  -- to represent the triangle mesh we're creating.
  local mesh = {}

  -- To ensure we get all of the functions we've defined on the Trimesh
  -- prototype object, we set Trimesh as the prototype of the new table
  setmetatable(mesh, Trimesh)

  -- how many triangles and vertices are there?
  local n_tris = #tri_verts
  local n_verts = #positions

  -- Define two new relations and store them in the mesh
  mesh.triangles = L.NewRelation { size = n_tris, name = 'triangles' }
  mesh.vertices  = L.NewRelation { size = n_verts, name = 'vertices' }

  -- We define two standard fields on a Trimesh,
  --        vertex positions (pos)
  --    and the topology linking triangles to vertices (v)
  -- Note that v is defined as a vector of 3 keys
  mesh.vertices:NewField('pos', L.vec3d)
  mesh.triangles:NewField('v', L.vector(mesh.vertices, 3))

  -- Since our data was supplied in lists, we can just load it direclty
  mesh.vertices.pos:Load(positions)
  mesh.triangles.v:Load(tri_verts)

  -- Finally, we'd like to augment this basic structure with another
  -- relation representing edges of the mesh.  We defer this
  -- computation into a method of the prototype (defined below)
  mesh:build_edges(tri_verts)

  -- and return the resulting mesh
  return mesh
end


-- Here we build a relation of directed edges
-- and set it up to allow easy use from client code
function Trimesh:build_edges(vs)
  if self.edges then return end -- exit if we've already built edges
  local mesh = self

  -- We're going to build a representation of the edge graph
  local neighbors = {} -- vertex to vertex graph
  for k = 1, mesh:nVerts() do neighbors[k] = {} end

  -- We record each edge of each triangle with each of the two
  -- possible orientations.  While some edges will be marked as true
  -- multiple times, this will just result in a single graph entry.
  for i = 1, mesh:nTris() do
    neighbors[vs[i][1]+1][vs[i][2]+1] = true
    neighbors[vs[i][1]+1][vs[i][3]+1] = true

    neighbors[vs[i][2]+1][vs[i][1]+1] = true
    neighbors[vs[i][2]+1][vs[i][3]+1] = true

    neighbors[vs[i][3]+1][vs[i][1]+1] = true
    neighbors[vs[i][3]+1][vs[i][2]+1] = true
  end

  -- Now that we've built the graph, we're going to 
  -- serialize it into a compressed row-storage like form
  -- To do so, we build two arrays identifying the head and tail
  -- vertex indices for each directed edge, sorted primarily on tail
  local n_edges = 0
  local e_tail = {}
  local e_head = {}
  for i = 1, mesh:nVerts() do
    for j,_ in pairs(neighbors[i]) do
      table.insert(e_tail, i-1)
      table.insert(e_head, j-1)
      n_edges = n_edges + 1
    end
  end

  -- Now we can create the edge relation
  mesh.edges = L.NewRelation { size = n_edges, name = 'edges' }
  mesh.edges:NewField('tail', mesh.vertices):Load(e_tail)
  mesh.edges:NewField('head', mesh.vertices):Load(e_head)

  -- And group the edges by their tail
  mesh.edges:GroupBy('tail')

  -- Since we grouped the edges by their tail, we can now
  -- issue Where queries from the vertices relation into
  -- the edges relation looking for all edges with a given tail
  -- In order to abstract this query for the user, we install a macro
  -- on the vertices relation
  mesh.vertices:NewFieldMacro('edges', L.NewMacro(function(v)
    return liszt ` L.Where(mesh.edges.tail, v)
  end))
  -- We also store a slight modification where neighboring vertices
  -- are given directly instead of the edges themselves
  mesh.vertices:NewFieldMacro('neighbors', L.NewMacro(function(v)
    return liszt ` L.Where(mesh.edges.tail, v).head
  end))

  -- Finally, to make it easier to update quantities on the edges
  -- from the triangles, we set up 6 more fields for convenience
  mesh.triangles:NewField('e12', mesh.edges):Load(0)
  mesh.triangles:NewField('e21', mesh.edges):Load(0)
  mesh.triangles:NewField('e13', mesh.edges):Load(0)
  mesh.triangles:NewField('e31', mesh.edges):Load(0)
  mesh.triangles:NewField('e23', mesh.edges):Load(0)
  mesh.triangles:NewField('e32', mesh.edges):Load(0)
  -- and then we compute these links by mapping a liszt function
  -- over the triangles.  We have to do a somewhat silly
  -- search to find the correct edge keys to write in here.
  local liszt compute_tri_pointers ( t : mesh.triangles )
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
  mesh.triangles:map(compute_tri_pointers)
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


