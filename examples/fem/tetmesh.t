import "compiler.liszt"


local Tetmesh = {}
Tetmesh.__index = Tetmesh
package.loaded["examples.fem.tetmesh"] = Tetmesh



------------------------------------------------------------------------------

--[[ edges are duplicated, one way for each direction
local function build_edges(mesh, v1s, v2s, v3s)
  local neighbors = {} -- vertex to vertex graph
  for k = 1,(mesh:nVerts()) do neighbors[k] = {} end

  -- construct an edge for each triangle
  for i = 1,(mesh:nTris()) do
    neighbors[v1s[i]+1][v2s[i]+1] = true
    neighbors[v1s[i]+1][v3s[i]+1] = true

    neighbors[v2s[i]+1][v1s[i]+1] = true
    neighbors[v2s[i]+1][v3s[i]+1] = true

    neighbors[v3s[i]+1][v1s[i]+1] = true
    neighbors[v3s[i]+1][v2s[i]+1] = true
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
  mesh.edges = L.NewRelation(n_edges, 'edges')
  mesh.edges:NewField('tail', mesh.vertices):Load(e_tail)
  mesh.edges:NewField('head', mesh.vertices):Load(e_head)

  mesh.vertices:NewField('degree', L.int):Load(degrees)

  -- index the edges
  mesh.edges:GroupBy('tail')
  mesh.vertices:NewFieldMacro('edges', L.NewMacro(function(v)
    return liszt ` L.Where(mesh.edges.tail, v)
  end))
  mesh.vertices:NewFieldMacro('neighbors', L.NewMacro(function(v)
    return liszt ` L.Where(mesh.edges.tail, v).head
  end))

  -- set up the pointers from triangles to edges
  mesh.triangles:NewField('e12', mesh.edges):Load(0)
  mesh.triangles:NewField('e21', mesh.edges):Load(0)
  mesh.triangles:NewField('e13', mesh.edges):Load(0)
  mesh.triangles:NewField('e31', mesh.edges):Load(0)
  mesh.triangles:NewField('e23', mesh.edges):Load(0)
  mesh.triangles:NewField('e32', mesh.edges):Load(0)
  local compute_tri_pointers = liszt kernel ( t : mesh.triangles )
    for e in t.v1.edges do
      if e.head == t.v2 then t.e12 = e end
      if e.head == t.v3 then t.e13 = e end
    end
    for e in t.v2.edges do
      if e.head == t.v1 then t.e21 = e end
      if e.head == t.v3 then t.e23 = e end
    end
    for e in t.v3.edges do
      if e.head == t.v1 then t.e31 = e end
      if e.head == t.v2 then t.e32 = e end
    end
  end
  compute_tri_pointers(mesh.triangles)
end]]


-- Includes all vertices directly connected to a vertex, through an element,
-- including the indexing vertex itself.
-- This includes double copies for a pair of different vertices.
-- => there will be one row for (x, y) and another for (y, x).
-- The function orders rows by 1st vertex and then the 2nd, and works for
-- only a tetrahedral mesh.
local function build_element_vertices(mesh, elements)
  local neighbors = {} -- vertex to vertex graph
  for k = 1,(mesh:nVerts()) do neighbors[k] = {} end

  -- add an entry for each tetahedron
  for i = 1,(mesh:nTets()) do
    local vertices = elements[i]
    for x = 1,4 do
      for y = 1,4 do
        neighbors[vertices[x] + 1][vertices[y] + 1] = true
      end
    end
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
  -- this is not exactly edges, as it includes rows of type (x, x)
  mesh.edges = L.NewRelation(n_edges, 'edges')
  mesh.edges:NewField('tail', mesh.vertices):Load(e_tail)
  mesh.edges:NewField('head', mesh.vertices):Load(e_head)

  mesh.vertices:NewField('degree', L.int):Load(degrees)

  -- index the edges
  mesh.edges:GroupBy('tail')
  mesh.vertices:NewFieldMacro('edges', L.NewMacro(function(v)
    return liszt ` L.Where(mesh.edges.tail, v)
  end))
  mesh.vertices:NewFieldMacro('neighbors', L.NewMacro(function(v)
    return liszt ` L.Where(mesh.edges.tail, v).head
  end))

  -- set up pointers from tetrahedra to edges
  mesh.tetrahedra:NewField('e', L.smallmatrix(mesh.edges, 4, 4))
  local compute_tet_edges = liszt kernel (t : mesh.tetrahedra)
    for i = 0,3 do
      for e in t.v[i].edges do
        for j = 0,3 do
          if e.head == t.v[j] then t.e[i, j] = e end
        end
      end
    end
  end

  compute_tet_edges(mesh.tetrahedra)
end


-- Let's define a new function as an entry in the Tetmesh table
-- This function is going to be responsible for constructing the
-- Relations representing a tetrahedral mesh.
function Tetmesh.LoadFromLists(vertices, elements)
  -- We're going to pack everything into a new table encapsulating
  -- the tetrahedral mesh.
  local mesh = {}

  -- First, we set Trimesh as the prototype of the new table
  setmetatable(mesh, Tetmesh)

  local n_tets  = #elements
  local n_verts = #vertices

  -- Define two new relations and store them in the mesh
  mesh.tetrahedra = L.NewRelation(n_tets,  'tetrahedra')
  mesh.vertices   = L.NewRelation(n_verts, 'vertices')

  -- Define the fields
  mesh.vertices:NewField('pos', L.vec3d)
  mesh.tetrahedra:NewField('v', L.vector(mesh.vertices, 4))

  -- Load the supplied data
  mesh.vertices.pos:Load(vertices)
  mesh.tetrahedra.v:Load(elements)

  -- build vertex-vertex relation for storing mass matrix (and other fields?)
  build_element_vertices(mesh, elements)

  -- and return the resulting mesh
  return mesh
end




------------------------------------------------------------------------------


function Tetmesh:nTets()
  return self.tetrahedra:Size()
end
function Tetmesh:nVerts()
  return self.vertices:Size()
end


------------------------------------------------------------------------------

