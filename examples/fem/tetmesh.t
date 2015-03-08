import "compiler.liszt"


local Tetmesh = {}
Tetmesh.__index = Tetmesh
package.loaded["examples.fem.tetmesh"] = Tetmesh



------------------------------------------------------------------------------


-- Includes all vertices directly connected to a vertex, through an element,
-- including the indexing vertex itself.
-- This includes double copies for a pair of different vertices.
-- => there will be one row for (x, y) and another for (y, x).
-- The function orders rows by 1st vertex and then the 2nd, and works for
-- only a tetrahedral mesh.
local function build_element_edges(mesh, elements)
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
  mesh.edges = L.NewRelation { size = n_edges, name = 'edges' }
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
  mesh.tetrahedra:NewField('e', L.matrix(mesh.edges, 4, 4))
  local liszt compute_tet_edges (t : mesh.tetrahedra)
    for i = 0,4 do
      for e in t.v[i].edges do
        for j = 0,4 do
          if e.head == t.v[j] then t.e[i, j] = e end
        end
      end
    end
  end
  mesh.tetrahedra:map(compute_tet_edges)

  -- set up pointers from vertices to self edges
  mesh.vertices:NewField('diag', mesh.edges)
  local liszt compute_self_edges (v : mesh.vertices)
    for e in v.edges do
      if e.head == v then
        v.diag = e
      end
    end
  end
  mesh.vertices:map(compute_self_edges)

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
  mesh.tetrahedra = L.NewRelation { size = n_tets, name = 'tetrahedra' }
  mesh.vertices   = L.NewRelation { size = n_verts, name = 'vertices' }

  -- Define the fields
  mesh.vertices:NewField('pos', L.vec3d)
  mesh.tetrahedra:NewField('v', L.vector(mesh.vertices, 4))

  -- Load the supplied data
  mesh.vertices.pos:Load(vertices)
  mesh.tetrahedra.v:Load(elements)

  -- build vertex-vertex relation for storing mass matrix (and other fields?)
  build_element_edges(mesh, elements)

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

