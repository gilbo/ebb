import "compiler.liszt"


local Tetmesh = {}
Tetmesh.__index = Tetmesh
package.loaded["domains.tetmesh"] = Tetmesh


------------------------------------------------------------------------------


function Tetmesh.LoadFromLists(vertices, elements)
  local mesh = setmetatable({}, Tetmesh)

  local n_tets  = #elements
  local n_verts = #vertices

  mesh.tetrahedra = L.NewRelation { size = n_tets, name = 'tetrahedra' }
  mesh.vertices   = L.NewRelation { size = n_verts, name = 'vertices' }

  -- Define the basic fields
  mesh.vertices:NewField('pos', L.vec3d)
  mesh.tetrahedra:NewField('v', L.vector(mesh.vertices, 4))

  -- Load the supplied data
  mesh.vertices.pos:Load(vertices)
  mesh.tetrahedra.v:Load(elements)

  -- build edges
  mesh:build_edges(elements)
  mesh:build_triangles(elements)

  -- and return the resulting mesh
  return mesh
end


function Tetmesh:build_edges(elements)
  if self.edges then return end -- exit if we've already built edges
  local mesh = self

  -- build vertex-to-vertex graph
  local neighbors = {} -- vertex to vertex graph
  for k = 1,(mesh:nVerts()) do neighbors[k] = {} end

  for e = 1,(mesh:nTets()) do
    local vs = elements[e]
    for i = 1,4 do
      for j = 1,4 do
        if i ~= j then
          neighbors[vs[i] + 1][vs[j] + 1] = true
        end
      end
    end
  end

  -- serialize graph into a dense packing
  local n_edges = 0
  local e_tail = {}
  local e_head = {}
  for i = 1,mesh:nVerts() do
    for j,_ in pairs(neighbors[i]) do
      n_edges = n_edges + 1
      e_tail[n_edges] = i-1
      e_head[n_edges] = j-1
    end
  end

  -- use that to build the edges
  mesh.edges = L.NewRelation { size = n_edges, name = 'edges' }
  mesh.edges:NewField('tail', mesh.vertices):Load(e_tail)
  mesh.edges:NewField('head', mesh.vertices):Load(e_head)

  -- and group the edges for access from vertices
  mesh.edges:GroupBy('tail')
  mesh.vertices:NewFieldMacro('edges', L.NewMacro(function(v)
    return liszt ` L.Where(mesh.edges.tail, v)
  end))
  mesh.vertices:NewFieldMacro('neighbors', L.NewMacro(function(v)
    return liszt ` L.Where(mesh.edges.tail, v).head
  end))


--  -- set up pointers from tetrahedra to edges
--  mesh.tetrahedra:NewField('e', L.matrix(mesh.edges, 4, 4))
--  local liszt compute_tet_edges (t : mesh.tetrahedra)
--    for i = 0,4 do
--      for e in t.v[i].edges do
--        for j = 0,4 do
--          if e.head == t.v[j] then t.e[i, j] = e end
--        end
--      end
--    end
--  end
--  mesh.tetrahedra:map(compute_tet_edges)

--  -- set up pointers from vertices to self edges
--  mesh.vertices:NewField('diag', mesh.edges)
--  local liszt compute_self_edges (v : mesh.vertices)
--    for e in v.edges do
--      if e.head == v then
--        v.diag = e
--      end
--    end
--  end
--  mesh.vertices:map(compute_self_edges)

end

function Tetmesh:build_triangles(elements)
  if self.triangles then return end
  local mesh = self

  -- build face cache
  local face_cache = {}

  for e = 1,(mesh:nTets()) do
    local vs = elements[e]

    for i_not = 1,4 do
      local i = vs[ (i_not+0)%4 + 1 ]
      local j = vs[ (i_not+1)%4 + 1 ]
      local k = vs[ (i_not+2)%4 + 1 ]
      local triple = {i,j,k}
      -- need to flip some triples depending on parity
      -- i_not == 1 : flip
      -- i_not == 2 : no flip
      -- ...
      if i_not % 2 == 1 then triple = {j,i,k} end

      local sorted = {i,j,k}
      table.sort(sorted)
      local signature = tostring(sorted[1])..','..
                        tostring(sorted[2])..','..
                        tostring(sorted[3])

      face_cache[signature] = triple
    end
  end

  -- serialize faces into a dense packing
  local tri_vs = {}
  local n_tris = 0
  for _,triple in pairs(face_cache) do
    n_tris = n_tris + 1
    tri_vs[n_tris] = triple
  end

  mesh.triangles = L.NewRelation { size = n_tris, name = 'triangles' }
  mesh.triangles:NewField('v', L.vector(mesh.vertices, 3)):Load(tri_vs)
end

------------------------------------------------------------------------------


function Tetmesh:nTets()
  return self.tetrahedra:Size()
end
function Tetmesh:nEdges()
  return self.edges:Size()
end
function Tetmesh:nVerts()
  return self.vertices:Size()
end


------------------------------------------------------------------------------

