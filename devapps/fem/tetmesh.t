import "ebb"
local PN = require 'ebb.lib.pathname'

local Tetmesh = {}
Tetmesh.__index = Tetmesh
package.loaded["devapps.fem.tetmesh"] = Tetmesh


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
    return ebb ` L.Where(mesh.edges.tail, v)
  end))
  mesh.vertices:NewFieldMacro('neighbors', L.NewMacro(function(v)
    return ebb ` L.Where(mesh.edges.tail, v).head
  end))

  -- set up pointers from tetrahedra to edges
  mesh.tetrahedra:NewField('e', L.matrix(mesh.edges, 4, 4))
  local ebb compute_tet_edges (t : mesh.tetrahedra)
    for i = 0,4 do
      for e in t.v[i].edges do
        for j = 0,4 do
          if e.head == t.v[j] then t.e[i, j] = e end
        end
      end
    end
  end
  mesh.tetrahedra:foreach(compute_tet_edges)

  -- set up pointers from vertices to self edges
  mesh.vertices:NewField('diag', mesh.edges)
  local ebb compute_self_edges (v : mesh.vertices)
    for e in v.edges do
      if e.head == v then
        v.diag = e
      end
    end
  end
  mesh.vertices:foreach(compute_self_edges)

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

  -- Correct orientation for tetrahedra
  for i = 1, n_tets do
    local verts = elements[i]
    local x = {}
    for j = 1,4 do
      x[j] = vertices[verts[j] + 1]
    end
    local e = {}
    for j = 1,3 do
      e[j] = {}
      for c = 1,3 do
        e[j][c] = x[j][c] - x[4][c]
      end
    end
    local det  = ( e[1][1] * ( e[2][2] * e[3][3] - e[2][3] * e[3][2] ) +
                   e[1][2] * ( e[2][3] * e[3][1] - e[3][3] * e[2][1] ) +
                   e[1][3] * ( e[2][1] * e[3][2] - e[2][2] * e[3][1] )
                 )
    if det < 0 then
      local temp = verts[1]
      verts[1] = verts[2]
      verts[2] = temp
    end
  end

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

  -- get determintant for tetreahedron
  mesh.tetrahedra:NewFieldMacro('elementDet', L.NewMacro(function(t)
      return ebb quote
          var a = t.v[0].pos
          var b = t.v[1].pos
          var c = t.v[2].pos
          var d = t.v[3].pos
        in
          L.dot(a - d, L.cross(b - d, c - d))
        end
      end) )

  -- element density
  mesh.tetrahedra:NewFieldMacro('density', L.NewMacro(function(t)
    return ebb `mesh.density end ))

  -- lame constants
  mesh.tetrahedra:NewFieldMacro('lambdaLame', L.NewMacro(function(t)
    return ebb `L.double(mesh.lambdaLame)
  end ))
  mesh.tetrahedra:NewFieldMacro('muLame', L.NewMacro(function(t)
    return ebb `L.double(mesh.muLame)
  end ))

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


-- Print out fields over edges (things like stiffnexx matrix or mass).

function fieldToString(x)
  if type(x) == 'table' then
    local str = "{ "
    for k,v in ipairs(x) do
      str = str..fieldToString(v).." "
    end
    return (str.."}")
  else
    return tostring(x)
  end
end

function Tetmesh:dumpTetFieldToFile(field, file_name)
  local field_list = self.tetrahedra[field]:DumpToList()
  local field_file = PN.scriptdir() .. file_name
  local out = io.open(tostring(field_file), 'w')
  for i = 1, #field_list do
    out:write(fieldToString(field_list[i]) .. "\n" )
  end
  out:close()
end

function Tetmesh:dumpEdgeFieldToFile(field, file_name)
  local field_list = self.edges[field]:DumpToList()
  local tail_list = self.edges.tail:DumpToList()
  local head_list = self.edges.head:DumpToList()
  local field_file = PN.scriptdir() .. file_name
  local out = io.open(tostring(field_file), 'w')
  for i = 1, #field_list do
    out:write( tostring(tail_list[i]) .. "  " ..
               tostring(head_list[i]) .. "  " ..
               fieldToString(field_list[i]) .. "\n" )
  end
  out:close()
end

function Tetmesh:dumpVertFieldToFile(field, file_name)
  local field_list = self.vertices[field]:DumpToList()
  local field_file = PN.scriptdir() .. file_name
  local out = io.open(tostring(field_file), 'w')
  for i = 1, #field_list do
    out:write(fieldToString(field_list[i]) .. "\n" )
  end
  out:close()
end

function Tetmesh:dumpDeformationToFile(file_name)
  local pos = self.vertices.pos:DumpToList()
  local d = self.vertices.q:DumpToList()
  local field_file = PN.scriptdir() .. file_name
  local out = io.open(tostring(field_file), 'w')
  for i = 1, #pos do
    out:write(tostring(pos[i][1] + d[i][1]) .. ", " ..
              tostring(pos[i][2] + d[i][2]) .. ", " ..
              tostring(pos[i][3] + d[i][3]) .. "\n" )
  end
  out:close()
end
