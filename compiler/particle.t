import "compiler.liszt"
local Particle = {}
package.loaded["compiler.particle"] = Particle
local print, dot = L.print, L.dot

c = terralib.require "compiler.c"

local function RelationPairs(relation, first_name, second_name)
    local result = {}
    for i=0,relation:Size() - 1 do
        table.insert(result, {tonumber(relation[first_name].data[i]),
                              tonumber(relation[second_name].data[i])})
    end
    return (function(table, i)
        if table[i] == nil then return nil end
        return i + 1, unpack(table[i])
    end), result, 1
end

local function LuaToCArray(lua_array, tp)
    local result = terralib.cast(&tp, c.malloc(#lua_array * terralib.sizeof(tp)))
    for i=0,#lua_array - 1 do
        array[i] = lua_array[i + 1]
    end
    return result
end

local function GatherVertices(mesh, cells)
    for _,cell,vertex in RelationPairs(mesh.verticesofcell, "cell", "vertex") do
        if cells[cell] == nil then
            cells[cell] = {}
            cells[cell].lua_vertices = {}
            cells[cell].lua_faces = {}
            cells[cell].swift_index_of = {}
        end
        local position = mesh.vertices.position.data[vertex]
        for i=0,2 do
            table.insert(cells[cell].lua_vertices, position[i])
        end
        cells[cell].num_vertices = #cells[cell].lua_vertices / 3
        cells[cell].swift_index_of[vertex] = cells[cell].num_vertices - 1
    end
end

local function GatherFaces(mesh, cells)
    local faces = {}
    for _,face,vertex in RelationPairs(mesh.verticesofface, "face", "vertex") do
        if faces[face] == nil then faces[face] = terralib.newlist() end
        table.insert(faces[face], vertex)
    end
    for _,cell,face in RelationPairs(mesh.facesofcell, "cell", "face") do
        -- Decompose each face into tris
        for i=3,#faces[face] do
            table.insert(cells[cell].lua_faces, cells[cell].swift_index_of[faces[face][1]])
            table.insert(cells[cell].lua_faces, cells[cell].swift_index_of[faces[face][i - 1]])
            table.insert(cells[cell].lua_faces, cells[cell].swift_index_of[faces[face][i - 2]])
        end
        cells[cell].num_faces = #cells[cell].lua_faces / 3
    end
end

local function AddMeshCells(mesh)
    local cells = {}
    local liszt_id_of = {}
    GatherVertices(mesh, cells)
    GatherFaces(mesh, cells)
    for _,cell in ipairs(cells) do
        local cvertices = LuaToCArray(cell.lua_vertices, c.SWIFT_Real)
        local cfaces = LuaToCArray(cell.lua_faces, int)
        local id = terralib.global(int)
        c.swift_add_object(mesh._swift_scene, cvertices, cfaces,
                cell.num_vertices, cell.num_faces, id:getpointer(), true)
        liszt_id_of[swift_id] = id
        c.free(cfaces)
        c.free(cvertices)
    end
    terralib.tree.printraw(cells)
end

function Particle.init(mesh, numParticles)
    mesh.particles = LDB.NewRelation(numParticles, "particles")
    local position = mesh.particles:NewField('position', L.vector(L.float, 3))
    mesh.particles:NewField('cell', mesh.cells):LoadFromCallback(terra(mem : &uint64, i : uint)
        mem[0] = 0
    end)
    
    mesh.particles:NewField('_swift_id', L.int):LoadFromCallback(terra(mem : &int, i : uint)
        mem[0] = -1
    end)
    mesh._swift_scene = c.swift_create_scene()
    AddMeshCells(mesh)
    return position
end

function Particle.update(mesh)
    -- TODO: collision detection on mesh using acceleration structure
    (liszt kernel(p in mesh.particles)
        if dot(p.position, {1, 0, 0}) > 1.0 then
            p.cell = 1
        else
            p.cell = 0
        end
    end)()
end
