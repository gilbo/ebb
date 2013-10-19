local L = {}
local mesh_h = terralib.includec("runtime/common/mesh_crs.h")
local c      = terralib.includec("stdio.h")
--
local util   = terralib.require("runtime/util")
util.link_runtime()

--[[
  First field represents index field and second field represents corresponding
  values.
- Oriented relations have an additional field orientation.
- For relation tables for vertices, edges, faces and cells, elements are
  initialized to length-1, with corresponding fields if any from
  mesh_rels_topo.
--]]

-- basic relations

terra L.readMesh(filename : rawstring) : &mesh_h.Mesh
    var mesh : &mesh_h.Mesh = mesh_h.lMeshInitFromFile(filename)
    return mesh
end

return L
