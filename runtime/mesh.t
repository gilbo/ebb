local L = {}
local runtime = terralib.includec("runtime/src/liszt_runtime.h")
local mesh_h = terralib.includec("runtime/src/mesh_crs.h")
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

local terra loadMesh (filename : rawstring) : &runtime.lContext
	var ctx : &runtime.lContext = runtime.lLoadContext(filename)
	return ctx
end

L.loadMesh = function (filename)
	local ctx = loadMesh(filename)
	local mesh = runtime.lMeshFromContext(ctx)
	return ctx, mesh
end

return L
