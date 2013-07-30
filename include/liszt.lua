-- import runtime, privately to this module (so that it is not exposed to liszt application programmers)
terralib.require('runtime/liszt')
local runtime = runtime
_G.runtime    = nil

local LisztObj = { }

--[[ String literals ]]--
local NOTYPE = 'Notype'
local TABLE = 'table'
local INT = 'Int'
local FLOAT = 'Float'
local VERTEX = 'Vertex'
local EDGE = 'Edge'
local FACE = 'Face'
local CELL = 'Cell'

--[[ Liszt Types ]]--
local TopoElem = setmetatable({kind = "topoelem"}, { __index = LisztObj, __metatable = "TopoElem" })
local TopoSet  = setmetatable({kind = "toposet"}, { __index = LisztObj, __metatable = "TopoSet" })
local Field    = setmetatable({kind = "field", topo_type = NOTYPE, data_type = NOTYPE}, { __index = LisztObj, __metatable = "Field" })

Mesh   = setmetatable({kind = "mesh"}, { __index = LisztObj, __metatable = "Mesh"})
Cell   = setmetatable({kind = "cell"}, { __index = TopoElem, __metatable = "Cell"})
Face   = setmetatable({kind = "face"}, { __index = TopoElem, __metatable = "Face"})
Edge   = setmetatable({kind = "edge"}, { __index = TopoElem, __metatable = "Edge"})
Vertex = setmetatable({kind = "vertex"}, { __index = TopoElem, __metatable = "Vertex"})

DataType = setmetatable({kind = "datatype"}, { __index=LisztObj, __metatable="DataType"})
Vector   = setmetatable({kind = "vector"}, { __index=DataType})

function Field:set_topo_type(topo_type)
   -- TODO: handle errors
   if (type(topo_type) ~= TABLE) then
	   print("*** Topological type should be a table!!")
   end
   if topo_type == Vertex then
	   self.topo_type = VERTEX
   elseif topo_type == Edge then
	   self.topo_type = EDGE
   elseif topo_type == Face then
	   self.topo_type = FACE
   elseif topo_type == Cell then
	   self.topo_type = CELL
   else
	   print("*** Unrecognized topological type!!")
   end
end

function Field:set_data_type(data_type)
   -- TODO: handle errors
   if data_type == int  then
	   self.data_type = INT
   elseif data_type == float  then
	   self.data_type = FLOAT
   else
	   print("*** Unrecognized data type!!")
   end
end

function Vector.type (data_type, num)
   if not (data_type == int or
           data_type == float) then
      error("First argument to Vector.type() should be a terra data type!")
   end
   if not type(num) == "number" or num < 1 or num % 1 ~= 0 then
      error("Second argument to Vector.type() should be a non-negative integer!")
   end
   return setmetatable({size = num, data_type = data_type}, {__index = Vector})
end

function Vector.new(data_type, ...) 
   local vec = { ... }
   local num = #vec
   vec.data_type = data_type
   vec.size = num
   setmetatable(vec, {__index = Vector})
   return vec
end

function Vector.isVector (obj)
   return getmetatable(obj) == Vector
end

--[[ Mesh Construction and methods ]]--
local function toposet_stub ()
   local tmp = { }
   tmp.mt    = TopoSet
   return tmp
end

Mesh.cells    = toposet_stub
Mesh.faces    = toposet_stub
Mesh.vertices = toposet_stub
Mesh.edges    = toposet_stub

local lElementTypeMap = {
   [Vertex] = runtime.L_VERTEX,
   [Cell]   = runtime.L_CELL,
   [Face]   = runtime.L_FACE,
   [Edge]   = runtime.L_EDGE
}


local lKeyTypeMap = {
   [int]   = runtime.L_INT,
   [float] = runtime.L_FLOAT
}

local function runtimeDataType (data_type)
   if getmetatable(data_type) == Vector then
      return data_type.data_type, data_type.num
   else
      return lKeyTypeMap[data_type], 1.0
   end
end


function Mesh:field (topo_type, data_type, initial_val)
   local field = { }
   field.mesh  = self
   setmetatable(field, { __index = Field })
   field:set_topo_type(topo_type)
   field:set_data_type(data_type)
   local val_type, val_len = runtimeDataType(data_type)
   field.field = runtime.initField(self, lElementTypeMap[topo_type], val_type, val_len)
   return field
end

function Mesh:fieldWithLabel (topo_type, data_type, label)
   local field = { }
   field.mesh  = self
   setmetatable(field, { __index = Field })
   field:set_topo_type(topo_type)
   field:set_data_type(data_type)
   field.field = runtime.loadField(self, label, lElementTypeMap[topo_type], data_type, 3)
   field.id    = self.num_fields
   self.num_fields = self.num_fields + 1
   return field
end

function Mesh.new () 
   local m      = setmetatable({ }, { __index = Mesh } )
   m.num_fields = 0
   return m
end

LoadMesh = function (filename)
   local mesh      = Mesh.new()
   mesh.ctx        = runtime.loadMesh(filename)
   return mesh
end
