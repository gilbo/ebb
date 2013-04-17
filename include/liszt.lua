LisztObj = { }
function LisztObj.isTopoElem () return false end
function LisztObj.isTopoSet  () return false end
function LisztObj.isDataType () return false end
LisztObj.__index = LisztObj

--[[ Liszt Types ]]--
TopoElem = { }
setmetatable(TopoElem, LisztObj)
function TopoElem.isTopoElem () return true end
TopoElem.__index = TopoElem

TopoSet  = { }
setmetatable(TopoSet, LisztObj)
function TopoSet.isTopoSet   () return true end
TopoSet.__index = TopoSet

Mesh  = { }
Mesh.__index = Mesh
setmetatable(Mesh, LisztObj)

Field = { }
setmetatable(Field, LisztObj)

Cell   = { }
Face   = { }
Edge   = { }
Vertex = { }
setmetatable(Cell,   TopoElem)
setmetatable(Face,   TopoElem)
setmetatable(Edge,   TopoElem)
setmetatable(Vertex, TopoElem)

DataType = { }
setmetatable(DataType, LisztObj)
function DataType.isDataType () return true end
DataType.__index = DataType

Int   = { }
Float = { }

setmetatable(Int,   DataType)
setmetatable(Float, DataType)

Vector = { }

function Vector.type (num, data_type) 
   if not type(num) == "number" or num < 1 or num % 1 ~= 0 then
      error("First argument to Vector.type() should be a non-negative integer!")
   end
   if not (data_type.isDataType and data_type.isDataType()) then
      error("second argument to Vector.type() should be a Liszt data type!")
   end
   return { size = num, data_type = data_type }
end

function Vector.new(data_type, ...) 
   local vec = {}
   setmetatable(vec, Vector)
   return vec
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

function Mesh.field (self, topo_elem, data_type, initial_val)
   local field = { }
   field.mesh  = self
   setmetatable(field, Field)
   return field
end

function Mesh.fieldWithLabel (self, topo_elem, data_type, label)
   local field = { }
   field.mesh  = self
   setmetatable(field, Field)
   return field
end

function Mesh.new () 
   local mesh = { }
   setmetatable(mesh, Mesh)
   return mesh
end

LoadMesh = function (filename)
   return Mesh.new()
end




