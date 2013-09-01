-- import runtime, privately to this module (so that it is not exposed to liszt application programmers)
terralib.require('runtime/liszt')
local runtime = runtime
_G.runtime    = nil

local LisztObj = { }

--[[
-- data_type represents type of elements for vectors/ sets/ fields.
-- topo_type represents the type of the topolgical element for a field over
-- some topological set.
-- data_type is int/ float for vectors, vertex/ edge/ face/ cell for
-- topological sets, and ObjType for fields. topo_type is always vertex/ edge/
-- face/ cell. ObjType contains strings for types ("int" and "float") and not
-- the actual type (int or float), since the type could be a vector too.
--]]

--[[ String literals ]]--
local NOTYPE = 'notype'
local TABLE  = 'table'
local INT    = 'int'
local FLOAT  = 'float'
local VECTOR = 'vector'
local VERTEX = 'vertex'
local EDGE   = 'edge'
local FACE   = 'face'
local CELL   = 'cell'

-- need to use this for fields to store the type of field, due to nested types
local ObjType = 
{
    -- type of the object
	obj_type = NOTYPE,
    -- if object consists of elements, then type of elements
	elem_type = NOTYPE,
	-- size of the object (example, vector length)
	size = 0,
}

function ObjType:new()
	return setmetatable({}, {__index = self})
end

--[[ Liszt Types ]]--
local TopoElem = setmetatable({kind = "topoelem"}, { __index = LisztObj, __metatable = "TopoElem" })
local TopoSet  = setmetatable({kind = "toposet", data_type = NOTYPE},  { __index = LisztObj, __metatable = "TopoSet" })
local Field    = setmetatable({kind = "field",  topo_type = NOTYPE, data_type = ObjType}, { __index = LisztObj, __metatable = "Field" })
Scalar   = setmetatable({kind = "scalar", data_type = NOTYPE},                     { __index = LisztObj, __metatable = "Scalar"})

Field  = setmetatable({kind = "field",  topo_type = NOTYPE, data_type = NOTYPE}, { __index = LisztObj, __metatable = "Field" })
Scalar = setmetatable({kind = "scalar", data_type = NOTYPE},                     { __index = LisztObj, __metatable = "Scalar"})

Mesh   = setmetatable({kind = "mesh"},   { __index = LisztObj, __metatable = "Mesh"})
Cell   = setmetatable({kind = "cell"},   { __index = TopoElem, __metatable = "Cell"})
Face   = setmetatable({kind = "face"},   { __index = TopoElem, __metatable = "Face"})
Edge   = setmetatable({kind = "edge"},   { __index = TopoElem, __metatable = "Edge"})
Vertex = setmetatable({kind = "vertex"}, { __index = TopoElem, __metatable = "Vertex"})

local DataType   = setmetatable({kind = "datatype"}, { __index=LisztObj, __metatable="DataType"})
Vector           = setmetatable({kind = "vector", data_type = NOTYPE, size = 0},   { __index=DataType})
local VectorType = setmetatable({kind = "vector", data_type = NOTYPE, size = 0}, { __index=DataType})
Vector.__index     = Vector
VectorType.__index = VectorType


local function contains_entry (tbl, entry)
   local set
   pcall(function () set = tbl[entry] ~= nil end)
   return set
end

-------------------------------------------------
--[[ Field methods                           ]]--
-------------------------------------------------
local elemTypeToStr = {
   [Vertex] = VERTEX,
   [Edge]   = EDGE,
   [Face]   = FACE,
   [Cell]   = CELL
}

local vectorTypeToStr = {
   [int]   = INT,
   [float] = FLOAT
}

function Field:set_topo_type(topo_type)
   if (type(topo_type) ~= TABLE) then
	   error("Field over unrecognized topological type!!", 3)
   end
   self.topo_type = elemTypeToStr[topo_type]

   if not self.topo_type then
	   error("Field over unrecognized topological type!!", 3)
   end
end

function Field:set_data_type(data_type)
	if vectorTypeToStr[data_type] then
		self.data_type.obj_type  = vectorTypeToStr[data_type]
		self.data_type.elem_type = vectorTypeToStr[data_type]
		self.data_type.size = 1
	elseif Vector.isVector(data_type) or Vector.isVectorType(data_type) then
		self.data_type.obj_type = VECTOR
		if vectorTypeToStr[data_type.data_type] then
			self.data_type.elem_type = vectorTypeToStr[data_type.data_type]
		else
			error("Field over unsupported data type!!", 3)
		end
	   self.data_type.size = data_type.size
   else
	   error("Field over unsupported data type!!", 3)
   end
end

function Field:lField ()
   return self.lfield
end

function Field:lkField ()
   return self.lkfield
end

-------------------------------------------------
--[[ Scalar methods                          ]]--
-------------------------------------------------
function Scalar:lScalar ()
   return self.lscalar
end

function Scalar:lkScalar()
   return self.lkscalar
end

-------------------------------------------------
--[[ Runtime type conversion                 ]]--
-------------------------------------------------
-- topological element types mapped to Liszt types
local lElementTypeMap = {
   [Vertex] = runtime.L_VERTEX,
   [Cell]   = runtime.L_CELL,
   [Face]   = runtime.L_FACE,
   [Edge]   = runtime.L_EDGE
}

-- Valid scalar types, mapped to Liszt types
local lKeyTypeMap = {
   [int]   = runtime.L_INT,
   [float] = runtime.L_FLOAT,
   [bool]  = runtime.L_BOOL,
}

local function runtimeDataType (data_type)
   if Vector.isVectorType(data_type) then
      return lKeyTypeMap[data_type.data_type], data_type.size
   else
      return lKeyTypeMap[data_type], 1.0
   end
end

local function isPositiveInteger (elem)
   return type(elem) == 'number' and elem > 0 and elem % 1 == 0
end


-------------------------------------------------
--[[ Vector methods                          ]]--
-------------------------------------------------
function Vector.type (data_type, size)
   if not lKeyTypeMap[data_type] then
      error("First argument to Vector.type() should be a Liszt-supported terra data type!", 2)
   end
   if not isPositiveInteger(size) then
      error("Second argument to Vector.type() should be a non-negative integer!", 2)
   end

   return setmetatable({size = size, data_type = data_type}, VectorType)
end

-- second argument specifies either the length of the vector, or the contents
function Vector.new(data_type, arg) 
   if not lKeyTypeMap[data_type] then
      error("First argument to Vector.new() should be a Liszt-supported terra data type!", 2)
   end
   if type(arg) ~= 'table' and isPositiveInteger(arg) then
      error("Second argument to Vector.new() should be a list of numbers or a non-negative integer!", 2)
   end

   local init, size
   if type(arg) == 'table' then
      size = #arg
      init = arg
   else
      size = arg
      init = {}
      for i = 1, size do init[i] = 0 end
   end

   local v = global(vector(data_type, size))

   for i = 1, size do
      -- Check type of each entry in initialization vector
      if data_type == float or data_type == int then
         if type(init[i]) ~= 'number' 
            then error("Cannot initialize vector with non-numeric type!", 2) 
         end
      else
         if (type(init[i]) ~= 'boolean') then 
            error("Cannot initialize vector with non-boolean type!", 2)
         end
      end

      v[i-1] = init[i]
   end

   return setmetatable({size = size, data_type = data_type, __data = v, initialized= #init > 0 }, Vector)
end

function Vector.isVector (obj)
   return getmetatable(obj) == Vector
end

function Vector.isVectorType (obj)
   return getmetatable(obj) == VectorType
end

function Vector.add (v1, v2)
   if not Vector.isVector(v2) then
      error("Cannot add non-vector type " .. type(v2) .. "to vector")
   elseif v1.data_type ~= v2.data_type then
      error("Cannot add vectors of differing types!", 2)
   elseif v1.size ~= v2.size then
      error("Cannot add vectors of differing lengths!", 2)
   end
end



-------------------------------------------------
--[[ TopoSet methods                         ]]--
-------------------------------------------------
local function toposet_stub (topoelem)
   local tmp = {data_type = topoelem}
   tmp.size  = function () return 10 end
   tmp.map   = function (...) end
   setmetatable(tmp, {__index = TopoSet})
   return tmp
end

Mesh.cells    = toposet_stub(CELL)
Mesh.faces    = toposet_stub(FACE)
Mesh.vertices = toposet_stub(VERTEX)
Mesh.edges    = toposet_stub(EDGE)


-------------------------------------------------
--[[ Mesh methods                            ]]--
-------------------------------------------------
function Mesh:field (topo_type, data_type, initial_val)
   local field = setmetatable({ mesh = self, topo_type = NOTYPE, data_type = ObjType:new() }, { __index = Field })
   field:set_topo_type(topo_type)
   field:set_data_type(data_type)
   local val_type, val_len = runtimeDataType(data_type)
   field.lfield  = runtime.initField(self.ctx, lElementTypeMap[topo_type], val_type, val_len)
   field.lkfield = runtime.getlkField(field.lfield)
   return field
end

function Mesh:fieldWithLabel (topo_type, data_type, label)
   local field = {topo_type = NOTYPE, data_type = ObjType:new(), mesh = self}
   setmetatable(field, { __index = Field })
   field:set_topo_type(topo_type)
   field:set_data_type(data_type)
   local val_type, val_len = runtimeDataType(data_type)
   field.lfield  = runtime.loadField(self.ctx, label, lElementTypeMap[topo_type], val_type, val_len)
   field.lkfield = runtime.getlkField(field.lfield)
   return field
end

function Mesh:scalar (data_type)
   local scalar_type, scalar_length
   if Vector.isVector and lKeyTypeMap[data_type.data_type] then
	   scalar_type = lKeyTypeMap[data_type.data_type]
	   scalar_length = data_type.size
   elseif lKeyTypeMap[data_type] then
	   scalar_type = lKeyTypeMap[data_type]
	   scalar_length = 1
   else
      error("First argument to mesh:scalar must be a Liszt-supported data type!", 2)
   end

   local s    = setmetatable({}, {__index = Scalar})
   s.lscalar  = runtime.initScalar(self.ctx, scalar_type, scalar_length)
   s.lkscalar = runtime.getlkScalar(s.lscalar)
   return s
end

function Mesh.new () 
   return setmetatable({ }, { __index = Mesh } )
end

LoadMesh = function (filename)
   local mesh = Mesh.new()
   mesh.ctx   = runtime.loadMesh(filename)
   return mesh
end
