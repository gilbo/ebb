terralib.require('runtime/liszt')
-- Keep runtime module private
local runtime = runtime
_G.runtime    = nil

local semant = require 'semant'
_G.semant    = nil

--[[ String literals ]]--
local NOTYPE   = semant._NOTYPE_STR
local TABLE    = semant._TABLE_STR
local INT      = semant._INT_STR
local FLOAT    = semant._FLOAT_STR
local BOOL     = semant._BOOL_STR
local VECTOR   = semant._VECTOR_STR
local VERTEX   = semant._VERTEX_STR
local EDGE     = semant._EDGE_STR
local FACE     = semant._FACE_STR
local CELL     = semant._CELL_STR
local MESH     = semant._MESH_STR
local FIELD    = semant._FIELD_STR
local SCALAR   = semant._SCALAR_STR
local TOPOSET  = semant._TOPOSET_STR
local TOPOELEM = semant._ELEM_STR

--[[
-- data_type represents type of elements for vectors/ sets/ fields.
-- topo_type represents the type of the topological element type covered by
-- a field or a toposet.
-- data_type is int/float/bool for vectors, can be a vector for fields
-- topo_type is always vertex/ edge/
-- face/ cell. ObjType contains strings for types ("int" and "float") and not
-- the actual type (int or float), since the type could be a vector too.
--]]
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

local LisztObj = { }

--[[ Liszt Types ]]--
local TopoElem = setmetatable({kind = TOPOELEM},                                         { __index = LisztObj, __metatable = "TopoElem" })
local TopoSet  = setmetatable({kind = TOPOSET, topo_type = NOTYPE },                     { __index = LisztObj, __metatable = "TopoSet"})
local Field    = setmetatable({kind = FIELD,   topo_type = NOTYPE, data_type = ObjType}, { __index = LisztObj, __metatable = "Field" })
local Scalar   = setmetatable({kind = SCALAR,                      data_type = NOTYPE},  { __index = LisztObj, __metatable = "Scalar"})

Mesh   = setmetatable({kind = MESH},   { __index = LisztObj, __metatable = "Mesh"})
Cell   = setmetatable({kind = CELL},   { __index = TopoElem, __metatable = "Cell"})
Face   = setmetatable({kind = FACE},   { __index = TopoElem, __metatable = "Face"})
Edge   = setmetatable({kind = EDGE},   { __index = TopoElem, __metatable = "Edge"})
Vertex = setmetatable({kind = VERTEX}, { __index = TopoElem, __metatable = "Vertex"})

Vector           = setmetatable({kind = VECTOR, data_type = NOTYPE, size = 0}, { __index=LisztObj})
local VectorType = setmetatable({kind = VECTOR, data_type = NOTYPE, size = 0}, { __index=LisztObj})
Vector.__index     = Vector
VectorType.__index = VectorType


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
   [float] = FLOAT,
   [bool]  = BOOL,
}

function Field:set_topo_type(topo_type)
   self.topo_type = elemTypeToStr[topo_type]
   if not self.topo_type then
	   error("Field over unrecognized topological type", 3)
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
			error("Field over unsupported data type", 3)
		end
	   self.data_type.size = data_type.size
   else
	   error("Field over unsupported data type", 3)
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

function Scalar:setTo(val)
end

function Scalar:value()
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

local function isValidVectorType (tp)
   return lKeyTypeMap[tp] ~= nil
end


local function isValidDataType (tp)
   if isValidVectorType(tp) then return true end
   if Vector.isVectorType(tp) and isValidVectorType(tp.data_type) then return true end
   return false
end

-- Can the instance be converted into a meaningful Liszt data type?
local function conformsToDataType (inst, tp)
   if tp == int or tp == float or tp == double then return type(inst) == 'number'
   elseif tp == bool then return type(inst) == 'boolean'
   elseif Vector.isVectorType(tp) then
      -- behavior here is that any numeric vector can be initialized by another numeric
      -- vector, but bool vectors can only be initialized by bool vectors
      if Vector.isVector(inst) then return type(inst.data_type) == type(tp.data_type) and inst.size == tp.size end

      -- we also accept arrays as instances of vectors
      if type(inst) == 'table' then
         -- make sure array is of the correct length
         if #inst ~= tp.size then return false end
         -- make sure each element conforms to the vector's data type
         for i = 1, #inst do
            if not conformsToDataType(inst[i], tp.data_type) then return false end
         end
         return true
      end
   end
   return false
end

-- assuming data-type refers to a valid liszt data type, 
-- return the liszt lType and size of the type, both of which
-- are needed as arguments for field/scalar runtime functions
local function runtimeDataType (data_type)
   if Vector.isVectorType(data_type) then
      return lKeyTypeMap[data_type.data_type], data_type.size
   else
      return lKeyTypeMap[data_type], 1.0
   end
end

-- used for verifying size_t-type arguments to constructor fns, etc.
local function isPositiveInteger (elem)
   return type(elem) == 'number' and elem > 0 and elem % 1 == 0
end


-------------------------------------------------
--[[ Vector methods                          ]]--
-------------------------------------------------
function Vector.type (data_type, size)
   if not isValidVectorType(data_type) then
      error("First argument to Vector.type() should be a Liszt-supported terra data type", 2)
   end
   if not isPositiveInteger(size) then
      error("Second argument to Vector.type() should be a non-negative integer", 2)
   end

   return setmetatable({size = size, data_type = data_type}, VectorType)
end

function Vector.new(data_type, init) 
   if not isValidVectorType(data_type) then
      error("First argument to Vector.new() should be a Liszt-supported terra data type", 2)
   end

   -- Vectors can be initialized with table literals (e.g. "{1, 2, 3}") or other Liszt Vectors
   if not conformsToDataType(init, Vector.type(data_type, #init)) then
      error("Second argument to Vector.new() should be a Vector or an array", 2)
   end

   local data = {}
   if Vector.isVector(init) then init = init.data end
   for i = 1, #init do
      data[i] = data_type == int and init[i] % 1 or init[i]
   end

   return setmetatable({size = #init, data_type = data_type, data = data}, Vector)
end

function Vector:__codegen ()
   if self.size == 3 then
      local v1, v2, v3 = self.data[1], self.data[2], self.data[3]
      local s = symbol()

      local q = quote
         var [s] = vectorof(self.data_type, v1, v2, v3)
      in
         [s]
      end
      return q
   elseif self.size == 4 then
      local v1, v2, v3, v4 = self.data[1], self.data[2], self.data[3], self.data[4]
      local s = symbol()

      local q = quote
         var [s] = vectorof(self.data_type, v1, v2, v3, v4)
      in
         [s]
      end
      return q
   elseif self.size == 5 then
      local v1, v2, v3, v4, v5 = self.data[1], self.data[2], self.data[3], self.data[4], self.data[5]
      local s = symbol()

      local q = quote
         var [s] = vectorof(self.data_type, v1, v2, v3, v4, v5)
      in
         [s]
      end
      return q
   end

   local s = symbol()
   local t = symbol()
   local q = quote
      var [s] : vector(self.data_type, self.size)
      var [t] = [&self.data_type](&s)
   end

   for i = 1, self.size do
      local val = self.data[i]
      q = quote 
         [q] 
         @[t] = [val]
         t = t + 1
      end
   end
   return quote [q] in [s] end
end

function Vector.isVector (obj)
   return getmetatable(obj) == Vector
end

function Vector.isVectorType (obj)
   return getmetatable(obj) == VectorType
end

function new_type (t1, t2)
   if t1 == double or  t2 == double then return double end
   if t1 == float  or  t2 == float  then return float  end
   return int
end

function Vector.__add (v1, v2)
   if not Vector.isVector(v1) or not Vector.isVector(v2) then
      error("Cannot add non-vector type to vector", 2)
   elseif v1.size ~= v2.size then
      error("Cannot add vectors of differing lengths", 2)
   elseif v1.data_type == bool or v2.data_type == bool then
      error("Cannot add boolean vectors", 2)
   end

   local data = { }
   local tp = new_type(v1.data_type, v2.data_type)

   for i = 1, #v1.data do
      data[i] = v1.data[i] + v2.data[i]
   end
   return Vector.new(tp, data)
end

function Vector.__sub (v1, v2)
   if not Vector.isVector(v1) then
      error("Cannot subtract vector from non-vector type", 2)
   elseif not Vector.isVector(v2) then
      error("Cannot subtract non-vector type from vector", 2)
   elseif v1.size ~= v2.size then
      error("Cannot subtract vectors of differing lengths", 2)
   elseif v1.data_type == bool or v2.data_type == bool then
      error("Cannot subtract boolean vectors", 2)
   end

   local data = { }
   local tp = new_type(v1.data_type, v2.data_type)

   for i = 1, #v1.data do
      data[i] = v1.data[i] - v2.data[i]
   end
   return Vector.new(tp, data)
end

function Vector.__mul (a1, a2)
   if Vector.isVector(a1) and Vector.isVector(a2) then
      error("Cannot multiply two vectors", 2)
   end
   local v, a
   if Vector.isVector(a1) then v, a = a1, a2 else v, a = a2, a1 end

   if     v.type == bool      then error("Cannot multiply a non-numeric vector", 2)
   elseif type(a) ~= 'number' then error("Cannot multiply a vector by a non-numeric type", 2)
   end

   local data = {}
   for i = 1, #v.data do
      data[i] = v.data[i] * a
   end
   return Vector.new(float, data)
end

function Vector.__div (v, a)
   if     Vector.isVector(a)  then error("Cannot divide by a vector", 2)
   elseif v.data_type == bool then error("Cannot divide a non-numeric vector", 2)
   elseif type(a) ~= 'number' then error("Cannot divide a vector by a non-numeric type", 2)
   end

   local data = {}
   for i = 1, #v.data do
      data[i] = v.data[i] / a
   end
   return Vector.new(float, data)
end

function Vector.__mod (v1, a2)
   if Vector.isVector(a2) then error("Cannot modulus by a vector", 2) end
   local data = {}
   for i = 1, v1.size do
      data[i] = v1.data[i] % a2
   end
   local tp = new_type(v1.data_type, float)
   return Vector.new(tp, data)
end

function Vector.__unm (v1)
   if v1.data_type == bool then error("Cannot negate a non-numeric vector", 2) end
   local data = {}
   for i = 1, #v1.data do
      data[i] = -v1.data[i]
   end
end

function Vector.__eq (v1, v2)
   if v1.size ~= v2.size then return false end
   for i = 1, v1.size do
      if v1.data[i] ~= v2.data[i] then return false end
   end
   return true
end


-------------------------------------------------
--[[ TopoSet methods                         ]]--
-------------------------------------------------
TopoSet.__index = TopoSet

local size_fn = {
   [Cell]   = runtime.numCells,
   [Face]   = runtime.numFaces,
   [Edge]   = runtime.numEdges,
   [Vertex] = runtime.numVertices
}

function TopoSet.new (mesh, topo_type)
   local size   = size_fn[topo_type](mesh.__ctx)
   local ts     = setmetatable({__mesh=mesh, __type=topo_type,__size=size}, TopoSet)
   -- table with weak keys/values to keep a small cache of most-previously used kernels
   ts.__kernels = setmetatable({}, {__mode="kv"})
   return ts
end

function TopoSet:size ()
   return self.__size
end

local topoToInitFn = {
   [Cell]   = runtime.lCellsOfMesh,
   [Face]   = runtime.lFacesOfMesh,
   [Edge]   = runtime.lEdgesOfMesh,
   [Vertex] = runtime.lVerticesOfMesh
}

function TopoSet:__init_symbol (ctx_symb, set_symb)
   local setInitFunction = topoToInitFn[self.__type]
   return quote setInitFunction([ctx_symb], [set_symb]) end
end

local topoToLiszt = {
   [Cell]   = runtime.L_CELL,
   [Face]   = runtime.L_FACE,
   [Edge]   = runtime.L_EDGE,
   [Vertex] = runtime.L_VERTEX
}
function TopoSet:map (kernel)
   if not kernel:acceptsType(self.__type) then error("Kernel cannot iterate over set of this topological type") end

   local run_kernel = self.__kernels[kernel]
   if not run_kernel then
      local kernel_fn = kernel:generate(self)

      local set = symbol()
      local l_topo = topoToLiszt[self.__type]

      run_kernel = terra (ctx : &runtime.lContext)
         var [set] : &runtime.lSet = runtime.lNewlSet()
         [self:__init_symbol(ctx, set)]
         runtime.lKernelRun(ctx, [set], l_topo, 0, kernel_fn)
         runtime.lFreelSet([set])
      end

      self.__kernels[kernel] = run_kernel
   end

   run_kernel(self.__mesh.__ctx)
end


-------------------------------------------------
--[[ Mesh methods                            ]]--
-------------------------------------------------
function Mesh:field (topo_type, data_type, initial_val)
   local field = setmetatable({ mesh = self, topo_type = NOTYPE, data_type = ObjType:new() }, { __index = Field })
   field:set_topo_type(topo_type)
   field:set_data_type(data_type)
   local val_type, val_len = runtimeDataType(data_type)
   field.lfield  = runtime.initField(self.__ctx, lElementTypeMap[topo_type], val_type, val_len)
   field.lkfield = runtime.getlkField(field.lfield)
   return field
end

function Mesh:fieldWithLabel (topo_type, data_type, label)
   local field = setmetatable({topo_type = NOTYPE, data_type = ObjType:new(), mesh = self}, { __index=Field})
   field:set_topo_type(topo_type)
   field:set_data_type(data_type)
   local val_type, val_len = runtimeDataType(data_type)
   field.lfield  = runtime.loadField(self.__ctx, label, lElementTypeMap[topo_type], val_type, val_len)
   field.lkfield = runtime.getlkField(field.lfield)
   return field
end

function Mesh:scalar (data_type, init)
   if not isValidDataType(data_type) then
      error("First argument to mesh:scalar must be a Liszt-supported data type", 2)
   end
   if not conformsToDataType(init, data_type) then
      error("Second argument to mesh:scalar must be an instance of the specified data type", 2)
   end
   local scalar_type, scalar_length = runtimeDataType(data_type)
   
   local lscalar  = runtime.initScalar(self.__ctx, scalar_type, scalar_length)
   local lkscalar = runtime.getlkScalar(lscalar)
   return setmetatable({ lscalar = lscalar, lkscalar = lkscalar }, {__index = Scalar})
end

LoadMesh = function (filename)
   local mesh    = setmetatable({}, {__index=Mesh})
   mesh.__ctx    = runtime.loadMesh(filename)
   mesh.cells    = TopoSet.new(mesh, Cell)
   mesh.faces    = TopoSet.new(mesh, Face)
   mesh.edges    = TopoSet.new(mesh, Edge)
   mesh.vertices = TopoSet.new(mesh, Vertex)
   return mesh
end
