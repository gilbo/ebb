-- Privately import from types module
local _types = types
terralib.require('compiler/types')
local Type  = types.Type
local t     = types.t
local tutil = types.usertypes
_G.types    = _types

-- Keep runtime module private
local _runtime = runtime
terralib.require('runtime/liszt')
local runtime = runtime
_G.runtime    = _runtime


-------------------------------------------------
--[[ Liszt Objects ]]--
-------------------------------------------------
local function make_prototype(tb)
   tb.__index = tb
   return tb
end

local TopoElem = make_prototype {kind=Type.kinds.topo}
local TopoSet  = make_prototype {kind=Type.kinds.set}
local Field    = make_prototype {kind=Type.kinds.field}
local Scalar   = make_prototype {kind=Type.kinds.scalar}

local TopoElemMT = {__index=TopoElem}
local TopoSetMT  = {__index=TopoSet}
local FieldMT    = {__index=Field}
local ScalarMT   = {__index=Scalar}

Mesh   = {}
Cell   = setmetatable({type=t.cell},   TopoElemMT)
Face   = setmetatable({type=t.face},   TopoElemMT)
Edge   = setmetatable({type=t.edge},   TopoElemMT)
Vertex = setmetatable({type=t.vertex}, TopoElemMT)

local function VectorType (_vt, dt, N)
   if not tutil.isPrimitiveType(dt) then error("First argument to Vector() should be a Liszt-supported terra data type", 2) end
   if not tutil.isSize(N)           then error("Second argument to Vector() should be a non-negative integer", 2) end
   return t.vector(tutil.ltype(dt),N)
end
Vector   = setmetatable({kind=Type.kinds.vector}, {__call=VectorType})
VectorMT = {__index=Vector}


-------------------------------------------------
--[[ Field methods                           ]]--
-------------------------------------------------
function Field:setType(topo_obj, data_obj)
   if not tutil.isTopoType(topo_obj) then
      error("Field over unrecognized topological type", 3)
   end
   if not tutil.isDataType(data_obj) then
      error("Field over unsupported data type", 3)
   end
   self.topo = tutil.ltype(topo_obj)
   self.type = tutil.ltype(data_obj)
end


-------------------------------------------------
--[[ Scalar methods                          ]]--
-------------------------------------------------
function Scalar:setTo(val)
   if not tutil.conformsToType(val, self.type) then error("Cannot set scalar to value of incorrect type", 2) end

   -- compute expression for val expression contents
   local v = self.type:isVector() and Vector.new(self.type:terraBaseType(), val):__codegen() or `val
   local sctype, sclen = self.type:runtimeType()

   local terra writeSc () 
      var p : self.type:terraType() = [v]
      runtime.lScalarWrite(self.__ctx,self.__lscalar,runtime.L_ASSIGN,sctype,sclen,0,sclen,&p)
   end
   writeSc()
end

function Scalar:value()
   local sctype, sclen = self.type:runtimeType()
   if self.type:isVector() then
      local terra getSc (i : uint)
         var p : self.type:terraBaseType()
         runtime.lScalarRead(self.__ctx, self.__lscalar,sctype,sclen,i,1,&p)
         return p
      end

      local data = { }
      for i = 1, self.type.N do
         data[i] = getSc(i-1)
      end
      return Vector.new(self.type:terraBaseType(), data)
   else
      local terra getSc ()
         var p : self.type:terraType()
         runtime.lScalarRead(self.__ctx, self.__lscalar,sctype,sclen,0,sclen,&p)
         return p
      end
      return getSc()
   end
end


-------------------------------------------------
--[[ Vector methods                          ]]--
-------------------------------------------------

function Vector.new(dt, init)
   if not tutil.isPrimitiveType(dt)            then error("First argument to Vector.new() should be a Liszt-supported terra data type", 2) end
   if not Vector.isVector(init) and #init == 0 then error("Second argument to Vector.new should be either a Vector or an array", 2) end
   local N = Vector.isVector(init) and init.N or #init
   if not tutil.conformsToType(init, Vector(dt, N)) then
      error("Second argument to Vector.new() does not conform to specified type", 2)
   end

   local data = {}
   if Vector.isVector(init) then init = init.data end
   for i = 1, N do
      data[i] = dt == int and init[i] - init[i] % 1 or init[i] -- convert to integer if necessary
   end

   return setmetatable({N=N, type=t.vector(tutil.ltype(dt),N), data=data}, VectorMT)
end

function Vector:__codegen ()
   local v1, v2, v3, v4, v5, v6 = self.data[1], self.data[2], self.data[3], self.data[4], self.data[5], self.data[6]
   local btype = self.type:terraBaseType()

   if     self.N == 2 then
      return `vectorof(btype, v1, v2)
   elseif self.N == 3 then
      return `vectorof(btype, v1, v2, v3)
   elseif self.N == 4 then
      return `vectorof(btype, v1, v2, v3, v4)
   elseif self.N == 5 then
      return `vectorof(btype, v1, v2, v3, v4, v5)
   elseif self.N == 6 then
      return `vectorof(btype, v1, v2, v3, v4, v5, v6)
   end

   local s = symbol(self.type:terraType())
   local t = symbol()
   local q = quote
      var [s]
      var [t] = [&btype](&s)
   end

   for i = 1, self.N do
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
   return getmetatable(obj) == VectorMT
end

function VectorMT.__add (v1, v2)
   if not Vector.isVector(v1) or not Vector.isVector(v2) then
      error("Cannot add non-vector type to vector", 2)
   elseif v1.N ~= v2.N then
      error("Cannot add vectors of differing lengths", 2)
   elseif v1.type == t.bool or v2.type == t.bool then
      error("Cannot add boolean vectors", 2)
   end

   local data = { }
   local tp = types.type_meet(v1.type:baseType(), v2.type:baseType())

   for i = 1, #v1.data do
      data[i] = v1.data[i] + v2.data[i]
   end
   return Vector.new(tp:terraType(), data)
end

function VectorMT.__sub (v1, v2)
   if not Vector.isVector(v1) then
      error("Cannot subtract vector from non-vector type", 2)
   elseif not Vector.isVector(v2) then
      error("Cannot subtract non-vector type from vector", 2)
   elseif v1.N ~= v2.N then
      error("Cannot subtract vectors of differing lengths", 2)
   elseif v1.type == bool or v2.type == bool then
      error("Cannot subtract boolean vectors", 2)
   end

   local data = { }
   local tp = types.type_meet(v1.type:baseType(), v2.type:baseType())

   for i = 1, #v1.data do
      data[i] = v1.data[i] - v2.data[i]
   end

   return Vector.new(tp:terraType(), data)
end

function VectorMT.__mul (a1, a2)
   if Vector.isVector(a1) and Vector.isVector(a2) then error("Cannot multiply two vectors", 2) end
   local v, a
   if Vector.isVector(a1) then v, a = a1, a2 else v, a = a2, a1 end

   if     v.type == t.bool    then error("Cannot multiply a non-numeric vector", 2)
   elseif type(a) ~= 'number' then error("Cannot multiply a vector by a non-numeric type", 2)
   end

   local tm = float
   if v.type == int and a % 1 == 0 then tm = int end

   local data = {}
   for i = 1, #v.data do
      data[i] = v.data[i] * a
   end
   return Vector.new(tm, data)
end

function VectorMT.__div (v, a)
   if     Vector.isVector(a)  then error("Cannot divide by a vector", 2)
   elseif v.type == bool      then error("Cannot divide a non-numeric vector", 2)
   elseif type(a) ~= 'number' then error("Cannot divide a vector by a non-numeric type", 2)
   end

   local data = {}
   for i = 1, #v.data do
      data[i] = v.data[i] / a
   end
   return Vector.new(float, data)
end

function VectorMT.__mod (v1, a2)
   if Vector.isVector(a2) then error("Cannot modulus by a vector", 2) end
   local data = {}
   for i = 1, v1.N do
      data[i] = v1.data[i] % a2
   end
   local tp = types.type_meet(v1.type:baseType(), t.float)
   return Vector.new(tp:terraType(), data)
end

function VectorMT.__unm (v1)
   if v1.type == bool then error("Cannot negate a non-numeric vector", 2) end
   local data = {}
   for i = 1, #v1.data do
      data[i] = -v1.data[i]
   end
end

function VectorMT.__eq (v1, v2)
   if v1.N ~= v2.N then return false end
   for i = 1, v1.N do
      if v1.data[i] ~= v2.data[i] then return false end
   end
   return true
end


-------------------------------------------------
--[[ TopoSet methods                         ]]--
-------------------------------------------------
local size_fn = {
   [Cell]   = runtime.numCells,
   [Face]   = runtime.numFaces,
   [Edge]   = runtime.numEdges,
   [Vertex] = runtime.numVertices
}

function TopoSet.new (mesh, topo_type)
   local size   = size_fn[topo_type](mesh.__ctx)
   local ts     = setmetatable({__mesh=mesh, __type=topo_type,__size=size}, TopoSetMT)
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
   if not kernel.isKernel                 then error("Cannot map over object of type " .. type(kernel))         end
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
   local field = setmetatable({mesh=self}, FieldMT)
   field:setType(topo_type, data_type)
   if not tutil.conformsToType(initial_val, field.type) then
      error("Initializer is not of type " .. field.type:toString(), 2)
   end
   local valtype, vallen = field.type:runtimeType()
   local eltype          = field.topo:runtimeType()
   field.__lfield  = runtime.initField(self.__ctx, eltype, valtype, vallen)
   field.__lkfield = runtime.getlkField(field.__lfield)

   -- Initialize field to initial_val
   -- simple type:
   local ttype = field.type:terraType()
   if vallen == 1 then
      local terra init_field () : {}
         var v : ttype = initial_val
         runtime.lFieldBroadcast([self.__ctx], [field.__lfield], eltype, valtype, vallen, &v)
      end
      init_field()

   -- Vector type: use vector codegen functionality!
   else
      local v = Vector.new(field.type:terraBaseType(), initial_val)
      local terra init_field () : {}
         var v : ttype  = [v:__codegen()]
         runtime.lFieldBroadcast([self.__ctx], [field.__lfield], eltype, valtype, vallen, &v)
      end         
      init_field()
   end
   return field
end

function Mesh:fieldWithLabel (topo_type, data_type, label)
   local field = setmetatable({mesh=self}, FieldMT)
   field:setType(topo_type,data_type)
   local valtype, vallen = field.type:runtimeType()
   local eltype          = field.topo:runtimeType()
   field.__lfield  = runtime.loadField    (self.__ctx, label, eltype, valtype, vallen)
   field.__lkfield = runtime.getlkField(field.__lfield)
   return field
end

function Mesh:scalar (data_type, init)
   if not tutil.isDataType(data_type) then error("First argument to mesh:scalar must be a Liszt-supported data type", 2) end

   local ltype = tutil.ltype(data_type)
   if not tutil.conformsToType(init, ltype) then error("Second argument to mesh:scalar must be an instance of the specified data type", 2) end

   local sc = setmetatable({type=tutil.ltype(data_type)}, ScalarMT)
   local sctype, sclen = sc.type:runtimeType()
   sc.__lscalar  = runtime.initScalar(self.__ctx, sctype, sclen)
   sc.__lkscalar = runtime.getlkScalar(sc.__lscalar)
   sc.__ctx      = self.__ctx
   sc:setTo(init)
   return sc
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
