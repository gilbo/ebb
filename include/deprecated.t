-- Privately import from types module 
local types = terralib.require('compiler/types')
local Type  = types.Type
local t     = types.t
local tutil = types.usertypes

-- Keep runtime module private
local _runtime = runtime
terralib.require('runtime/liszt')
local runtime = runtime
_G.runtime    = _runtime

-- Expose types
L = { }
L.int    = t.int
L.uint   = t.uint
L.float  = t.float
L.bool   = t.bool
L.vector = t.vector

L.cell   = t.cell
L.vertex = t.vertex
L.edge   = t.edge
L.face   = t.face

terralib.require('include/builtins')

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

Vector   = {kind=Type.kinds.vector}
VectorMT = {__index=Vector}

local topo_objs = {
   [Cell]   = t.cell,
   [Edge]   = t.edge,
   [Face]   = t.face,
   [Vertex] = t.vertex
}


-------------------------------------------------
--[[ Field methods                           ]]--
-------------------------------------------------
function Field:setType(topo_tp, data_tp)
   if not (Type.isLisztType(topo_tp) and topo_tp:isTopo()) then
      error("Field over unrecognized topological type", 3)
   end
   if not (Type.isLisztType(data_tp) and data_tp:isExpressionType()) then
      error("Field over unsupported data type", 3)
   end
   self.topo = topo_tp
   self.type = data_tp
end


-------------------------------------------------
--[[ Scalar methods                          ]]--
-------------------------------------------------
function Scalar:setTo(val)
   if not tutil.conformsToType(val, self.type) then error("Cannot set scalar to value of incorrect type", 2) end

   -- compute expression for val expression contents
   local v = self.type:isVector() and Vector.new(self.type:baseType(), val):__codegen() or `val
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
      return Vector.new(self.type:baseType(), data)
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
      local v = Vector.new(field.type:baseType(), initial_val)
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
   if not Type.isLisztType(data_type) or not data_type:isExpressionType() then error("First argument to mesh:scalar must be a Liszt-supported data type", 2) end
   if not tutil.conformsToType(init, data_type) then error("Second argument to mesh:scalar must be an instance of the specified data type", 2) end

   local sc = setmetatable({type=data_type}, ScalarMT)
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
