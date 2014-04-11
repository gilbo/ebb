local L = {}
package.loaded["compiler.lisztlib"] = L


-------------------------------------------------------------------------------
--[[ Liszt modules:                                                        ]]--
-------------------------------------------------------------------------------

function L.require( str )
    local loaded_module = terralib.require( str )
    return loaded_module
end

-------------------------------------------------------------------------------
--[[ Liszt object prototypes:                                              ]]--
-------------------------------------------------------------------------------
local function make_prototype(objname,name)
    local tb = {}
    tb.__index = tb
    L["is_"..name] = function(obj) return getmetatable(obj) == tb end
    L[objname] = tb
    return tb
end
local LRelation  = make_prototype("LRelation","relation")
local LField     = make_prototype("LField","field")
local LGlobal    = make_prototype("LGlobal","global")
local LVector    = make_prototype("LVector","vector")
local LMacro     = make_prototype("LMacro","macro")
local Kernel     = make_prototype("LKernel","kernel")

local C = terralib.require "compiler.c"
local T = terralib.require "compiler.types"
local ast = terralib.require "compiler.ast"
terralib.require "compiler.builtins"
local LDB = terralib.require "compiler.ldb"
local semant = terralib.require "compiler.semant"
local codegen = terralib.require "compiler.codegen"
--L.LDB = LDB

--[[
- An LRelation contains its size and fields as members.  The _index member
- refers to an array of the compressed row values for the index field.

- An LField stores its fieldname, type, an array of data, and a pointer
- to another LRelation if the field itself represents relational data.
--]]
local is_vector = L.is_vector --cache lookup for efficiency

-------------------------------------------------------------------------------
--[[ LGlobals:                                                             ]]--
-------------------------------------------------------------------------------
function L.NewGlobal (typ, init)
    if not T.isLisztType(typ) or not typ:isValueType() then error("First argument to L.NewGlobal must be a Liszt expression type", 2) end
    if not T.luaValConformsToType(init, typ) then error("Second argument to L.NewGlobal must be an instance of type " .. typ:toString(), 2) end

    local s  = setmetatable({type=typ}, LGlobal)
    local tt = typ:terraType()
    s.data   = terralib.cast(&tt, C.malloc(terralib.sizeof(tt)))
    s:setTo(init)
    return s
end


function LGlobal:setTo(val)
   if not T.luaValConformsToType(val, self.type) then error("value does not conform to type of global: " .. self.type:toString(), 2) end
      if self.type:isVector() then
          local v     = is_vector(val) and val or L.NewVector(self.type:baseType(), val)
          local sdata = terralib.cast(&self.type:terraBaseType(), self.data)
          for i = 0, v.N-1 do
              sdata[i] = v.data[i+1]
          end
    -- primitive is easy - just copy it over
    else
        self.data[0] = self.type == L.int and val - val % 1 or val
    end
end

function LGlobal:value()
    if self.type:isPrimitive() then return self.data[0] end

    local ndata = {}
    local sdata = terralib.cast(&self.type:terraBaseType(), self.data)
    for i = 1, self.type.N do ndata[i] = sdata[i-1] end

    return L.NewVector(self.type:baseType(), ndata)
end


-------------------------------------------------------------------------------
--[[ LVectors:                                                             ]]--
-------------------------------------------------------------------------------
function L.NewVector(dt, init)
    if not (T.isLisztType(dt) and dt:isPrimitive()) then
        error("First argument to L.NewVector() should "..
              "be a primitive Liszt type", 2)
    end
    if not is_vector(init) and #init == 0 then
        error("Second argument to L.NewVector should either be "..
              "an LVector or an array", 2)
    end
    local N = is_vector(init) and init.N or #init
    if not T.luaValConformsToType(init, L.vector(dt, N)) then
        error("Second argument to L.NewVector() does not "..
              "conform to specified type", 2)
    end

    local data = {}
    if is_vector(init) then init = init.data end
    for i = 1, N do
        -- convert to integer if necessary
        data[i] = dt == L.int and init[i] - init[i] % 1 or init[i] 
    end

    return setmetatable({N=N, type=L.vector(dt,N), data=data}, LVector)
end

function LVector.__add (v1, v2)
    if not is_vector(v1) or not is_vector(v2) then
        error("Cannot add non-vector type to vector", 2)
    elseif v1.N ~= v2.N then
        error("Cannot add vectors of differing lengths", 2)
    elseif v1.type == L.bool or v2.type == L.bool then
        error("Cannot add boolean vectors", 2)
    end

    local data = { }
    local tp = T.type_meet(v1.type:baseType(), v2.type:baseType())

    for i = 1, #v1.data do
        data[i] = v1.data[i] + v2.data[i]
    end
    return L.NewVector(tp, data)
end

function LVector.__sub (v1, v2)
    if not is_vector(v1) then
        error("Cannot subtract vector from non-vector type", 2)
    elseif not is_vector(v2) then
        error("Cannot subtract non-vector type from vector", 2)
    elseif v1.N ~= v2.N then
        error("Cannot subtract vectors of differing lengths", 2)
    elseif v1.type == bool or v2.type == bool then
        error("Cannot subtract boolean vectors", 2)
    end

    local data = { }
    local tp = T.type_meet(v1.type:baseType(), v2.type:baseType())

    for i = 1, #v1.data do
        data[i] = v1.data[i] - v2.data[i]
    end

    return L.NewVector(tp, data)
end

function LVector.__mul (a1, a2)
    if is_vector(a1) and is_vector(a2) then
        error("Cannot multiply two vectors", 2)
    end
    local v, a
    if is_vector(a1) then   v, a = a1, a2
    else                    v, a = a2, a1 end

    if     v.type:isLogical()  then
        error("Cannot multiply a non-numeric vector", 2)
    elseif type(a) ~= 'number' then
        error("Cannot multiply a vector by a non-numeric type", 2)
    end

    local tm = L.float
    if v.type == int and a % 1 == 0 then tm = L.int end

    local data = {}
    for i = 1, #v.data do
        data[i] = v.data[i] * a
    end
    return L.NewVector(tm, data)
end

function LVector.__div (v, a)
    if     is_vector(a)    then error("Cannot divide by a vector", 2)
    elseif v.type:isLogical()  then error("Cannot divide a non-numeric vector", 2)
    elseif type(a) ~= 'number' then error("Cannot divide a vector by a non-numeric type", 2)
    end

    local data = {}
    for i = 1, #v.data do
        data[i] = v.data[i] / a
    end
    return L.NewVector(L.float, data)
end

function LVector.__mod (v1, a2)
    if is_vector(a2) then error("Cannot modulus by a vector", 2) end
    local data = {}
    for i = 1, v1.N do
        data[i] = v1.data[i] % a2
    end
    local tp = T.type_meet(v1.type:baseType(), L.float)
    return L.NewVector(tp, data)
end

function LVector.__unm (v1)
    if v1.type:isLogical() then error("Cannot negate a non-numeric vector", 2) end
    local data = {}
    for i = 1, #v1.data do
        data[i] = -v1.data[i]
    end
    return L.NewVector(v1.type, data)
end

function LVector.__eq (v1, v2)
    if v1.N ~= v2.N then return false end
    for i = 1, v1.N do
        if v1.data[i] ~= v2.data[i] then return false end
    end
    return true
end


-------------------------------------------------------------------------------
--[[ LMacros:                                                              ]]--
-------------------------------------------------------------------------------
function L.NewMacro(generator)
    return setmetatable({genfunc=generator}, LMacro)    
end

L.Where = L.NewMacro(function(field,key)
    if field == nil or key == nil then
        error("Where expects 2 arguments")
    end
    local w = ast.Where:DeriveFrom(field)
    w.field = field
    w.key   = key
    local q = ast.QuoteExpr:DeriveFrom(field)
    q.exp   = semant.check({}, w)
    q.node_type = q.exp.node_type
    return q
end)


-------------------------------------------------------------------------------
--[[ Kernels/Runtimes                                                      ]]--
-------------------------------------------------------------------------------
local rtlib = terralib.require 'compiler/runtimes'

L.singleCore = rtlib.singleCore
L.gpu        = rtlib.gpu

Kernel.__call  = function (kobj, relation)
    if not relation then
        error("A kernel must be called on a relation.  "..
              "No relation specified.", 2)
    end
    local runtime = L.singleCore
    --if not runtime then runtime = L.singleCore end
    --if not rtlib.is_valid_runtime(runtime) then 
    --    error('Argument is not a valid runtime')
    --end
	if not kobj.__kernels[runtime] or
       not kobj.__kernels[runtime][relation]
    then
        kobj:generate(runtime, relation)
    end
	kobj.__kernels[runtime][relation]()
end

function L.NewKernel(kernel_ast, env)
    local new_kernel = setmetatable({
        __kernels={}
    }, Kernel)

    new_kernel.typed_ast = semant.check(env, kernel_ast)

	return new_kernel
end

function Kernel:generate (runtime, relation)
    -- Right now, we require that the relation match exactly
    local ast_relation = self.typed_ast.relation
    if ast_relation ~= relation then
        error('Kernels may only be called on relation they were typed with')
    end

    if not self.__kernels[runtime] then
        self.__kernels[runtime] = {}
    end
	if not self.__kernels[runtime][relation] then
		self.__kernels[runtime][relation] =
            codegen.codegen(runtime, self.typed_ast, relation)
	end
end


