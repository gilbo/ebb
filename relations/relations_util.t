local L = {}

local C = terralib.includecstring [[
    #include <stdlib.h>
    #include <string.h>
]]

--[[
- A table contains size, fields and _indexrelations, which point to tables
that are indexed by this table. Example, _indexrelations for vertices table
will point to vtov, vtoe etc.
- A table also contains _index that has the compressed row values for index
field, and the corresponding expanded index field and other field values.
- A field contains fieldname, type of field, pointer to its table and expanded
  data.
--]]

local table = {}
table.__index = table
function L.istable(t)
    return getmetatable(t) == table
end

local key = {}

function L.newtable(size, debugname)
    return setmetatable( {
        _size = size,
        _fields = terralib.newlist(),
        _indexrelations = {},
        _debugname = debugname or "anon"
        },
        table)
end

local field = {}

field.__index = field
function L.isfield(f)
    return getmetatable(f) == field
end

function L.newfield(t)
    return { type = t } 
end

function table:__newindex(fieldname,value)
    local typ = value.type --TODO better error checking
    local f = setmetatable({},field)
    rawset(self,fieldname,f)
    f.name = fieldname
    f.table = self
    f.type = typ
    f.realtype = L.istable(f.type) and uint32 or f.type
    self._fields:insert(f)
end 

function table:getrelation(relname)
    return self._indexrelations[relname]
end

function field:loadfrommemory(mem)
    assert(self.data == nil)
    local nbytes = self.table._size * terralib.sizeof(self.realtype)
    local bytes = C.malloc(nbytes)
    self.data = terralib.cast(&self.realtype,bytes)
    local memT = terralib.typeof(mem)
    assert(memT == &self.realtype)
    C.memcpy(self.data,mem,nbytes)
end

function field:loadalternatefrommemory(mem)
    assert(self.data == nil)
    local nelems = self.table._size
    local nbytes = nelems * terralib.sizeof(self.realtype)
    local bytes = C.malloc(nbytes)
    self.data = terralib.cast(&self.realtype,bytes)
    -- TODO: error checking here
    for i = 0, nelems-1 do
        self.data[i] = mem[2*i]
    end
end

function table:loadindexfrommemory(fieldname,row_idx)
    assert(self._index == nil)
    local f = self[fieldname]
    assert (f)
    assert(f.data == nil)
    assert(L.istable(f.type))
    f.type._indexrelations[fieldname] = f
    local realtypesize = terralib.sizeof(f.realtype)
    local nbytes = (f.type._size + 1)*realtypesize
    rawset(self, "_index", terralib.cast(&f.realtype,C.malloc(nbytes)))
    local memT = terralib.typeof(row_idx)
    assert(memT == &f.realtype)
    C.memcpy(self._index,row_idx,nbytes)
    f.data = terralib.cast(&f.realtype,C.malloc(self._size*realtypesize))
    for i = 0, f.type._size - 1 do
        local b = self._index[i]
        local e = self._index[i+1]
        for j = b, e - 1 do
            f.data[j] = i
        end
    end
end

function field:dump()
    print(self.name..":")
    if not self.data then
        print("...not initialized")
        return
    end
    local N = self.table._size
    for i = 0,N - 1 do
        print("",i, self.data[i])
    end
end

function table:dump()
    print(self._debugname, "size: "..self._size)
    for i,f in ipairs(self._fields) do
        f:dump()
    end
end

return L
