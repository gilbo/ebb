

local dld = {}
package.loaded["compiler.dld"] = dld
dld.__index = dld


-- DATA LAYOUT DESCRIPTION

--[[

  {
    type: {
      vector_size: (number, 1 usually could be a tuple of 2,3,4 elements)
      base_type_str: (string expression of type)
      base_bytes: (number, # of bytes used for base type)
    }
    logical_size: (number, the count of the items in the array)
    address: (pointer, where to find the data)
    stride:  (number, how much space to expect between entries)
    offset:  (number, how many bytes to skip after address)
  }

]]--

function dld.new(params)
  local desc = setmetatable({}, dld)
  if params then desc:init(params) end
  return desc
end

function dld.is_dld(o)
  return getmetatable(o) == dld
end

function dld:matchType(rhs)
  local lhtyp = self.type
  local rhtyp = rhs.type

  return lhtyp.vector_size   == rhtyp.vector_size and
         lhtyp.base_type_str == rhtyp.base_type_str and
         lhtyp.base_bytes    == rhtyp.base_bytes
end

function dld:matchAll(rhs)
  return self:matchType(rhs) and
         self.logical_size == rhs.logical_size and
         self.address      == rhs.address and
         self.stride       == rhs.stride and
         self.offset       == rhs.offset
end

function dld:init(params)
  params = params or {}

  if params.type then
    if terralib.types.istype(params.type) then
      self:setTerraType(params.type, params.type_n)
    end
  end

  if params.logical_size then
    self:setLogicalSize(params.logical_size)
  end

  if params.data then
    self:setData(params.data)
  end

  if params.stride then
    self:setStride(params.stride)
  end

  if params.offset then
    self:setOffset(params.offset)
  end

  if params.compact then
    self:setCompactStrideOffset()
  end
end

function dld:setTerraType(typ, n)
  if not terralib.types.istype(typ) then
    error('setTerraType() only accepts Terra types as arguments', 2)
  end
  if not typ:isprimitive() then
    error('Can only use setTerraType on primitives', 2)
  end

  n = n or 1

  self.type = {
    vector_size   = n,
    base_type_str = tostring(typ),
    base_bytes    = terralib.sizeof(typ)
  }
end

function dld:setLogicalSize(n)
  self.logical_size = n
end

function dld:setData(ptr)
  self.address = ptr
end

function dld:setStride(stride)
  self.stride = stride
end

function dld:setOffset(offset)
  self.offset = offset
end

function dld:setCompactStrideOffset()
  self.stride = self.type.vector_size * self.type.base_bytes
  self.offset = 0
end

function dld:getPhysicalSize()
  return self.logical_size * self.stride
end


