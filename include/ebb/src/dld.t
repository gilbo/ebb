local dld = {}
package.loaded["ebb.src.dld"] = dld
dld.__index = dld


-- DATA LAYOUT DESCRIPTION

--[[

  {
    location*    : string, which memory/processor the data is on
                   [ not included in ctype ]  -- can include this in ctype as a boolean flag if needed
    address      : pointer, where to find data
    type         : {
      base_type* : string indicating type of primitive that builds field type
                   [ not included in ctype ]
      base_bytes : number of bytes for primitive that builds field type
      size_bytes : number of bytes fir tge entire entry
      ndims      : 0, 1 or 2, indicating field type is scalar, vector or matrix
      dims       : list of number of primitives in each dimension (1 for scalar, vec len for vector, matrix dim for matrices)
                   [ array of numbers for ctype ]
      stride     : list of numbers indicating layout for one field element (for vector/ matrix)
                   [ array of numbers for ctype ]
    }
    ndims        : 1, 2 or 3 indicating underlying unstructured relation/ 1D grid, 2D or 3D grid
    dims         : list of number of elements along each dimension
                   [ array of numbers for ctype ]
    stride       : number, how much space to expect between field entries
                   [ array of numbers for ctype ]
    offset       : number, how many bytes to skip after address
  }

]]--

-- TODO(Chinmayee): Change dims to bounds with lower and upper bound when
-- support for distributing Ebb programs (partitioning) is added.


-------------------------------------------------------------------------------
-------------------------------------------------------------------------------
--[[  DLD initalizers and setters                                          ]]--
-------------------------------------------------------------------------------
-------------------------------------------------------------------------------


function dld.new(params)
  local desc = setmetatable({}, dld)
  if params then desc:init(params) end
  return desc
end

function dld.is_dld(o)
  return getmetatable(o) == dld
end

function dld:init(params)
  params = params or {}

  if params.location then
    self:SetLocation(params.location)
  end

  if params.address then
    self:SetDataPointer(params.address)
  end

  if params.type then
    self:SetType(params.type)
  end

  if params.dims then
    self:SetDims(params.dims)
  end

  if params.stride then
    self:SetStride(params.stride)
  end

  if params.offset then
    self:SetOffset(params.offset)
  end

  if params.compact then
    self:SetCompactStrideOffset()
  end
end

function dld:SetLocation(loc)
  self.location = loc
end

function dld:SetDataPointer(addr)
  self.address = addr
end

function dld:SetType(type)
  self.type = {}
  local ttype      = type:terraType()
  local base_type  = type:terraBaseType()
  self.type.base_type  = tostring(base_type)
  local base_bytes = terralib.sizeof(base_type)
  self.type.base_bytes = base_bytes
  self.type.size_bytes = terralib.sizeof(ttype)
  if type:isPrimitive() then
    self.type.ndims  = 0
    self.type.dims   = nil
    self.type.stride = nil
  elseif type:isVector() then
    self.type.ndims  = 1
    self.type.dims   = { type.N }
    self.type.stride = { base_bytes }
  elseif type:isMatrix() then
    self.type.ndims  = 2
    self.type.dims   = { type.Nrow, type.Ncol }
    self.type.size_bytes = base_bytes * type.Ncol * type.Nrow
    self.type.stride = { base_bytes * type.Ncol, base_bytes }  -- row major
  elseif type:isKey() then
    self.type.ndims  = 1
    self.type.dims   = { type.ndims }
    self.type.stride = { base_bytes }
  else
    error('DLD requested for unknown field type')
  end
end

function dld:SetDims(dims)
  self.ndims = #dims
  self.dims = {}
  for i,n in ipairs(dims) do
    self.dims[i] = n
  end
end

function dld:SetStride(stride)
  self.stride = stride
end

function dld:SetOffset(offset)
  self.offset = offset
end

function dld:SetCompactStrideOffset()
  -- default is column major, must set stride explicitly otherwise
  local size_entry = self.type.size_bytes
  if self.ndims == 1 then
    self.stride = { size_entry }
  elseif self.ndims == 2 then
    self.stride = { size_entry, size_entry * self.dims[1] }
  elseif self.ndims == 3 then
    self.stride = { size_entry,
                    size_entry * self.dims[1],
                    size_entry * self.dims[1] * self.dims[2] }
  end
  self.offset = 0
end


-------------------------------------------------------------------------------
-------------------------------------------------------------------------------
--[[  DLD methods and accessors                                            ]]--
-------------------------------------------------------------------------------
-------------------------------------------------------------------------------


function dld:MatchType(rhs)
  local lhtyp = self.type
  local rhtyp = rhs.type

  local match_dims = (lhtyp.ndims == rhtyp.ndims)
  if match_dims then
    for i = 1, lhtyp.ndims do
      match_dims = match_dims and (lhtyp.dims[i] == rhtyp.dims[i]) and
                                  (lhtyp.stride[i]  == rhtyp.stride[i] )
    end
  end

  return lhtyp.base_type   == rhtyp.base_type and match_dims
end

function dld:MatchAll(rhs)
  local match_dims = (self.ndims == rhs.ndims)
  if match_dims then
    for i = 1, self.ndims do
      match_dims = match_dims and (self.dims[i] == rhs.dims[i]) and
                                  (self.stride[i]  == rhs.stride[i] )
    end
  end
  return match_dims and self:MatchType(rhs) and
         self.offset == rhs.offset
end

function dld:GetPhysicalSize()
  local total_size = 0
  for i = 1, self.ndims do
    local s = self.dims[i] * self.stride[i]
    total_size = (s > total_size) and s or total_size
  end
  return total_size
end

function dld:IsComplete()
  return (self.address and self.type and self.ndims and self.dims and
          self.stride or self.offset)
end


-------------------------------------------------------------------------------
-------------------------------------------------------------------------------
--[[  DLD C API                                                            ]]--
-------------------------------------------------------------------------------
-------------------------------------------------------------------------------


struct dld.ctype {
  address  : &opaque;
  type     : struct {
    base_bytes : uint8;
    size_bytes : uint8;
    ndims      : uint8;
    dims       : uint8[2];
    stride     : uint8[2];
  };
  ndims    : uint8;
  dims     : uint64[3];
  stride   : uint64[3];
  offset   : uint64;
}

function dld:Compile()
  if not self:IsComplete() then
    error('Cannot compile incomplete dld', 2)
  end
  local typ_dims = {1, 1}
  local typ_stride  = {0, 0}
  for i = 1, self.type.ndims do
    typ_dims[i]   = self.type.dims[i]
    typ_stride[i] = self.type.stride[i]
  end
  local dims = {1, 1, 1}
  local stride  = {0, 0, 0}
  for i = 1, self.ndims do
    dims[i] = self.dims[i]
    stride[i]  = self.stride[i]
  end
  local d = terralib.new(dld.ctype,
                         { address  = self.address,
                           type     = { base_bytes = self.type.base_bytes,
                                        size_bytes = self.type.size_bytes,
                                        ndims      = self.type.ndims,
                                        dims       = typ_dims,
                                        stride     = typ_stride
                                      },
                           ndims    = self.ndims,
                           dims     = dims,
                           stride   = stride,
                           offset   = self.offset
                         })
  return d
end
