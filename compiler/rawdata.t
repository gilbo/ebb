

-- file/module namespace table
local Raw = {}
package.loaded["compiler.rawdata"] = Raw


local C = require "compiler.c"
local G = require "compiler.gpu_util"


-------------------------------------------------------------------------------
--[[ DataArray methods                                                     ]]--
-------------------------------------------------------------------------------

local DataArray = {}
DataArray.__index = DataArray
Raw.DataArray = DataArray

function DataArray.New(params)
  params.processor = params.processor or L.default_processor
  params.size = params.size or 0
  if not params.type then error('must provide type') end

  local array = setmetatable({
    _size = params.size,
    _processor = params.processor,
    _type = params.type,
    _data = nil,
  }, DataArray)

  if array._size > 0 then array:allocate() end
  return array
end

function DataArray.Wrap(params)
  if not params.size then error('must provide size') end
  if not params.processor then error('must provide location') end
  if not params.type then error('must provide type') end
  if not params.data then error('must provide data to wrap') end

  local array = setmetatable({
    _size = params.size,
    _processor = params.processor,
    _type = params.type,
    _data = params.data,
  }, DataArray)
  return array
end


function DataArray:ptr()
  return self._data
end

-- State:
function DataArray:isallocated()    return self._data ~= nil     end
 -- These do not check allocation
function DataArray:size()           return self._size            end
function DataArray:location()       return self._processor       end



terra allocateAligned(alignment : uint64, size : uint64)
  var r : &opaque
  C.posix_memalign(&r,alignment,size)
  return r
end
-- vector(double,4) requires 32-byte alignment
-- note: it _is safe_ to free memory allocated this way with C.free
local function cpu_allocate(array)
  return terralib.cast(&array._type,
            allocateAligned(32, array._size * terralib.sizeof(array._type)))
end
local function cpu_free(data)
  C.free(data)
end
local function cpu_cpu_copy(dst, src, size)
  C.memcpy(dst, src, size)
end

local gpu_allocate = function(a, b)
  error('gpu not supported')
end
local gpu_free = gpu_allocate
local cpu_gpu_copy = gpu_allocate
local gpu_cpu_copy = gpu_allocate
local gpu_gpu_copy = gpu_allocate

if terralib.cudacompile then
  gpu_allocate = function(array)
    return terralib.cast(&array._type,
                         G.malloc(array._size * terralib.sizeof(array._type)))
  end
  gpu_free      = G.free
  cpu_gpu_copy  = G.memcpy_cpu_from_gpu
  gpu_cpu_copy  = G.memcpy_gpu_from_cpu
  gpu_gpu_copy  = G.memcpy_gpu_from_gpu
end

-- idempotent
function DataArray:allocate()
  if self._data then return end
  if self._size == 0 then return end -- do not allocate if size 0

  if self._processor == L.CPU then
    self._data = cpu_allocate(self)
  elseif self._processor == L.GPU then
    self._data = gpu_allocate(self)
  else
    error('unrecognized processor')
  end
end

-- idempotent
function DataArray:free()
  if self._data then
    if self._processor == L.CPU then
      cpu_free(self._data)
    elseif self._processor == L.GPU then
      gpu_free(self._data)
    else
      error('unrecognized processor')
    end
    self._data = nil
  end
end

function DataArray:byteSize()
  return self:size() * sizeof(self._type)
end

-- copy as much data as possible given sizes
function DataArray:copy(src, size)
  local dst = self
  if not src._data or not dst._data then return end

  -- adjust size to a byte_count
  if not size then size = math.min(dst:size(), src:size()) end
  local byte_size = size * terralib.sizeof(src._type)

  if     dst._processor == L.CPU and src._processor == L.CPU then
    cpu_cpu_copy( dst:ptr(), src:ptr(), byte_size )
  elseif dst._processor == L.CPU and src._processor == L.GPU then
    cpu_gpu_copy( dst:ptr(), src:ptr(), byte_size )
  elseif dst._processor == L.GPU and src._processor == L.CPU then
    gpu_cpu_copy( dst:ptr(), src:ptr(), byte_size )
  elseif dst._processor == L.GPU and src._processor == L.GPU then
    gpu_gpu_copy( dst:ptr(), src:ptr(), byte_size )
  else
    error('unsupported processor')
  end
end

-- general purpose relocation/resizing of data
function DataArray:reallocate(params)
  -- If no change detected, and we have allocated data, then short-circuit
  if params.processor == self._processor and
     params.size == self._size and
     self._data
  then
    return
  end

  -- allocate new data
  local new_array = DataArray.New {
    processor = params.processor,
    size = params.size,
    type = self._type,
  }

  -- if we're already allocated, copy and self-destruct
  if self:isallocated() then
    new_array:copy(self)
    self:free()
  end

  -- write the new array values over into self
  self._processor  = new_array._processor
  self._size       = new_array._size
  self._data       = new_array._data
end

-- single dimension re-allocations
function DataArray:moveto(new_location)
  self:reallocate{ processor = new_location, size = self._size }
end
function DataArray:resize(new_size)
  self:reallocate{ processor = self._processor, size = new_size }
end

function DataArray:write_ptr(f)
  if self:location() == L.CPU then
    f(self:ptr())
  else
    local buf = DataArray.New {
      processor = L.CPU,
      size = self:size(),
      type = self._type
    }
    f(buf:ptr())
    self:copy(buf)
    buf:free()
  end
end
function DataArray:read_ptr(f)
  if self:location() == L.CPU then
    f(self:ptr())
  else
    local buf = DataArray.New {
      processor = L.CPU,
      size = self:size(),
      type = self._type
    }
    buf:copy(self)
    f(buf:ptr())
    buf:free()
  end
end
function DataArray:readwrite_ptr(f)
  local loc = self:location()
  self:moveto(L.CPU)
  f(self:ptr())
  self:moveto(loc)
end



-------------------------------------------------------------------------------
--[[ DynamicArray                                                          ]]--
-------------------------------------------------------------------------------





local DynamicArray = {}
DynamicArray.__index = DynamicArray
Raw.DynamicArray = DynamicArray

function DynamicArray.New(params)
  if not params.type then error('must provide type') end

  local data_array = DataArray.New(params)
  local dyn = setmetatable({
    _data_array = data_array,
    _used_size  = params.size, -- different than the backing size
  }, DynamicArray)
  return dyn
end

function DynamicArray.Wrap(params)
  if not params.size then error('must provide size') end
  if not params.processor then error('must provide location') end
  if not params.type then error('must provide type') end
  if not params.data then error('must provide data to wrap') end

  local da = DataArray.Wrap(params)
  local dyn = setmetatable({
    _data_array = da,
    _used_size  = params.size
  }, DynamicArray)
  return dyn
end

function DynamicArray:ptr()
  return self._data_array:ptr()
end

 -- These do not check allocation
function DynamicArray:size()       return self._used_size              end
function DynamicArray:location()   return self._data_array:location()  end

function DynamicArray:free()
  if not self._data_array then return end
  self._data_array:free()
  self._data_array = nil
  self._used_size = 0
end


-- (SHOULD THIS RESIZE THINGS? DOESN'T NOW)
function DynamicArray:copy(src)
  local dst  = self
  local size = math.min(dst:size(), src:size())
  dst._data_array:copy(src._data_array, size)
end


function DynamicArray:moveto(new_location)
  self._data_array:moveto(new_location)
end
function DynamicArray:resize(new_size)
  -- Try to short circuit most resize requests
  local old_actual_size = self._data_array:size()
  if new_size > old_actual_size then
    -- if we have to,
    -- increase capacity by at least doubling space (policy choice)
    local new_actual_size = math.max(new_size, old_actual_size*2)
    self._data_array:resize(new_actual_size)
  end

  self._used_size = new_size
end

function DynamicArray:write_ptr(f)
  self._data_array:write_ptr(f)
end
function DynamicArray:read_ptr(f)
  self._data_array:read_ptr(f)
end
function DynamicArray:readwrite_ptr(f)
  self._data_array:readwrite_ptr(f)
end


