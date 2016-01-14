-- The MIT License (MIT)
-- 
-- Copyright (c) 2015 Stanford University.
-- All rights reserved.
-- 
-- Permission is hereby granted, free of charge, to any person obtaining a
-- copy of this software and associated documentation files (the "Software"),
-- to deal in the Software without restriction, including without limitation
-- the rights to use, copy, modify, merge, publish, distribute, sublicense,
-- and/or sell copies of the Software, and to permit persons to whom the
-- Software is furnished to do so, subject to the following conditions:
-- 
-- The above copyright notice and this permission notice shall be included
-- in all copies or substantial portions of the Software.
-- 
-- THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
-- IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
-- FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
-- AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
-- LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
-- FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
-- DEALINGS IN THE SOFTWARE.
local exports = {}
package.loaded["ebb.lib.dld"] = exports

local bit = require 'bit'

-- DATA LAYOUT DESCRIPTION

--[[

  First, we describe the corresponding C-type.
  Since that will be the most universally accepted form
  of the DLD metadata object, it needs to be absolutely clear and
  precise on its own without appealing to any other encoding.
  
  struct DLD {
    uint8_t         version[2];             /* This is version 1,0 */
    uint16_t        base_type;              /* enumeration / flags */
    uint8_t         location;               /* enumeration */
    uint8_t         type_stride;            /* in bytes */
    uint8_t         type_dims[2];           /* 1,1 scalar; n,1 vector; ... */

    uint64_t        address;                /* void* */
    uint64_t        dim_size[3];            /* # elements in each dimension */
    uint64_t        dim_stride[3];          /* 1 = 1 element, not 1 byte */
  };

  Since this is immediately encodable as a Terra struct, let's do that.

--]]

exports.version = {1,0}

-- Total Size: 64B
local struct DLD {
  version           : uint8[2];           -- 1,0
  base_type         : uint16;             -- enumeration via flags
  location          : uint8;              -- enumeration
  type_stride       : uint8;              -- skip within type
  type_dims         : uint8[2];           -- 1,1 scalar; n,1 vector; ...

  address           : uint64;             -- i.e. &opaque
  dim_size          : uint64[3];          -- # elements in each dimension
  dim_stride        : uint64[3];          -- 1 = 1 element, not 1 byte
}
DLD:complete()
exports.C_DLD = DLD

local DLD_ENUMS = {}

--[[

  Most of these entries are immediately interpretable, but a few aren't.
  In particular, 'location' and 'base_type' are both enumerations, which
  means we need to specify what those enumeration values are.

  The version bits are intended to provide a way for future updates to
  this specification to clearly signal that they are using a different
  specification.  In particular, if the first byte is different than
  the recipient expects, or if the later byte is greater than the
  recipient expects, then the recipient should not expect to be able to
  correctly read or write the data.

  The Location enumeration is very simple right now. There are two values:

  CPU = 0;
  GPU = 1;

--]]

DLD_ENUMS.CPU     = 0
DLD_ENUMS.GPU     = 1

--[[

  The Base Type enumeration is considerably more complicated.
  We want to be able to encode signed and unsigned integers, floating-point
  values, tightly-packed key data and bits.  We use a system of flags
  that can be easily encoded in a 16-bit value to represent these options.

  |_ _ _ _|_|_|_|_|   |_ _|_ _|_ _|_ _|
      ^    ^ ^ ^ ^      ^   ^   ^   ^
      |    | | | |      |   |   |   |
      |    | | | |      |   |   |    \__ last   dimension # bits (8,16,32,64)
      |    | | | |      |   |   \_______ middle dimension # bits
      |    | | | |      |    \__________ first  dimension # bits
      |    | | | |       \______________ # key dimensions (0 == not key)
      |    | | | |
      |    | | |  \__ if set, this is a signed integer
      |    | |  \____ if set, this is data is tightly packed bits
      |    |  \______ if set, this is a 32-bit single-precision float
      |     \________ if set, this is a 64-bit double-precision float
       \_____________ unused bits

  The basic type enumerations and some example keys

  UINT_8        = 0x0
  UINT_16       = 0x1
  UINT_32       = 0x2
  UINT_64       = 0x3

  SINT_8        = 0x100
  SINT_16       = 0x101
  SINT_32       = 0x102
  SINT_64       = 0x103

  BIT           = 0x200
  FLOAT         = 0x400
  DOUBLE        = 0x800

  KEY_32        = 0x42   ( 01 00 00 10 )
  KEY_16_32_64  = 0xDB   ( 11 01 10 11 )
  KEY_64_32     = 0x8E   ( 10 00 11 10 )

  Because it's possible to construct types that have non-power-of-2 sizes,
  we include a type_stride parameter that describes how these values
  are aligned within a vector or matrix structured element.

  SPECIAL CASE:  BIT
  If the BIT type is used, then the type_dim[] and type_stride can be
  ignored.  Whichever dimension the bits are tightly packed along should
  have dim_stride[dim] set to ** 0 ** to indicate tight packing.

--]]

DLD_ENUMS.UINT_8          = 0x0
DLD_ENUMS.UINT_16         = 0x1
DLD_ENUMS.UINT_32         = 0x2
DLD_ENUMS.UINT_64         = 0x3

DLD_ENUMS.SINT_8          = 0x100
DLD_ENUMS.SINT_16         = 0x101
DLD_ENUMS.SINT_32         = 0x102
DLD_ENUMS.SINT_64         = 0x103

DLD_ENUMS.BIT             = 0x200
DLD_ENUMS.FLOAT           = 0x400
DLD_ENUMS.DOUBLE          = 0x800

-- programmatically construct shorthands for all the key types
for dim1 = 0,3 do
  local bits_1d     = dim1
  local suffix_1d   = '_'..tostring(math.pow(2,dim1+3))
  DLD_ENUMS['KEY'..suffix_1d] = bit.lshift(1,6) + bits_1d

  for dim2 = 0,3 do
    local bits_2d     = bit.lshift(dim2, 2) + bits_1d
    local suffix_2d   = '_'..tostring(math.pow(2,dim2+3))..suffix_1d
    DLD_ENUMS['KEY'..suffix_2d] = bit.lshift(2,6) + bits_2d

    for dim3 = 0,3 do
      local bits_3d     = bit.lshift(dim3, 4) + bits_2d
      local suffix_3d   = '_'..tostring(math.pow(2,dim3+3))..suffix_2d
      DLD_ENUMS['KEY'..suffix_3d] = bit.lshift(3,6) + bits_3d
    end
  end
end

-- For convenience stick all these enumerations onto the Terra struct table
-- itself:
for k,v in pairs(DLD_ENUMS) do
  DLD[k]      = v
  exports[k]  = v
end


-------------------------------------------------------------------------------
--[[  Terra Convenience Functions                                          ]]--
-------------------------------------------------------------------------------

terra DLD:init()
  self.version        = arrayof(uint8, 1,0)
  self.base_type      = DLD_ENUMS.DOUBLE
  self.location       = DLD_ENUMS.CPU
  self.type_stride    = 8 -- to a double
  self.type_dims      = arrayof(uint8, 1,1)

  self.address        = 0
  self.dim_size       = arrayof(uint64, 1,1,1)
  self.dim_stride     = arrayof(uint64, 1,1,1)
end

terra DLD:setlocation( enum : uint8 )
  self.location       = enum
end

terra DLD:settype( enum : uint16, stride : uint8, dim1 : uint8, dim2 : uint8 )
  self.base_type      = enum
  self.type_stride    = stride
  self.type_dims      = array(dim1, dim2)
end
terra DLD:settype( enum : uint16, stride : uint8, dim1 : uint8 )
  self.base_type      = enum
  self.type_stride    = stride
  self.type_dims      = array(dim1, 1)
end
terra DLD:settype( enum : uint16, stride : uint8 )
  self.base_type      = enum
  self.type_stride    = stride
  self.type_dims      = arrayof(uint8, 1, 1)
end

terra DLD:setsize( dim1 : uint64, dim2 : uint64, dim3 : uint64 )
  self.dim_size       = array( dim1, dim2, dim3 )
end
terra DLD:setsize( dim1 : uint64, dim2 : uint64 )
  self.dim_size       = array( dim1, dim2, 1 )
end
terra DLD:setsize( dim1 : uint64 )
  self.dim_size       = array( dim1, 1, 1 )
end

terra DLD:setstride( s1 : uint64, s2 : uint64, s3 : uint64 )
  self.dim_stride     = array( s1, s2, s3 )
end
terra DLD:setstride( s1 : uint64, s2 : uint64 )
  self.dim_stride     = array( s1, s2, 1 )
end
terra DLD:setstride( s1 : uint64 )
  self.dim_stride     = array( s1, 1, 1 )
end



terra DLD:num_dims() : uint
  if      self.dim_size[2] > 1 then return 3
  elseif  self.dim_size[1] > 1 then return 2
                               else return 1 end
end
terra DLD:num_type_dims() : uint
  if      self.type_dims[1] > 1 then return 2
  elseif  self.type_dims[0] > 1 then return 1
                                else return 0 end
end
terra DLD:total_size()
  return self.dim_size[0] * self.dim_size[1] * self.dim_size[2]
end



-------------------------------------------------------------------------------
-------------------------------------------------------------------------------
--[[  Lua Version of DLD                                                   ]]--
-------------------------------------------------------------------------------
-------------------------------------------------------------------------------

local Lua_DLD   = {}
Lua_DLD.__index = Lua_DLD
exports.Lua_DLD = Lua_DLD

local function is_lua_dld(obj) return getmetatable(obj) == Lua_DLD end
exports.is_lua_dld = is_lua_dld

local function is_c_dld(obj)
  return getmetatable(obj) == 'cdata' and terralib.typeof(obj) == DLD
end
exports.is_c_dld = is_c_dld

function exports.NewDLD(params)
  params = params or {}
  if params.base_type then assert(params.type_stride) end
  local dld = setmetatable({
    version     = {1,0},
    base_type   = params.base_type    or DLD_ENUMS.DOUBLE,
    location    = params.location     or DLD_ENUMS.CPU,
    type_stride = params.type_stride  or 8,
    type_dims   = params.type_dims    or {1,1},

    address     = params.address      or 0,
    dim_size    = params.dim_size     or {1,1,1},
    dim_stride  = params.dim_stride   or {1,1,1},
  }, Lua_DLD)
  return dld
end

function Lua_DLD:setlocation(loc)
  self.location   = loc
end
function Lua_DLD:settype( base_type, stride, dim1, dim2 )
  self.base_type      = assert(base_type)
  self.type_stride    = assert(stride)
  self.type_dims      = { dim1 or 1, dim2 or 1 }
end
function Lua_DLD:setsize( d1, d2, d3 )
  self.dim_size       = { d1 or 1, d2 or 1, d3 or 1 }
end
function Lua_DLD:setstride( s1, s2, s3 )
  self.dim_stride     = { s1 or 1, s2 or 1, s3 or 1 }
end

function Lua_DLD:num_dims()
  return (self.dim_size[3] > 1 and 3) or
         (self.dim_size[2] > 1 and 2) or 1
end
function Lua_DLD:num_type_dims()
  return (self.type_dims[2] > 1 and 2) or
         (self.type_dims[1] > 1 and 1) or 0
end
function Lua_DLD:total_size()
  return self.dim_size[1] * self.dim_size[2] * self.dim_size[3]
end


-------------------------------------------------------------------------------
--[[  Conversion between Terra and Lua versions                            ]]--
-------------------------------------------------------------------------------

function DLD:toLua()
  return exports.NewDLD {
    base_type       = tonumber(self.base_type),
    location        = tonumber(self.location),
    type_stride     = tonumber(self.type_stride),
    type_dims       = { tonumber(self.type_dims[0]),
                        tonumber(self.type_dims[1]), },

    address         = terralib.cast(&opaque, self.address),
    dim_size        = { tonumber(self.dim_size[0]),
                        tonumber(self.dim_size[1]),
                        tonumber(self.dim_size[2]), },
    dim_stride      = { tonumber(self.dim_stride[0]),
                        tonumber(self.dim_stride[1]),
                        tonumber(self.dim_stride[2]), },
  }
end

function Lua_DLD:toTerra()
  local dld = terralib.new(DLD)
  dld:init()
  dld:settype( self.base_type, self.type_stride,
               self.type_dims[1], self.type_dims[2] )
  dld:setlocation( self.location )

  dld.address = terralib.cast(uint64, self.address)
  dld:setsize( self.dim_size[1], self.dim_size[2], self.dim_size[3] )
  dld:setstride( self.dim_stride[1], self.dim_stride[2], self.dim_stride[3] )

  return dld
end

