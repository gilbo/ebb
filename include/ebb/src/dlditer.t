

-- file/module namespace table
local Exports = {}
package.loaded["ebb.src.dlditer"] = Exports


local C     = require "ebb.src.c"
local DLD   = require "ebb.src.dld"
--local G = require "ebb.src.gpu_util"


-------------------------------------------------------------------------------
--[[ Lua Iterator                                                          ]]--
-------------------------------------------------------------------------------


local function luaiter(dld, clbk)
  --local elemsize = dld.type_stride * dld.type_dims[1] * dld.type_dims[2]
  local Nx,Ny,Nz = unpack(dld.dim_size)
  local Sx,Sy,Sz = unpack(dld.dim_stride)

  if Nz == 1 then
    if Ny == 1 then -- 1d
      for i=0,Nx-1 do
        local lin = i*Sx
        clbk(lin, i)
      end
    else -- 2d
      for j=0, Ny-1 do
        for i=0,Nx-1 do
          local lin = i*Sx + j*Sy
          clbk(lin, i, j)
      end end
    end
  else -- 3d
    for k=0, Nz-1 do
      for j=0, Ny-1 do
        for i=0,Nx-1 do
          local lin = i*Sx + j*Sy + k*Sz
          clbk(lin, i, j, k)
    end end end
  end
end
Exports.luaiter = luaiter

local function printlua(dld)
  assert(DLD.is_lua_dld(dld), 'expecting Lua DLD')
  print('version:     ', dld.version[1],    dld.version[2])
  print('base_type:   ', dld.base_type      )
  print('location:    ', dld.location       )
  print('type_stride: ', dld.type_stride    )
  print('type_dims:   ', dld.type_dims[1],  dld.type_dims[2])
  print('address:     ', dld.address        )
  print('dim_size:    ', dld.dim_size[1],   dld.dim_size[2],
                                            dld.dim_size[3])
  print('dim_stride:  ', dld.dim_stride[1], dld.dim_stride[2],
                                            dld.dim_stride[3])
end
Exports.printlua = printlua

local terra printterra(dld : &DLD.C_DLD)
  C.printf('version:     %d %d\n',  dld.version[0],    dld.version[1])
  C.printf('base_type:   %d\n',     dld.base_type      )
  C.printf('location:    %d\n',     dld.location       )
  C.printf('type_stride: %d\n',     dld.type_stride    )
  C.printf('type_dims:   %d %d\n',  dld.type_dims[0],  dld.type_dims[1])
  C.printf('address:     %d\n',     dld.address        )
  C.printf('dim_size:    %d %d %d\n',       dld.dim_size[0],
                         dld.dim_size[1],   dld.dim_size[2])
  C.printf('dim_stride:  %d %d %d\n',       dld.dim_stride[0],
                         dld.dim_stride[1], dld.dim_stride[2])
end
Exports.printterra = printterra

--[[
local LuaIter = {}
LuaIter.__index = LuaIter
Exports.LuaIter = LuaIter

local function NewLuaIter(dld)
  if DLD.is_c_dld(dld) then dld = dld:toLua() end
  assert(DLD.is_lua_dld(dld), 'argument must be dld')

  local iter = setmetatable({
    _dld = dld,
  }, LuaIter)
end
Exports.NewLuaIter = NewLuaIter


function LuaIter:forall()



end
--]]