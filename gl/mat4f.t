

local mat4f = {}
package.loaded["gl.mat4f"] = mat4f
mat4f.__index = mat4f

local ffi = require 'ffi'


local function raw_mat()
  return setmetatable({
    col = ffi.new('float[16]')
  }, mat4f)
end

function mat4f.is_mat4f(o)
  return getmetatable(o) == mat4f
end


function mat4f:clone()
  local m = raw_mat()
  for k=0,15 do
    m.col[k] = self.col[k]
  end
  return m
end


local function set_entire_matrix(self, xs)
  if #xs ~= 16 then return nil end
  for i=0,3 do
    for j=0,3 do
      self.col[4*j + i] = xs[4*i + (j+1)]
    end
  end
end

function mat4f:set(i,j, val)
  if j then
    self.col[4*j + i] = val
  else
    set_entire_matrix(self, i) -- name overloading
  end
end

function mat4f:get(i,j)
  return self.col[4*j + i]
end

function mat4f.__tostring(self)
  local str='['
  for i = 0,3 do
    if i>0 then str = str..'; ' end
    for j = 0,3 do
      if j>0 then str = str..', ' end
      str = str..tostring(self:get(i,j))
    end
  end
  str = str..']'
  return str
end

function mat4f.zero()
  local m = raw_mat()
  m:set({
    0, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 0,
  })
  return m
end

function mat4f.id()
  local m = raw_mat()
  m:set({
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 1,
  })
  return m
end

function mat4f.rotz(rad)
  local sinr = math.sin(rad)
  local cosr = math.cos(rad)

  local m = raw_mat()
  m:set({
    cosr, -sinr, 0, 0,
    sinr,  cosr, 0, 0,
       0,     0, 1, 0,
       0,     0, 0, 1,
  })
  return m
end

function mat4f.roty(rad)
  local sinr = math.sin(rad)
  local cosr = math.cos(rad)

  local m = raw_mat()
  m:set({
     cosr, 0, sinr, 0,
        0, 1,    0, 0,
    -sinr, 0, cosr, 0,
        0, 0,    0, 1,
  })
  return m
end

function mat4f.rotx(rad)
  local sinr = math.sin(rad)
  local cosr = math.cos(rad)

  local m = raw_mat()
  m:set({
    1,    0,     0, 0,
    0, cosr, -sinr, 0,
    0, sinr,  cosr, 0,
    0,    0,     0, 1,
  })
  return m
end

function mat4f.view(ratio, scale)
  if not scale then scale = 1 end
  local inv_r = 1 / ratio
  local inv_s = 1 / scale

  local m = raw_mat()
  m:set({
    inv_r * inv_s,     0, 0, 0,
                0, inv_s, 0, 0,
                0,     0, 1, 0,
                0,     0, 0, 1,
  })
  return m
end

-- this is a frustum thing
-- We assume that the size of the plane at
-- z=1 is fixed
--function mat4f.proj(fov)
--  local m = raw_mat()
--  m:set({
--    1, 0,  0, 0,
--    0, 1,  0, 0,
--    0, 0,  0, 0,
--    0, 0, -1, 0,
--  })
--  return m
--end

function mat4f.ortho(ratio, scale, minz, maxz)
  local s = 1.0 / scale
  local range = maxz - minz

  local m = raw_mat()
  m:set({
    s / ratio,      0,         0,             0,
            0,      s,         0,             0,
            0,      0, 1 / range, -minz / range,
            0,      0,         0,             1
  })
  return m
end

function mat4f.perspective(fov, ratio, near, far)
  local f = 1.0 / math.tan(fov * (math.pi / 360.0))
  local denom = near - far

  local m = raw_mat()
  m:set({
    f / ratio,          0,                    0,                        0,
            0,          f,                    0,                        0,
            0,          0, (far + near) / denom, (2 * far * near) / denom,
            0,          0,                   -1,                        0,
  })
  return m
end




function mat4f.__unm(self)
  -- negate
  local m = self:clone()
  for k=0,15 do
    m.col[k] = -m.col[k]
  end
  return m
end

function mat4f.__add(self, rhs)
  if not mat4f.is_mat4f(rhs) then
    error('cannot add mat4f to a NOT mat4f', 2)
  end

  local m = self:clone()
  for k=0,15 do
    m.col[k] = m.col[k] + rhs.col[k]
  end
  return m
end

function mat4f.__sub(self, rhs)
  if not mat4f.is_mat4f(rhs) then
    error('cannot subtract a NOT mat4f from a mat4f', 2)
  end

  return self + (-rhs)
end


function mat4f.__mul(self, rhs)
  if type(rhs) == 'number' then
    -- multiply by a scalar
    local m = self:clone()
    for k=0, 15 do
      m.col[k] = m.col[k] * rhs
    end
    return m
  elseif not mat4f.is_mat4f(rhs) then
    error('cannot multiply a mat4f by a non-number, non-mat4f value', 2)
  end

  -- matrix multiplication...
  local m = mat4f.zero()
  for i=0,3 do
    for j=0,3 do
      for k=0,3 do
        m.col[4*j + i] = m.col[4*j + i] + self.col[4*k + i] * rhs.col[4*j + k]
      end
    end
  end
  return m
end











