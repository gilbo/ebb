



local VAObject = {}
VAObject.__index = VAObject

package.loaded["gl.vao"] = VAObject

local ffi = require 'ffi'
local gl  = terralib.require 'gl.gl'
local dld = terralib.require 'compiler.dld'



function VAObject.new()
    local p_vao = ffi.new 'unsigned int[1]'
    gl.glGenVertexArrays(1,p_vao)
    local vao_index = p_vao[0]

    local vao = setmetatable({
        vao_id    = vao_index,
        n_faces   = 0,
        attrs     = {},
    }, VAObject)

    vao:bind()
    return vao
end

function VAObject.bindnull()
    gl.glBindVertexArray(0)
end

function VAObject:bind()
    gl.glBindVertexArray(self.vao_id)
end

function VAObject:unbind()
    VAObject.bindnull()
end


function VAObject:setSize(n_faces)
    self.n_faces = n_faces
end
function VAObject:size()
    return self.n_faces
end

function VAObject:setData(sig_dld)
  local n_attrs = 0

  -- store data
  for name, blob in pairs(sig_dld) do
    if blob.dld.logical_size ~= self.n_faces * 3 then
      error('attribute "'..name..'" has an array of '..
            tostring(blob.dld.logical_size)..' elements, '..
            'but we were expecting '..tostring(self.n_faces*3)..
            ' ('..tostring(self.n_faces)..' * 3) elements.  '..
            'Did you remember to set the size of the VAO before '..
            'setting the data?', 2)
    end
    self.attrs[name] = {
      dld     = blob.dld,
      attr_id = blob.attr_id,
    }
    n_attrs = n_attrs + 1
  end

  -- generate buffers
  if n_attrs > 0 then
    self:bind()

    local p_buffers = ffi.new ('unsigned int['..tostring(n_attrs)..']')
    gl.glGenBuffers(n_attrs, p_buffers)

    local k = 0
    for _, data in pairs(self.attrs) do
        data.buf_id = p_buffers[k]
        k = k+1
    end
  end

  -- load data into GL
  for name, _ in pairs(self.attrs) do
    self:loadAttr(name)
  end
end


function VAObject:loadAttr(name)
  local blob = self.attrs[name]
  if not blob then return end

  local dld     = blob.dld
  local attr_id = blob.attr_id
  local buf_id  = blob.buf_id

  local gl_type = ({
    float  = gl.FLOAT,
    double = gl.DOUBLE,
  })[dld.type.base_type_str]
  if not gl_type then
    error('VAO doesn\'t currently support type '..dld.type.base_type_str)
  end

  local data_size = dld:getPhysicalSize()

  self:bind()

  gl.glEnableVertexAttribArray(attr_id)
  gl.glBindBuffer(gl.ARRAY_BUFFER, buf_id)
  gl.glBufferData(gl.ARRAY_BUFFER, dld:getPhysicalSize(), dld.address,
                  gl.STATIC_DRAW)
  -- inputs are:
  --  the attribute id
  --  the number of entries per element (for vectors)
  --  the type of data (base type for vectors)
  --  false means do not normalize vectors
  --  the stride, 0 means tightly packed
  --  the offset to start at
  gl.glVertexAttribPointer(attr_id,
                           dld.type.vector_size, gl_type,
                           false, dld.stride,
                           terralib.cast(&opaque, dld.offset))
  print(name)
end


return VAObject

