



local VAObject = {}
VAObject.__index = VAObject

package.loaded["gl.vao"] = VAObject

local ffi = require 'ffi'
local gl  = terralib.require 'gl.gl'
--local DLD = terralib.require 'compiler.dld'



function VAObject.new()
    local p_vao = ffi.new 'unsigned int[1]'
    gl.glGenVertexArrays(1,p_vao)
    local vao_index = p_vao[0]

    local vao = setmetatable({
        vao_id      = vao_index,
        n_faces     = 0,
        n_vertices  = 0,
        index       = nil, -- if not nil, then use index
        attrs       = {},
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





function VAObject:setSize(args)
  if type(args.faces) ~= 'number' or
     type(args.vertices) ~= 'number'
  then
    error('setSize() must be supplied with the number of faces, '..
          'and number of vertices in this VAO')
  end

  self.n_faces = args.faces
  self.n_vertices = args.vertices
end

function VAObject:nFaces()
  return self.n_faces
end
function VAObject:nVertices()
  return self.n_vertices
end


function VAObject:draw()
  self:bind()
  if not self.index then
    -- Draw as unindexed
    gl.glDrawArrays(gl.TRIANGLES, 0, self.n_vertices)
  else
    -- Draw as indexed
    gl.glDrawElements(gl.TRIANGLES, self.n_faces*3, gl.UNSIGNED_INT, nil)
  end
end


function VAObject:setData(args)
  self:bind()

  if args.index and args.attrs then
    self:setAttributes(args.attrs)
    self:setIndex(args.index)
  else
    if self.n_vertices ~= self.n_faces * 3 then
      error('Cannot load indexless data unless there are exactly 3 times '..
            'more vertices than faces (i.e. triangles).  '..
            'This VAO currently expects '..tostring(self.n_faces)..
            ' faces, but '..tostring(self.n_vertices)..' vertices.')
    end

    self:setAttributes(args)
  end
end

function VAObject:setIndex(dld)
  if dld.type.base_type_str ~= tostring(uint) then
    error('can only use indices of type UNSIGNED INT in VAOs')
  end
  if dld.logical_size ~= 3 * self.n_faces then
    error('The index must have as many entries as 3 times '..
          'the number of faces (#faces = '..tostring(self.n_faces)..').  '..
          'Did you remember to set the size of the VAO before setting the '..
          'data?')
  end

  -- store data
  self.index = {
    dld = dld
  }

  -- generate buffer
  local p_buffer = ffi.new('unsigned int[1]')
  gl.glGenBuffers(1, p_buffer)
  self.index.buf_id = p_buffer[0]

  -- bind the index buffer
  gl.glBindBuffer(gl.ELEMENT_ARRAY_BUFFER, self.index.buf_id)
  gl.glBufferData(gl.ELEMENT_ARRAY_BUFFER, dld:getPhysicalSize(), dld.address,
                  gl.STATIC_DRAW)
end

function VAObject:setAttributes(attrs)
  local n_attrs = 0

  -- store data
  for name, blob in pairs(attrs) do
    if blob.dld.logical_size ~= self.n_vertices then
      error('attribute "'..name..'" has an array of '..
            tostring(blob.dld.logical_size)..' elements/vertices, '..
            'but we were expecting '..tostring(self.n_vertices)..
            ' elements/vertices.  '..
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

  -- unnecessary right now, since should only be called from setData
  --self:bind()

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
end


return VAObject

