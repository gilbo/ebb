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




local VO = {}
VO.__index = VO

package.loaded["ebb.gl.vo"] = VO

local ffi = require 'ffi'
local gl  = require 'ebb.gl.gl'
local DLD = require 'ebb.lib.dld'



local VOSignature = {}
VOSignature.__index = VOSignature

function VOSignature.new(args)
  local vosig = setmetatable({
    n_tri = nil,
    n_vert = nil,
    n_attr = 0,
    index_dld = nil,
    attr_dlds = {},
  }, VOSignature)

  -- Determine Attributes
  if not args.attrs then
    error('You must at least supply an "attrs" table')
  end
  vosig.attr_dlds = args.attrs
  for _, dld in pairs(args.attrs) do
    vosig.n_attr = vosig.n_attr + 1
    if not DLD.is_dld(dld) then
      error('the entries in the supplied attribute table must all be '..
            'DLD (Data Layout Description) objects.', 2)
    end
  end
  if vosig.n_attr == 0 then
    error('the supplied attribute table has no entries.', 2)
  end

  -- Determine Index if Present
  if args.index then
    if not DLD.is_dld(args.index) then
      error('supplied index must be a DLD (Data Layout Description)', 2)
    elseif args.index.type.base_type_str ~= tostring(uint) or
           args.index.type.vector_size ~= 1
    then
      error('supplied index must have type uint.', 2)
    end
    vosig.index_dld = args.index
  end

  -- Determine the # of Vertices
  for _, dld in pairs(args.attrs) do
    if not vosig.n_vert then
      vosig.n_vert = dld.logical_size
    elseif vosig.n_vert ~= dld.logical_size then
      error('attribute arrays do not all have matching sizes.', 2)
    end
  end

  -- Determine the # of Triangles
  if vosig.index_dld then
    if vosig.index_dld.logical_size % 3 ~= 0 then
      error('The index must always have a multiple of 3 # of entries.', 2)
    end
    vosig.n_tri = vosig.index_dld.logical_size / 3
  else
    if vosig.n_vert % 3 ~= 0 then
      error('Non-Indexed Data must contain a multiple of 3 # of vertices.', 2)
    end
    vosig.n_tri = vosig.n_vert / 3
  end

  return vosig
end

function VOSignature:nTriangles()
  return self.n_tri
end
function VOSignature:nVertices()
  return self.n_vert
end
function VOSignature:nAttributes()
  return self.n_attr
end
function VOSignature:hasIndex()
  return self.index_dld ~= nil
end

-- check whether the index has changed (not 100% reliable but pretty good)
function VOSignature:matchIndex(rhs)
  return (
    not self.index_dld and not rhs.index_dld
  ) or (
    self.index_dld and rhs.index_dld and
    self.index_dld:matchAll(rhs.index_dld)
  )
end

function VOSignature:matchTypeOfAttrs(rhs)
  -- first check that every name in the rhs attributes is in the lhs
  for name, _ in pairs(rhs.attr_dlds) do
    if not self.attr_dlds[name] then return false end
  end

  -- then check whether each name on the left is present on the right,
  -- and has matching type signature
  for name, dld in pairs(self.attr_dlds) do
    if not rhs.attr_dlds[name] then return false end

    if not dld:matchType(rhs.attr_dlds[name]) then return false end
  end

  return true
end

-- Find all the attributes of the lhs that don't match in terms of
-- data layout (usually we've checked types at this point...)
function VOSignature:unmatchedAttrs(rhs)
  local unmatched = {}

  for name, dld in pairs(self.attr_dlds) do
    if rhs.attr_dlds[name] and
       not dld:matchAll(rhs.attr_dlds[name])
    then 
      table.insert(unmatched, name)
    end
  end

  return unmatched
end




function VO.new()
    local p_vao = ffi.new 'unsigned int[1]'
    gl.glGenVertexArrays(1,p_vao)
    local vao_index = p_vao[0]

    local vao = setmetatable({
        vao_id      = vao_index,
        vo_sig      = nil,
    }, VO)

    vao:bind()
    return vao
end

function VO.bindnull()
    gl.glBindVertexArray(0)
end
function VO:bind()
    gl.glBindVertexArray(self.vao_id)
end
function VO:unbind()
    VO.bindnull()
end

function VO:nTriangles()
  if self.vo_sig then return self.vo_sig:nTriangles() end
end
function VO:nVertices()
  if self.vo_sig then return self.vo_sig:nVertices() end
end

function VO:draw()
  if not self.vo_sig then
    error('The Vertex Object cannot be drawn before being initialized.')
  end

  self:bind()
  if not self.vo_sig:hasIndex() then
    -- Draw as unindexed
    gl.glDrawArrays(gl.TRIANGLES, 0, self:nVertices())
  else
    -- Draw as indexed
    gl.glDrawElements(gl.TRIANGLES, self:nTriangles()*3, gl.UNSIGNED_INT, nil)
  end
end


function VO:initData(args)
  if self.vo_sig then
    error('Cannot initialize already initialized Vertex Object')
  end

  self.vo_sig = VOSignature.new(args)
  self.attr_ids = args.attr_ids

  -- check for consistent name sets
  local attr_dlds = self.vo_sig.attr_dlds
  local attr_ids  = args.attr_ids
  for name, _ in pairs(attr_dlds) do
    if not attr_ids[name] then
      error('Could not find attr_dlds entry "'..name..'" in attr_ids')
    end
  end
  for name, _ in pairs(attr_ids) do
    if not attr_dlds[name] then
      error('Could not find attr_ids entry "'..name..'" in attr_dlds')
    end
  end

  if self.vo_sig:hasIndex() then
    self:initIndex()
  end

  self:initAttributes()
end


function VO:updateData(args)
  if not self.vo_sig then
    error('Cannot update an uninitialized Vertex Object')
  end

  local new_sig = VOSignature.new(args)

  -- We ignore attribute ids on updates

  -- check to make sure we're sufficiently consistent
  -- to perform an update
  if not self.vo_sig:matchIndex(new_sig) then
    error('Update data\'s index and Existing data\'s index do not match')
  end
  if not self.vo_sig:matchTypeOfAttrs(new_sig) then
    error("Update data attributes' types do not match "..
          "existing data attributes' types")
  end

  self.vo_sig = new_sig
  self:updateAttributes()
end


function VO:initIndex()
  if self.index_buf_id then
    error('already have an index buffer.')
  end

  local dld = self.vo_sig.index_dld

  if dld.type.base_type_str ~= tostring(uint) then
    error('IMPOSSIBLE: can only use indices of type UNSIGNED INT in VOs')
  end
  if dld.logical_size ~= 3 * self:nTriangles() then
    error('IMPOSSIBLE: The index must have as many entries as 3 '..
          'times the number '..
          'of tris (#tris = '..tostring(self:nTriangles())..').')
  end

  -- generate buffer
  local p_buffer = ffi.new('unsigned int[1]')
  gl.glGenBuffers(1, p_buffer)
  self.index_buf_id = p_buffer[0] -- save the buffer id...

  -- bind the index buffer
  gl.glBindBuffer(gl.ELEMENT_ARRAY_BUFFER, self.index_buf_id)
  gl.glBufferData(gl.ELEMENT_ARRAY_BUFFER, dld:getPhysicalSize(), dld.address,
                  gl.STATIC_DRAW)
end

function VO:initAttributes(attr_ids)
  if self.attr_buf_ids then
    error('already have attribute buffers.')
  end

  self.attr_buf_ids = {}

  -- generate buffers
  local n_attrs = self.vo_sig:nAttributes()
  self:bind()

  local p_buffers = ffi.new ('unsigned int['..tostring(n_attrs)..']')
  gl.glGenBuffers(n_attrs, p_buffers)

  local k = 0
  for name, _ in pairs(self.attr_ids) do
    self.attr_buf_ids[name] = p_buffers[k]
    k = k + 1
  end

  -- load data into GL
  for name, _ in pairs(self.attr_ids) do
    self:initLoadAttr(name)
  end
end

function VO:updateAttributes()
  if not self.attr_buf_ids then
    error('Cannot update uninitialized vertex attributes.')
  end

  -- load data into GL
  for name, _ in pairs(self.attr_ids) do
    self:updateLoadAttr(name)
  end
end

function VO:initLoadAttr(name)
  if not self.attr_ids[name] then
    error('Do not have attr entry for name "'..name..'"')
  end

  local attr_id = self.attr_ids[name]
  gl.glEnableVertexAttribArray(attr_id)

  self:updateLoadAttr(name)
end

function VO:updateLoadAttr(name)
  if not self.attr_ids[name] then
    error('Do not have attr entry for name "'..name..'"')
  end

  local dld     = self.vo_sig.attr_dlds[name]
  local attr_id = self.attr_ids[name]
  local buf_id  = self.attr_buf_ids[name]

  gl.glBindBuffer(gl.ARRAY_BUFFER, buf_id)
  gl.glBufferData(gl.ARRAY_BUFFER, dld:getPhysicalSize(), dld.address,
                  gl.STATIC_DRAW)

  local gl_type = ({
    float  = gl.FLOAT,
    double = gl.DOUBLE,
  })[dld.type.base_type_str]
  if not gl_type then
    error('VAO doesn\'t currently support type '..dld.type.base_type_str)
  end

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


return VO

