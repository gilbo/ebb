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

-- Adding this module will patch in some functionality to the relations


local GClef_Module = {}
package.loaded["ebb.gl.gclef"] = GClef_Module

local L = require "ebblib"
local C = require "ebb.src.c"
local DLD = require "ebb.lib.dld"

local VO = require "ebb.gl.vo"


-- COPIED FROM ldb.t
terra allocateAligned(alignment : uint64, size : uint64)
    var r : &opaque
    C.posix_memalign(&r,alignment,size)
    return r
end
local function MallocArray(T,N)
    return terralib.cast(&T,allocateAligned(32,N * terralib.sizeof(T)))
end



local GClef = {}
GClef.__index = GClef

function GClef.new()
    local gclef = setmetatable({
        tri_index   = nil,
        vo          = nil,
        attrs       = {},
    }, GClef)
    return gclef
end

local terra tri_index_build(
    n_tris : uint,
    index  : &uint,
    v0 : &uint64, v1 : &uint64, v2 : &uint64
)
    for i = 0,n_tris do
        index[3*i + 0] = v0[i]
        index[3*i + 1] = v1[i]
        index[3*i + 2] = v2[i]
    end
end

function L.LRelation:CreateGClef(args)
    if self._gclef then
        error("CreateGClef(): Relation already has a GClef")
    end
    args = args or {}

    if not args.triangle or #(args.triangle) ~= 3 then
        error("CreateGClef(): Must supply 'triangle' list in argument, "..
              "specifying three fields of Row type, "..
              "referencing the same relation.")
    end
    if not args.attr_ids then
        error("CreateGClef(): Must supply argument 'attrs', a table "..
              "mapping fields of the vertex relation (to be used as "..
              "opengl vertex attributes) to attribute ids, which will "..
              "be bound to the shader program inputs.")
    end

    -- get the requested triangle fields
    local trif = {}
    local vertex_rel = nil
    for i=1,3 do
        local name = args.triangle[i]
        local f = self[name]
        if not f or not L.is_field(f) then
            error("CreateGClef(): Triangle field '"..name.."' not found.")
        end
        if not f.type:isRow() then
            error("CreateGClef(): Triangle field '"..name.."' not Row Type.")
        end
        table.insert(trif, f)
    end
    vertex_rel = trif[1].type.relation
    if trif[2].type.relation ~= vertex_rel or
       trif[3].type.relation ~= vertex_rel
    then
        error("CreateGClef(): Triangle fields "..
              "do not all refer to the same relation")
    end

    -- get the requested vertex fields
    local attrf = {}
    for attr, _ in pairs(args.attr_ids) do
        local f = vertex_rel[attr]
        if not f or not L.is_field(f) then
            error("CreateGClef(): Vertex field / attribute "..
                  "'"..attr"' not found.")
        end
        attrf[attr] = f
    end


    local clef  = GClef.new()
    rawset(self, '_gclef', clef)

    local n_tri  = self:Size()
    local n_vert = vertex_rel:Size()

    -- build the tri index
    clef.tri_index = {
        data        = MallocArray(uint, n_tri*3)
    }
    tri_index_build(n_tri, clef.tri_index.data,
        trif[1]:DataPtr(), trif[2]:DataPtr(), trif[3]:DataPtr())
    clef.tri_index.dld = DLD.new({
        location        = "CPU"
        type            = uint,
        logical_size    = n_tri * 3,
        data            = clef.tri_index.data,
        compact         = true
    })

    -- record the attributes
    local attr_dlds = {}
    for attr, id in pairs(args.attr_ids) do
        local f     = attrf[attr]
        local dld   = f:getDLD()
        attr_dlds[attr] = dld
        clef.attrs[attr] = {
            field   = f,
            id      = id,
            dld     = dld
        }
        print(dld.stride)
    end

    -- allocate the vertex object and initialize it
    clef.vo = VO.new()
    clef.vo:initData({
        index = clef.tri_index.dld,
        attrs = attr_dlds,
        attr_ids = args.attr_ids
    })
end


function L.LRelation:UpdateGClef()
    if not self._gclef then
        error('UpdateGClef(): This Relation Does not have a GClef.')
    end

    local attr_dlds = {}
    for attr, blob in pairs(self._gclef.attrs) do
        attr_dlds[attr] = blob.dld
    end
    self._gclef.vo:updateData({
        index = self._gclef.tri_index.dld,
        attrs = attr_dlds
    })
end


function L.LRelation:DrawGClef()
    if not self._gclef then
        error('DrawGClef(): This Relation Does not have a GClef.')
    end

    self._gclef.vo:draw()
end


function L.LRelation:GetGClefVO()
  if not self._gclef then
    error('GetGClefVO(): This Relation Does not have a GClef.')
  end
  return self._gclef.vo
end


