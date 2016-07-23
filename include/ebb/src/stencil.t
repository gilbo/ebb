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
local Module = {}
package.loaded['ebb.src.stencil'] = Module

local use_exp    = not not rawget(_G, 'EBB_USE_EXPERIMENTAL_SIGNAL')
local use_single = not use_exp

local ast   = require "ebb.src.ast"
local T     = require "ebb.src.types"
local B     = require "ebb.src.builtins"
local Util  = require "ebb.src.util"

local NewRect2d, NewRect3d = Util.NewRect2d, Util.NewRect3d
local isrect2d,  isrect3d  = Util.isrect2d,  Util.isrect3d

------------------------------------------------------------------------------
--[[ Stencils                                                             ]]--
------------------------------------------------------------------------------

local Stencil2d = {}
Stencil2d.__index = Stencil2d
local Stencil3d = {}
Stencil3d.__index = Stencil3d
-- Stencil Any is used for non-grids
-- to encode the fact the accesses are indeterminate
local StencilAny = {}
StencilAny.__index = StencilAny

local function isstencil2d(obj) return getmetatable(obj) == Stencil2d end
local function isstencil3d(obj) return getmetatable(obj) == Stencil3d end
local function isstencilany(obj) return getmetatable(obj) == StencilAny end

local function NewStencil2d(params)
  params = params or {}
  local s = setmetatable({
    --_rects      = isrect2d(params.rect) and params.rect or {},
    _rect       = isrect2d(params.rect) and params.rect
                                        or NewRect2d({math.huge,-math.huge},
                                                     {math.huge,-math.huge})
  }, Stencil2d)
  return s
end
local function NewStencil3d(params)
  params = params or {}
  local s = setmetatable({
    --_rects      = isrect3d(params.rect) and params.rect or {},
    _rect       = isrect3d(params.rect) and params.rect
                                        or NewRect3d({math.huge,-math.huge},
                                                     {math.huge,-math.huge},
                                                     {math.huge,-math.huge})
  }, Stencil3d)
  return s
end
local function NewStencilAny()
  return setmetatable({},StencilAny)
end


local function all2dStencil()
  local s = NewStencil2d()
  s._is_all = true
  return s
end
local function all3dStencil()
  local s = NewStencil3d()
  s._is_all = true
  return s
end

local function unknown_stencil(typ)
  assert(typ:isscalarkey(),
         'unknown_stencil() only works for scalar keys')
  local rel = typ.relation
  if #rel:Dims() > 1 then 
    return (#rel:Dims() == 2) and all2dStencil() or all3dStencil()
  else
    return NewStencilAny()
  end
end

function Stencil2d:isall() return self._is_all end
function Stencil3d:isall() return self._is_all end
function StencilAny:isall() return true end

-- simplest implementation is to just concatenate rectangle lists
function Stencil2d:join(rhs)
  if self:isall() or rhs:isall() then return all2dStencil() end
  return NewStencil2d{ rect = self._rect:join(rhs._rect) }
  --local s = NewStencil2d()
  --s._rects = {unpack(self._rects)}
  --for _,r in ipairs(rhs._rects) do table.insert(s._rects, r) end
  --return s
end
function Stencil3d:join(rhs)
  if self:isall() or rhs:isall() then return all3dStencil() end
  return NewStencil3d{ rect = self._rect:join(rhs._rect) }
  --local s = NewStencil3d()
  --s._rects = {unpack(self._rects)}
  --for _,r in ipairs(rhs._rects) do table.insert(s._rects, r) end
  --return s
end
function StencilAny:join(rhs)
  return NewStencilAny()
end

function Stencil2d:envelopeRect()
  if self:isall() then return NewRect2d({-math.huge, math.huge},
                                        {-math.huge, math.huge}) end
  return self._rect
  --if #self._rects == 0 then return NewRect2d({math.huge, -math.huge},
  --                                           {math.huge, -math.huge}) end
  --local envelope = self._rects[1]
  --for i=2,#self._rects do
  --  local r = self._rects[i]
  --  envelope = envelope:join(r)
  --end
  --return r
end
function Stencil3d:envelopeRect()
  if self:isall() then return NewRect3d({-math.huge, math.huge},
                                        {-math.huge, math.huge},
                                        {-math.huge, math.huge}) end
  return self._rect
  --if #self._rects == 0 then return NewRect3d({math.huge, -math.huge},
  --                                           {math.huge, -math.huge},
  --                                           {math.huge, -math.huge}) end
  --local envelope = self._rects[1]
  --for i=2,#self._rects do
  --  local r = self._rects[i]
  --  envelope = envelope:join(r)
  --end
  --return r
end

-- The implementation may use different rounding semantics
-- We may want to tighten this analysis for that reason, but until we
-- do, rounding will guarantee safe stencil analysis for any rounding
-- semantics we choose when faced with non-integral affine-transform
-- coefficients.
local function round_interval(lo,hi)
  return { math.floor(lo), math.ceil(hi) }
end

function Stencil2d:affineTransform(A)
  local Nrow = #A
  if self:isall() then
    return Nrow == 2 and all2dStencil() or all3dStencil()
  end

  -- do this by taking the four corners of the rectangle and sending them
  -- through the transformation.  Then fit a box around that.

  -- we arrange the corners of the rectangle into a 4x2 matrix
  local xmin,xmax = self._rect:xminmax()
  local ymin,ymax = self._rect:yminmax()
  local P = { { xmin, ymin },
              { xmin, ymax },
              { xmax, ymin },
              { xmax, ymax } }

  -- then A * P^t (accounting for constant offset) will yield an OUT x 4
  -- dimensioned matrix.  We can collapse that to the min and max ranges
  -- in each dimension by taking the maximum or minimum over the point-dim.
  local outranges = {}
  for i=1,Nrow do
    local A_i = A[i]
    local outmin,outmax = math.huge, -math.huge
    for k=1,4 do -- # points
      local P_k = P[k]
      local sum = A_i[3]
      for j=1,2 do -- # columns in A
        sum = sum + A_i[j] * P_k[j]
      end
      outmin = math.min(outmin, sum)
      outmax = math.max(outmax, sum)
    end
    outranges[i] = round_interval(outmin,outmax)
  end

  -- return appropriate kind of stencil
  return Nrow == 2 and NewStencil2d{ rect = NewRect2d(unpack(outranges)) }
                    or NewStencil3d{ rect = NewRect3d(unpack(outranges)) }
end
function Stencil3d:affineTransform(A)
  local Nrow = #A
  if self:isall() then
    return Nrow == 2 and all2dStencil() or all3dStencil()
  end

  -- do this by taking the eight corners of the box and sending them
  -- through the transformation.  Then fit a box around that.

  -- we arrange the corners of the rectangle into a 8x3 matrix
  local xmin,xmax = self._rect:xminmax()
  local ymin,ymax = self._rect:yminmax()
  local zmin,zmax = self._rect:zminmax()
  local P = { { xmin, ymin, zmin },
              { xmin, ymin, zmax },
              { xmin, ymax, zmin },
              { xmin, ymax, zmax },
              { xmax, ymin, zmin },
              { xmax, ymin, zmax },
              { xmax, ymax, zmin },
              { xmax, ymax, zmax }, }

  -- then A * P^t (accounting for constant offset) will yield an OUT x 8
  -- dimensioned matrix.  We can collapse that to the min and max ranges
  -- in each dimension by taking the maximum or minimum over the point-dim.
  local outranges = {}
  for i=1,Nrow do
    local A_i = A[i]
    local outmin,outmax = math.huge, -math.huge
    for k=1,8 do -- # points
      local P_k = P[k]
      local sum = A_i[4]
      for j=1,3 do -- # columns in A
        sum = sum + A_i[j] * P_k[j]
      end
      outmin = math.min(outmin, sum)
      outmax = math.max(outmax, sum)
    end
    outranges[i] = round_interval(outmin,outmax)
  end

  -- return appropriate kind of stencil
  return Nrow == 2 and NewStencil2d{ rect = NewRect2d(unpack(outranges)) }
                    or NewStencil3d{ rect = NewRect3d(unpack(outranges)) }
end

-- crude approximation scheme 
--function Stencil2d:simplified()
--  return NewStencil2d{ rect = self:envelopeRect() }
--end
--function Stencil3d:simplified()
--  return NewStencil3d{ rect = self:envelopeRect() }
--end

------------------------------------------------------------------------------
--[[ Access Pattern                                                       ]]--
------------------------------------------------------------------------------
-- An access-pattern is supposed to subsume the previous concept of a phase /
-- phase-type.  A given access pattern captures how a given field is being
-- accessed, both in terms of read/write/reduce, whether the access is
-- considered to be "centered" or not, as well as a stencil to provide
-- the backend with more granular information about what data is being
-- accessed.

local AccessPattern   = {}
AccessPattern.__index = AccessPattern

local function isaccesspattern(obj)
  return getmetatable(obj) == AccessPattern
end

local function NewAccessPattern(params)
  local ap = setmetatable({
    _phase_write    = params.write,
    _phase_read     = params.read,
    _phase_reduce   = params.reduceop and true,
    _phase_reduceop = params.reduceop,
    _centered       = params.centered,

    _stencil        = params.stencil,

    _field          = params.field,
  }, AccessPattern)
  return ap
end

--[[
params {
 field,
 read,
 write
}
--]]
function Module.NewCenteredAccessPattern(params)
  local ndims = #params.field:Relation():Dims()
  local centered_stencil
  if ndims == 1 then  -- unstructured cases
    centered_stencil = NewStencilAny()
  end
  if ndims == 2 then
    centered_stencil = NewStencil2d{ rect = NewRect2d({0,0},{0,0}) }
  else
    centered_stencil = NewStencil3d{ rect = NewRect2d({0,0},{0,0},{0,0}) }
  end
  return NewAccessPattern {
    write    = params.write,
    read     = params.read,
    centered = true,
    stencil  = centered_stencil,
    field    = params.field
  }
end

-- directly merge new pattern
function AccessPattern:accum(params)
  assert(self._field == params.field,
         'trying to accumulate access patterns for different fields')

  self._phase_write       = self._phase_write   or params.write
  self._phase_read        = self._phase_read    or params.read
  self._phase_reduce      = self._phase_reduce  or (params.reduceop and true)
  if self._phase_reduceop and params.reduceop then
    self._phase_reduceop  = 'multiop'
  elseif params.reduceop then
    self._phase_reduceop  = params.reduceop
  end
  self._centered          = self._centered      and params.centered

  self._stencil           = self._stencil:join(params.stencil)
end

function AccessPattern:join(rhs)
  assert(self._field == rhs._field,
         'trying to join access patterns for different fields')

  error('unimplemented AccessPattern:join() ')
  --NewAccessPattern({
  --
  --}):accum({
  --
  --})
end

function AccessPattern:getstencil()
  return self._stencil
end
function AccessPattern:getfield()
  return self._field
end

-- does this access need exclusive permissions, regardless of whether
-- we have centered access privilleges or not?
function AccessPattern:requiresExclusive()
  return self._phase_write or
         (self._phase_read and self._phase_reduce) or
         self._phase_reduceop == 'multiop'
end

function AccessPattern:isReadOnly()
  return self._phase_read and
         not self._phase_write and
         not self._phase_reduce
end
function AccessPattern:isCentered()
  return self._centered
end
function AccessPattern:reductionOp()
  return self.reduceop
end
function AccessPattern:iserror()
  return not self._centered and self:requiresExclusive()
end

------------------------------------------------------------------------------
--[[ Context:                                                             ]]--
------------------------------------------------------------------------------

local Context = {}
Context.__index = Context

function Context.new()
  local ctxt = setmetatable({
    _env             = terralib.newenvironment(nil),
    _ediag           = terralib.newdiagnostics(),
    _wdiag           = terralib.newdiagnostics(),
    _field_accesses  = {},
  }, Context)
  return ctxt
end
function Context:error(ast, ...)
  self._ediag:reporterror(ast, ...)
end
function Context:warn(ast, ...)
  self._has_warnings = true
  self._wdiag:reporterror(ast, ...)
end
function Context:had_warnings()
  return self._has_warnings
end
function Context:begin()
end
function Context:finish() -- will crash on errors
  --self._wdiag:finish() -- no need to do
  if self._wdiag:haserrors() then
    local warns = self._wdiag:errorlist()
    local flatlist = {"Warnings during stencil checks for Ebb\n"}
    local function rec_ins(x)
      if type(x) == 'table' then
        for i,e in ipairs(x) do rec_ins(e) end
      else table.insert(flatlist, x) end
    end
    rec_ins(self._wdiag:errorlist())
    self._wdiag:clearfilecache()
    print(table.concat(flatlist))
  end
  self._ediag:finishandabortiferrors(
    "Errors during stencil checks for Ebb", 2)
end


function Context:record_key_stencil(key, stencil)
  self._env:localenv()[key] = stencil
end
function Context:lookup_key_stencil(key, stencil)
  return self._env:localenv()[key]
end


function Context:log_access(params)
  local lookup = self._field_accesses[params.field]
  if lookup then
    lookup:accum(params)
  else
    self._field_accesses[params.field] = NewAccessPattern(params)
  end
end

function Context:get_access_patterns()
  return self._field_accesses
end


------------------------------------------------------------------------------
--[[ Stencil Pass:                                                        ]]--
------------------------------------------------------------------------------

function Module.stencilPass(ufunc_ast)
  local ctxt = Context.new()

  -- record the relation being mapped over
  ctxt.relation = ufunc_ast.relation

  ctxt:begin()
    --ufunc_ast:pretty_print()
    ufunc_ast:stencilPass(ctxt)
  ctxt:finish()

  -- if there were no warnings, then the grid-relation accesses should
  -- all have non-trivial stencils; If not, warn the user.
  if not ctxt:had_warnings() and use_exp then
    local aps = ctxt:get_access_patterns()
    local field_warning_list = {}

    for f,ap in pairs(aps) do
      if f:Relation():isGrid() then
        if ap:getstencil():isall() then
          table.insert(field_warning_list,"  "..f:FullName().."\n")
        end
      end
    end

    if #field_warning_list > 0 then 
      print("WARNING: Stencil analysis decided that accesses to the "..
            "following fields could not be bounded/limited:\n"..
            table.concat(field_warning_list)..
            "Therefore, it will not be possible to automatically scale "..
            "this application well in a distributed setting.\n"..
            "  THIS WARNING indicates that either the stencil analysis "..
            "is broken or that the application makes use of features "..
            "that the developers have not yet figured out how to "..
            "scale.\n"..
            "  REGARDLESS, please let the developers know that you saw "..
            "this message so that they can help you resolve the problem.\n")
    end
  end

  return ctxt:get_access_patterns()
end


------------------------------------------------------------------------------
--[[ AST Nodes:                                                           ]]--
------------------------------------------------------------------------------

ast.NewInertPass('stencilPass')

function ast.UserFunction:stencilPass(ctxt)
  local keysym = self.params[1]
  local keytyp = self.ptypes[1]

  if keytyp.ndims == 2 then
    ctxt:record_key_stencil(keysym,
                            NewStencil2d{ rect=NewRect2d({0,0},{0,0}) })
  elseif keytyp.ndims == 3 then
    ctxt:record_key_stencil(keysym,
                            NewStencil3d{ rect=NewRect3d({0,0},{0,0},{0,0}) })
  else
    ctxt:record_key_stencil(keysym, NewStencilAny())
  end

  if self.body then self.body:stencilPass(ctxt) end
  if self.exp then self.exp:stencilPass(ctxt) end
end


-- Name: If this looks up a key, we want to return the corresponding
--       stencil threaded through the symbolic execution here.
function ast.Name:stencilPass(ctxt)
  local stencil = ctxt:lookup_key_stencil(self.name)
  return stencil
end

-- CANNOT assign to key-type variable, so this shouldn't be
-- necessary.
--function ast.Assignment:stencilPass(ctxt)
--end

-- variable declarations are places where we need to record the
-- stencil corresponding a key
function ast.DeclStatement:stencilPass(ctxt)
  if self.node_type:iskey() then
    assert(self.initializer, 'INTERNAL ERROR: declaration of key-type '..
                             'variables should always be initialized.')
    local rhs = self.initializer:stencilPass(ctxt)

    -- sanity check rhs
    if self.node_type:isscalarkey() then
      if self.node_type.ndims == 2 then
        assert(isstencil2d(rhs), 'expected 2d stencil')
      elseif self.node_type.ndims == 3 then
        assert(isstencil3d(rhs), 'expected 3d stencil')
      else
        assert(isstencilany(rhs), 'expected an anystencil')
      end
    else
      -- should be a vector or matrix of keys?
      assert(false, 'unexpected case in stencil analysis')
    end

    ctxt:record_key_stencil(self.name, rhs)
  elseif self.initializer then
    self.initializer:stencilPass(ctxt)
  end
end


function ast.Where:stencilPass(ctxt)
  local key_stencil = self.key:stencilPass(ctxt)
  -- Which field is the index effectively having us read?
  --local keyfield = self.relation:GroupedKeyField()
  local offfield = self.relation:_INTERNAL_GroupedOffset()
  local lenfield = self.relation:_INTERNAL_GroupedLength()
  ctxt:log_access { ast = self, field = offfield, stencil = key_stencil,
                    read = true, centered = self.key.is_centered }
  ctxt:log_access { ast = self, field = lenfield, stencil = key_stencil,
                    read = true, centered = self.key.is_centered }
end
function ast.GenericFor:stencilPass(ctxt)
  self.set:stencilPass(ctxt)

  -- deal with reads implied by projection
  local rel = self.set.node_type.relation
  for i,p in ipairs(self.set.node_type.projections) do
    local field = rel[p]
    rel = field:Type().relation

    ctxt:log_access { ast = self, field = field, stencil = NewStencilAny(),
                      read = true }
  end

  -- note a stencil for the loop variable
  assert(self.node_type:isscalarkey(),
         'REPORT TO DEVELOPERS: '..
         'Should vectors/matrices of keys be returned from query loops?')
  ctxt:record_key_stencil(self.name, unknown_stencil(self.node_type))

  -- process the loop body
  self.body:stencilPass(ctxt)
end

-- VectorLiteral, MatrixLiteral, SquareIndex:
--    For now, just don't track stencils through these constructs
function ast.VectorLiteral:stencilPass(ctxt)
  for i,e in ipairs(self.elems) do e:stencilPass(ctxt) end
  if self.node_type:iskey() then
    ctxt:warn("Stencil analysis does not currently track stencils through "..
              "vector constructors.  Please notify the developers, and "..
              "they can add support.")
  end
end
function ast.MatrixLiteral:stencilPass(ctxt)
  for i,e in ipairs(self.elems) do e:stencilPass(ctxt) end
  if self.node_type:iskey() then
    ctxt:warn("Stencil analysis does not currently track stencils through "..
              "matrix constructors.  Please notify the developers, and "..
              "they can add support.")
  end
end
function ast.SquareIndex:stencilPass(ctxt)
  self.base:stencilPass(ctxt)
  self.index:stencilPass(ctxt)
  if self.index2 then self.index2:stencilPass(ctxt) end

  if self.node_type:iskey() then
    assert(self.node_type:isscalarkey(),
           'should not have nested vectors/matrices of keys')
    return unknown_stencil(self.node_type)
  end
end

-- pass-through
function ast.Quote:stencilPass(ctxt)
  return self.code:stencilPass(ctxt)
end
function ast.LetExpr:stencilPass(ctxt)
  self.block:stencilPass(ctxt)
  return self.exp:stencilPass(ctxt)
end

-- Call : THIS is where Affine transformations are applied/handled
local function is_translation(A)
  local Nrow,Ncol = #A, #A[1]
  -- require that affine maps preserve dimension
  if Ncol ~= Nrow + 1 then return false end

  for i=1,Nrow do for j=1,Nrow do
    if      i==j and A[i][j] ~= 1 then return false
    elseif  i~=j and A[i][j] ~= 0 then return false end
  end end

  return true
end
function ast.Call:stencilPass(ctxt)
  local pstencils = {}
  for i,p in ipairs(self.params) do
    pstencils[i] = p:stencilPass(ctxt)
  end

  if self.func.is_a_terra_func then
    -- actually, this is handled by the above loop...

  elseif B.is_builtin(self.func) then
    if self.func == B.Affine then
      local matrix      = self.params[2].matvals
      local Nrow, Ncol  = #matrix, #matrix[1]

      local argstencil  = pstencils[3]
      --print(isstencil2d(argstencil), isstencil3d(argstencil),
      --      argstencil,
      --      Nrow, Ncol)
      assert( (isstencil2d(argstencil) and Ncol == 3) or
              (isstencil3d(argstencil) and Ncol == 4),
              'INTERNAL: matrix and arg3 dimensions do not match' )

      if not is_translation(matrix) then
        ctxt:error(self, 'The current implementation of stencil analysis '..
                         'only supports translations.  Please complain to '..
                         'the developers if this is a problem for you.')
      end
      local retstencil  = argstencil:affineTransform(matrix)
      return retstencil
    end
  end

  if self.node_type and self.node_type:iskey() then
    assert(self.node_type:isscalarkey(),
           'INTERNAL: not expecting a call to return a non-scalar key')
    return unknown_stencil(self.node_type)
  end
end


-- Various field access forms
function ast.FieldWrite:stencilPass(ctxt)
  -- process the right-hand side
  self.exp:stencilPass(ctxt)
  -- skip over the field access
  local key_stencil = self.fieldaccess.key:stencilPass(ctxt)

  if self.fieldaccess:is(ast.FieldAccessIndex) then
    self.fieldaccess.index:stencilPass(ctxt)
    if self.fieldaccess.index2 then
      self.fieldaccess.index2:stencilPass(ctxt)
    end
    -- then we don't know which of these indices is being accessed?
    -- shouldn't make much of a difference to a coarse analysis, but
    -- we may want to be more granular here in the future.
  end

  ctxt:log_access {
    ast       = self,
    field     = self.fieldaccess.field,
    stencil   = key_stencil,

    write     = not self.reduceop,
    reduceop  = self.reduceop,
    centered  = self.fieldaccess.key.is_centered,
  }
end

function ast.FieldAccessIndex:stencilPass(ctxt)
  self.index:stencilPass(ctxt)
  if self.index2 then self.index2:stencilPass(ctxt) end
  -- may want to be more granular and not just defer here in the future...
  return ast.FieldAccess.stencilPass(self, ctxt)
end

function ast.FieldAccess:stencilPass(ctxt)
  local key_stencil = self.key:stencilPass(ctxt)

  ctxt:log_access {
    ast       = self,
    field     = self.field,
    stencil   = key_stencil,

    read      = true,
    centered  = self.key.is_centered,
  }

  if self.node_type:iskey() then
    assert(self.node_type:isscalarkey(),
           'should not have nested vectors/matrices in a key-field')
    return unknown_stencil(self.node_type)
  end
end




