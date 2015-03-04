local Codegen = {}
package.loaded["compiler.codegen_common"] = Codegen

local ast = require "compiler.ast"


--[[--------------------------------------------------------------------]]--
--[[                 Context Object for Compiler Pass                   ]]--
--[[--------------------------------------------------------------------]]--

-- container class for context attributes specific to the GPU runtime: --
local GPUContext = {}
Codegen.GPUContext = GPUContext
GPUContext.__index = GPUContext

function GPUContext.New (ctxt, block_size)
  return setmetatable({ctxt=ctxt, block_size=block_size}, GPUContext)
end

-- container class that manages extra state needed to support reductions
-- on GPUs
local ReductionCtx = {}
Codegen.ReductionCtx = ReductionCtx
ReductionCtx.__index = ReductionCtx
function ReductionCtx.New (ctxt, block_size)
  local rc = setmetatable({
      ctxt=ctxt,
    },
    ReductionCtx)
    rc:computeGlobalReductionData(block_size)
    return rc
end

-- state should be stored in ctxt.gpu and ctxt.reduce
local Context = {}
Codegen.Context = Context
Context.__index = Context

function Context.new(env, bran)
    local ctxt = setmetatable({
        env  = env,
        bran = bran,
    }, Context)
    return ctxt
end
function Context:initializeGPUState(block_size)
  self.gpu = GPUContext.New(self, block_size)
  self.reduce = ReductionCtx.New(self, self.gpu:blockSize())
end

function Context:localenv()
  return self.env:localenv()
end
function Context:enterblock()
  self.env:enterblock()
end
function Context:leaveblock()
  self.env:leaveblock()
end

function Context:onGPU()
  return self.bran.proc == L.GPU
end

function Context:isElastic()
  return self.bran.is_elastic
end

function Context:dims()
  if not self.bran.dims then self.bran.dims = self.bran.relation:Dims() end
  return self.bran.dims
end

function Context:fieldPhase(field)
  return self.bran.kernel.field_use[field]
end

function Context:globalPhase(global)
  return self.bran.kernel.global_use[global]
end


--[[--------------------------------------------------------------------]]--
--[[                         Utility Functions                          ]]--
--[[--------------------------------------------------------------------]]--


function vec_mapgen(typ,func)
  local arr = {}
  for i=1,typ.N do arr[i] = func(i-1) end
  return `[typ:terraType()]({ array([arr]) })
end
function mat_mapgen(typ,func)
  local rows = {}
  for i=1,typ.Nrow do
    local r = {}
    for j=1,typ.Ncol do r[j] = func(i-1,j-1) end
    rows[i] = `array([r])
  end
  return `[typ:terraType()]({ array([rows]) })
end

function vec_foldgen(N, init, binf)
  local acc = init
  for ii = 1, N do local i = N - ii -- count down to 0
    acc = binf(i, acc) end
  return acc
end
function mat_foldgen(N,M, init, binf)
  local acc = init
  for ii = 1, N do local i = N - ii -- count down to 0
    for jj = 1, M do local j = M - jj -- count down to 0
      acc = binf(i,j, acc) end end
  return acc
end


--[[--------------------------------------------------------------------]]--
--[[                       Codegen Pass Cases                           ]]--
--[[--------------------------------------------------------------------]]--

function ast.AST:codegen (ctxt)
  error("Codegen not implemented for AST node " .. self.kind)
end

function ast.ExprStatement:codegen (ctxt)
  return self.exp:codegen(ctxt)
end

-- complete no-op
function ast.Quote:codegen (ctxt)
  return self.code:codegen(ctxt)
end

function ast.LetExpr:codegen (ctxt)
  ctxt:enterblock()
  local block = self.block:codegen(ctxt)
  local exp   = self.exp:codegen(ctxt)
  ctxt:leaveblock()

  return quote [block] in [exp] end
end

-- DON'T CODEGEN THE KERNEL THIS WAY; HANDLE IN Codegen.codegen()
--function ast.LisztKernel:codegen (ctxt)
--end

function ast.Block:codegen (ctxt)
  -- start with an empty ast node, or we'll get an error when appending new quotes below
  local code = quote end
  for i = 1, #self.statements do
    local stmt = self.statements[i]:codegen(ctxt)
    --print('stmt')
    --self:pretty_print()
    --stmt:printpretty()
    --print ('tmts')
    code = quote code stmt end
  end
  return code
end

function ast.CondBlock:codegen(ctxt, cond_blocks, else_block, index)
  index = index or 1

  local cond  = self.cond:codegen(ctxt)
  ctxt:enterblock()
  local body = self.body:codegen(ctxt)
  ctxt:leaveblock()

  if index == #cond_blocks then
    if else_block then
      return quote if [cond] then [body] else [else_block:codegen(ctxt)] end end
    else
      return quote if [cond] then [body] end end
    end
  else
    ctxt:enterblock()
    local nested = cond_blocks[index + 1]:codegen(ctxt, cond_blocks, else_block, index + 1)
    ctxt:leaveblock()
    return quote if [cond] then [body] else [nested] end end
  end
end

function ast.IfStatement:codegen (ctxt)
  return self.if_blocks[1]:codegen(ctxt, self.if_blocks, self.else_block)
end

function ast.WhileStatement:codegen (ctxt)
  local cond = self.cond:codegen(ctxt)
  ctxt:enterblock()
  local body = self.body:codegen(ctxt)
  ctxt:leaveblock()
  return quote while [cond] do [body] end end
end

function ast.DoStatement:codegen (ctxt)
  ctxt:enterblock()
  local body = self.body:codegen(ctxt)
  ctxt:leaveblock()
  return quote do [body] end end
end

function ast.RepeatStatement:codegen (ctxt)
  ctxt:enterblock()
  local body = self.body:codegen(ctxt)
  local cond = self.cond:codegen(ctxt)
  ctxt:leaveblock()

  return quote repeat [body] until [cond] end
end

function ast.NumericFor:codegen (ctxt)
  -- min and max expression should be evaluated in current scope,
  -- iter expression should be in a nested scope, and for block
  -- should be nested again -- that way the loop var is reset every
  -- time the loop runs.
  local minexp  = self.lower:codegen(ctxt)
  local maxexp  = self.upper:codegen(ctxt)
  local stepexp = self.step and self.step:codegen(ctxt) or nil

  ctxt:enterblock()
  local iterstr = self.name
  local itersym = symbol()
  ctxt:localenv()[iterstr] = itersym

  ctxt:enterblock()
  local body = self.body:codegen(ctxt)
  ctxt:leaveblock()
  ctxt:leaveblock()

  if stepexp then
    return quote for [itersym] = [minexp], [maxexp], [stepexp] do [body] end end
  end

  return quote for [itersym] = [minexp], [maxexp] do [body] end end
end

function ast.Break:codegen(ctxt)
  return quote break end
end

function ast.Name:codegen(ctxt)
  local s = ctxt:localenv()[self.name]
  assert(terralib.issymbol(s))
  return `[s]
end

function ast.Cast:codegen(ctxt)
  local typ = self.node_type
  local bt  = typ:terraBaseType()
  local valuecode = self.value:codegen(ctxt)

  if typ:isPrimitive() then
    return `[typ:terraType()](valuecode)

  elseif typ:isVector() then
    local vec = symbol(self.value.node_type:terraType())
    return quote var [vec] = valuecode in
      [ vec_mapgen(typ, function(i) return `[bt](vec.d[i]) end) ] end

  elseif typ:isMatrix() then
    local mat = symbol(self.value.node_type:terraType())
    return quote var [mat] = valuecode in
      [ mat_mapgen(typ, function(i,j) return `[bt](mat.d[i][j]) end) ] end

  else
    error("Internal Error: Type unrecognized "..typ:toString())
  end
end

-- By the time we make it to codegen, Call nodes are only used to represent builtin function calls.
function ast.Call:codegen (ctxt)
    return self.func.codegen(self, ctxt)
end


function ast.DeclStatement:codegen (ctxt)
  local varname = self.name
  local tp      = self.node_type:terraType()
  local varsym  = symbol(tp)

  if self.initializer then
    local exp = self.initializer:codegen(ctxt)
    ctxt:localenv()[varname] = varsym -- MUST happen after init codegen
    return quote 
      var [varsym] = [exp]
    end
  else
    ctxt:localenv()[varname] = varsym -- MUST happen after init codegen
    return quote var [varsym] end
  end
end

function ast.MatrixLiteral:codegen (ctxt)
  local typ = self.node_type

  return mat_mapgen(typ, function(i,j)
    return self.elems[i*self.m + j + 1]:codegen(ctxt)
  end)
end

function ast.VectorLiteral:codegen (ctxt)
  local typ = self.node_type

  return vec_mapgen(typ, function(i)
    return self.elems[i+1]:codegen(ctxt)
  end)
end

function ast.SquareIndex:codegen (ctxt)
  local base  = self.base:codegen(ctxt)
  local index = self.index:codegen(ctxt)

  -- Vector case
  if self.index2 == nil then
    return `base.d[index]
  -- Matrix case
  else
    local index2 = self.index2:codegen(ctxt)

    return `base.d[index][index2]
  end
end

function ast.Number:codegen (ctxt)
  return `[self.value]
end

function ast.Bool:codegen (ctxt)
  if self.value == true then
    return `true
  else 
    return `false
  end
end


function ast.UnaryOp:codegen (ctxt)
  local expr = self.exp:codegen(ctxt)
  local typ  = self.node_type

  if typ:isPrimitive() then
    if self.op == '-' then return `-[expr]
                      else return `not [expr] end
  elseif typ:isVector() then
    local vec = symbol(typ:terraType())

    if self.op == '-' then
      return quote var [vec] = expr in
        [ vec_mapgen(typ, function(i) return `-vec.d[i] end) ] end
    else
      return quote var [vec] = expr in
        [ vec_mapgen(typ, function(i) return `not vec.d[i] end) ] end
    end
  elseif typ:isMatrix() then
    local mat = symbol(typ:terraType())

    if self.op == '-' then
      return quote var [mat] = expr in
        [ mat_mapgen(typ, function(i,j) return `-mat.d[i][j] end) ] end
    else
      return quote var [mat] = expr in
        [ mat_mapgen(typ, function(i,j) return `not mat.d[i][j] end) ] end
    end

  else
    error("Internal Error: Type unrecognized "..typ:toString())
  end
end

function ast.BinaryOp:codegen (ctxt)
  local lhe = self.lhs:codegen(ctxt)
  local rhe = self.rhs:codegen(ctxt)

  -- handle case of two primitives
  return mat_bin_exp(self.op, self.node_type,
      lhe, rhe, self.lhs.node_type, self.rhs.node_type)
end

function ast.LuaObject:codegen (ctxt)
    return `{}
end

function ast.GenericFor:codegen (ctxt)
    local set       = self.set:codegen(ctxt)
    local iter      = symbol("iter")
    local rel       = self.set.node_type.relation
    -- the key being used to drive the where query should
    -- come from a grouped relation, which is necessarily 1d
    local projected = `[L.addr_terra_types[1]]({array([iter])})

    for i,p in ipairs(self.set.node_type.projections) do
        local field = rel[p]
        projected   = doProjection(projected,field,ctxt)
        rel         = field.type.relation
        assert(rel)
    end
    local sym = symbol(L.key(rel):terraType())
    ctxt:enterblock()
        ctxt:localenv()[self.name] = sym
        local body = self.body:codegen(ctxt)
    ctxt:leaveblock()
    local code = quote
        var s = [set]
        for [iter] = s.start,s.finish do
            var [sym] = [projected]
            [body]
        end
    end
    return code
end

function ast.Assignment:codegen (ctxt)
  local lhs   = self.lvalue:codegen(ctxt)
  local rhs   = self.exp:codegen(ctxt)

  local ltype, rtype = self.lvalue.node_type, self.exp.node_type

  if self.reduceop then
    rhs = mat_bin_exp(self.reduceop, ltype, lhs, rhs, ltype, rtype)
  end
  return quote [lhs] = rhs end
end


local fmin = terralib.externfunction("__nv_fmin", {double,double} -> double)
local fmax = terralib.externfunction("__nv_fmax", {double,double} -> double)


local minexp = macro(function(lhe,rhe)
    if lhe:gettype() == double and L.default_processor == L.GPU then
        return `fmin(lhe,rhe)
    else 
      return quote
        var a = [lhe]
        var b = [rhe]
        var result = a
        if result > b then result = b end
      in
        result
      end
    end
end)

local maxexp = macro(function(lhe,rhe)
    if lhe:gettype() == double and L.default_processor == L.GPU then
        return `fmax(lhe,rhe)
    else 
      return quote
        var a = [lhe]
        var b = [rhe]
        var result = a
        if result < b then result = b end
      in
        result
      end
    end
end)

function bin_exp (op, lhe, rhe)
  if     op == '+'   then return `[lhe] +   [rhe]
  elseif op == '-'   then return `[lhe] -   [rhe]
  elseif op == '/'   then return `[lhe] /   [rhe]
  elseif op == '*'   then return `[lhe] *   [rhe]
  elseif op == '%'   then return `[lhe] %   [rhe]
  elseif op == '^'   then return `[lhe] ^   [rhe]
  elseif op == 'or'  then return `[lhe] or  [rhe]
  elseif op == 'and' then return `[lhe] and [rhe]
  elseif op == '<'   then return `[lhe] <   [rhe]
  elseif op == '>'   then return `[lhe] >   [rhe]
  elseif op == '<='  then return `[lhe] <=  [rhe]
  elseif op == '>='  then return `[lhe] >=  [rhe]
  elseif op == '=='  then return `[lhe] ==  [rhe]
  elseif op == '~='  then return `[lhe] ~=  [rhe]
  elseif op == 'max' then return `maxexp(lhe, rhe)
  elseif op == 'min' then return `minexp(lhe, rhe)
  end
end


function atomic_gpu_red_exp (op, typ, lvalptr, update)
  local internal_error = 'unsupported reduction, internal error; '..
                         'this should be guarded against in the typechecker'
  if typ == L.float then
    if     op == '+'   then return `G.atomic_add_float(lvalptr,  update)
    elseif op == '-'   then return `G.atomic_add_float(lvalptr, -update)
    elseif op == '*'   then return `G.atomic_mul_float_SLOW(lvalptr, update)
    elseif op == '/'   then return `G.atomic_div_float_SLOW(lvalptr, update)
    elseif op == 'min' then return `G.atomic_min_float_SLOW(lvalptr, update)
    elseif op == 'max' then return `G.atomic_max_float_SLOW(lvalptr, update)
    end

  elseif typ == L.double then
    if     op == '+'   then return `G.atomic_add_double_SLOW(lvalptr,  update)
    elseif op == '-'   then return `G.atomic_add_double_SLOW(lvalptr, -update)
    elseif op == '*'   then return `G.atomic_mul_double_SLOW(lvalptr, update)
    elseif op == '/'   then return `G.atomic_div_double_SLOW(lvalptr, update)
    elseif op == 'min' then return `G.atomic_min_double_SLOW(lvalptr, update)
    elseif op == 'max' then return `G.atomic_max_double_SLOW(lvalptr, update)
    end

  elseif typ == L.int then
    if     op == '+'   then return `G.reduce_add_int32(lvalptr,  update)
    elseif op == '-'   then return `G.reduce_add_int32(lvalptr, -update)
    elseif op == '*'   then return `G.atomic_mul_int32_SLOW(lvalptr, update)
    elseif op == 'max' then return `G.reduce_max_int32(lvalptr, update)
    elseif op == 'min' then return `G.reduce_min_int32(lvalptr, update)
    end

  elseif typ == L.bool then
    if     op == 'and' then return `G.reduce_and_b32(lvalptr, update)
    elseif op == 'or'  then return `G.reduce_or_b32(lvalptr, update)
    end
  end
  error(internal_error)
end


-- ONLY ONE PLACE...
function let_vec_binding(typ, N, exp)
  local val = symbol(typ:terraType())
  local let_binding = quote var [val] = [exp] end

  local coords = {}
  if typ:isVector() then
    for i=1, N do coords[i] = `val.d[i-1] end
  else
    for i=1, N do coords[i] = `val end
  end

  return let_binding, coords
end

function let_mat_binding(typ, N, M, exp)
  local val = symbol(typ:terraType())
  local let_binding = quote var [val] = [exp] end

  local coords = {}
  for i = 1, N do
    coords[i] = {}
    for j = 1, M do
      if typ:isMatrix() then
        coords[i][j] = `val.d[i-1][j-1]
      else
        coords[i][j] = `val
      end
    end
  end
  return let_binding, coords
end

function symgen_bind(typ, exp, f)
  local s = symbol(typ:terraType())
  return quote var s = exp in [f(s)] end
end
function symgen_bind2(typ1, typ2, exp1, exp2, f)
  local s1 = symbol(typ1:terraType())
  local s2 = symbol(typ2:terraType())
  return quote
    var s1 = exp1
    var s2 = exp2
  in [f(s1,s2)] end
end

function mat_bin_exp(op, result_typ, lhe, rhe, lhtyp, rhtyp)
  if lhtyp:isPrimitive() and rhtyp:isPrimitive() then
    return bin_exp(op, lhe, rhe)
  end

  -- handles equality and inequality of keys
  if lhtyp:isKey() and rhtyp:isKey() then
    return bin_exp(op, lhe, rhe)
  end

  -- ALL THE CASES

  -- OP: Ord (scalars only)
  -- OP: Mod (scalars only)
  -- BOTH HANDLED ABOVE

  -- OP: Eq (=> DIM: == , BASETYPE: == )
    -- pairwise comparisons, and/or collapse
  local eqinitval = { ['=='] = `true, ['~='] = `false }
  if op == '==' or op == '~=' then
    if lhtyp:isVector() then -- rhtyp:isVector()
      return symgen_bind2(lhtyp, rhtyp, lhe, rhe, function(lvec,rvec)
        return vec_foldgen(lhtyp.N, eqinitval[op], function(i, acc)
          if op == '==' then return `acc and lvec.d[i] == rvec.d[i]
                        else return `acc or  lvec.d[i] ~= rvec.d[i] end
        end) end)

    elseif lhtyp:isMatrix() then -- rhtyp:isMatrix()
      return symgen_bind2(lhtyp, rhtyp, lhe, rhe, function(lmat,rmat)
        return mat_foldgen(lhtyp.Nrow, lhtyp.Ncol, eqinitval[op],
          function(i,j, acc)
            if op == '==' then return `acc and lmat.d[i][j] == rmat.d[i][j]
                          else return `acc or  lmat.d[i][j] ~= rmat.d[i][j] end
          end) end)

    end
  end

  -- OP: Logical (and or)
    -- map the OP
  -- OP: + - min max
    -- map the OP
  if op == 'and'  or op == 'or' or
     op == '+'    or op == '-'  or
     op == 'min'  or op == 'max'
  then
    if lhtyp:isVector() then -- rhtyp:isVector()
      return symgen_bind2(lhtyp, rhtyp, lhe, rhe, function(lvec,rvec)
        return vec_mapgen(result_typ, function(i)
          return bin_exp( op, `(lvec.d[i]), `(rvec.d[i]) ) end) end)

    elseif lhtyp:isMatrix() then
      return symgen_bind2(lhtyp, rhtyp, lhe, rhe, function(lmat,rmat)
        return mat_mapgen(result_typ, function(i,j)
          return bin_exp( op, (`lmat.d[i][j]), `(rmat.d[i][j]) ) end) end)

    end
  end

  -- OP: *
    -- DIM: Scalar _
    -- DIM: _ Scalar
      -- map the OP with expanding one side
  -- OP: /
    -- DIM: _ Scalar
      -- map the OP with expanding one side
  if op == '/' or
    (op == '*' and lhtyp:isPrimitive() or rhtyp:isPrimitive())
  then
    if lhtyp:isVector() then -- rhtyp:isPrimitive()
      return symgen_bind2(lhtyp, rhtyp, lhe, rhe, function(lvec,r)
        return vec_mapgen(result_typ, function(i)
          return bin_exp( op, (`lvec.d[i]), r ) end) end)

    elseif rhtyp:isVector() then -- lhtyp:isPrimitive()
      return symgen_bind2(lhtyp, rhtyp, lhe, rhe, function(l,rvec)
        return vec_mapgen(result_typ, function(i)
          return bin_exp( op, l, `(rvec.d[i]) ) end) end)

    elseif lhtyp:isMatrix() then -- rhtyp:isPrimitive()
      return symgen_bind2(lhtyp, rhtyp, lhe, rhe, function(lmat,r)
        return mat_mapgen(result_typ, function(i,j)
          return bin_exp( op, (`lmat.d[i][j]), r ) end) end)

    elseif rhtyp:isMatrix() then -- rhtyp:isPrimitive()
      return symgen_bind2(lhtyp, rhtyp, lhe, rhe, function(l,rmat)
        return mat_mapgen(result_typ, function(i,j)
          return bin_exp( op, l, `(rmat.d[i][j]) ) end) end)

    end
  end

  -- OP: *
    -- DIM: Vector(n) Matrix(n,_)
    -- DIM: Matrix(_,m) Vector(m)
    -- DIM: Matrix(_,m) Matrix(m,_)
      -- vector-matrix, matrix-vector, or matrix-matrix products
--  if op == '*' then
--    if lhtyp:isVector() and rhtyp:isMatrix() then
--      return symgen_bind2(lhtyp, rhtyp, lhe, rhe, function(lvec,rmat)
--        return vec_mapgen(result_typ, function(j)
--          return vec_foldgen(rmat.Ncol, `0, function(i, acc)
--            return `acc + lvec.d[i] * rmat.d[i][j] end) end) end)
--
--    elseif lhtyp:isMatrix() and rhtyp:isVector() then
--      return symgen_bind2(lhtyp, rhtyp, lhe, rhe, function(lmat,rvec)
--        return vec_mapgen(result_typ, function(i)
--          return vec_foldgen(lmat.Nrow, `0, function(j, acc)
--            return `acc + lmat.d[i][j] * rvec.d[j] end) end) end)
--
--    elseif lhtyp:isMatrix() and rhtyp:isMatrix() then
--      return symgen_bind2(lhtyp, rhtyp, lhe, rhe, function(lmat,rmat)
--        return mat_mapgen(result_typ, function(i,j)
--          return vec_foldgen(rmat.Ncol, `0, function(k, acc)
--            return `acc + lmat.d[i][k] * rmat.d[k][j] end) end) end)
--
--    end
--  end

  -- If we fell through to here we've run into an unhandled branch
  error('Internal Error: Could not find any code to generate for '..
        'binary operator '..op..' with opeands of type '..lhtyp:toString()..
        ' and '..rhtyp:toString())
end
